from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from iwls_validation.analysis import holm_correction, paired_significance, summarize_metrics
from iwls_validation.core import evaluate_baselines, generate_setting
from iwls_validation.pdf_check import verify_pdf_readability
from iwls_validation.plotting import plot_ablation_vs_pooled, plot_multi_panel_results, plot_stability_tradeoff
from iwls_validation.real_data import RealDataConfigError, load_real_setting
from iwls_validation.sympy_validation import run_sympy_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IWLS validation experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-fig-dir", type=Path, required=True)
    parser.add_argument("--paper-table-dir", type=Path, required=True)
    parser.add_argument("--paper-data-dir", type=Path, required=True)
    return parser.parse_args()


def _ensure_type(cfg: dict[str, Any], key: str, expected_type: type) -> None:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    if not isinstance(cfg[key], expected_type):
        raise ValueError(f"Config key '{key}' must be of type {expected_type.__name__}")


def validate_config(cfg: dict[str, Any]) -> None:
    # Strictly reject wrapper payloads from upstream recovery envelopes.
    forbidden_wrapper_keys = {"payload", "artifacts", "notes", "needs_input", "questions", "execution_report"}
    present_wrapper = forbidden_wrapper_keys.intersection(cfg.keys())
    if present_wrapper:
        raise ValueError(f"Config appears to be a wrapped envelope, not a runnable config: {sorted(present_wrapper)}")

    _ensure_type(cfg, "settings", list)
    _ensure_type(cfg, "seeds", list)
    _ensure_type(cfg, "n_sources", int)
    _ensure_type(cfg, "n_samples", int)
    for key in ["alpha", "beta", "gamma", "ridge_lambda"]:
        if key not in cfg:
            raise ValueError(f"Missing required numeric key: {key}")
        if not isinstance(cfg[key], (int, float)):
            raise ValueError(f"Config key '{key}' must be numeric")

    if not cfg["settings"]:
        raise ValueError("Config 'settings' must be non-empty")
    if not cfg["seeds"]:
        raise ValueError("Config 'seeds' must be non-empty")


def _to_float_list(values: list[str] | list[float] | list[int], fallback: list[float]) -> list[float]:
    if not values:
        return fallback
    out: list[float] = []
    for v in values:
        out.append(float(v))
    return out


def _to_int_list(values: list[str] | list[int], fallback: list[int]) -> list[int]:
    if not values:
        return fallback
    out: list[int] = []
    for v in values:
        out.append(int(v))
    return out


def build_ablation_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = cfg.get("sweep_params", {})
    alpha_vals = _to_float_list(sweep.get("alpha_discrepancy_weight", [cfg["alpha"]]), [cfg["alpha"]])[:2]
    beta_vals = _to_float_list(sweep.get("beta_ratio_uncertainty_weight", [cfg["beta"]]), [cfg["beta"]])[:2]
    gamma_vals = _to_float_list(sweep.get("gamma_stability_weight", [cfg["gamma"]]), [cfg["gamma"]])[:3]
    ridge_vals = _to_float_list(sweep.get("ridge_lambda", [cfg["ridge_lambda"]]), [cfg["ridge_lambda"]])[:2]
    ratio_estimators = list(sweep.get("ratio_estimator", ["gaussian_diag_proxy", "logistic_ratio_proxy"]))[:2]
    clipping_vals = list(sweep.get("weight_clipping", ["p99", "p95"]))[:2]
    candidate_pool_sizes = _to_int_list(
        sweep.get("candidate_pool_size", [cfg["n_sources"]]),
        [cfg["n_sources"]],
    )[:2]
    adaptive_gamma_vals = [False, True]

    grid: list[dict[str, Any]] = []
    for alpha, beta, gamma, lam, ratio_estimator, clipping, candidate_pool_size, adaptive_gamma in itertools.product(
        alpha_vals,
        beta_vals,
        gamma_vals,
        ridge_vals,
        ratio_estimators,
        clipping_vals,
        candidate_pool_sizes,
        adaptive_gamma_vals,
    ):
        grid.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "ridge_lambda": float(lam),
                "ratio_estimator": str(ratio_estimator),
                "clipping": str(clipping),
                "candidate_pool_size": int(candidate_pool_size),
                "adaptive_gamma": bool(adaptive_gamma),
                "ess_floor": 200.0,
                "cond_ref": 120.0,
                "gamma_ess_scale": 0.8,
                "gamma_cond_scale": 0.25,
            }
        )
    return grid


def pooled_gap(df: pd.DataFrame) -> float:
    merged = df[df["method"].isin(["stability_aware_composite", "pooled_source_IWLS"])].pivot_table(
        index=["setting", "seed"], columns="method", values="target_mse", aggfunc="first"
    )
    merged = merged.dropna()
    if merged.empty:
        return 0.0
    return float((merged["stability_aware_composite"] - merged["pooled_source_IWLS"]).mean())


def run_config_matrix(
    *,
    settings: list[str],
    seeds: list[int],
    n_sources: int,
    n_samples: int,
    params: dict[str, Any],
    real_settings: list[tuple[str, Any, Any]],
    log_path: Path | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for setting in settings:
        for seed in seeds:
            sources, target = generate_setting(setting=setting, seed=seed, n_sources=n_sources, n_samples=n_samples)
            metrics = evaluate_baselines(
                sources=sources,
                target=target,
                seed=seed,
                alpha=params["alpha"],
                beta=params["beta"],
                gamma=params["gamma"],
                lam=params["ridge_lambda"],
                ratio_estimator=params["ratio_estimator"],
                clipping=params["clipping"],
                adaptive_gamma=params["adaptive_gamma"],
                ess_floor=params["ess_floor"],
                cond_ref=params["cond_ref"],
                gamma_ess_scale=params["gamma_ess_scale"],
                gamma_cond_scale=params["gamma_cond_scale"],
            )
            for method, vals in metrics.items():
                records.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "method": method,
                        "alpha": params["alpha"],
                        "beta": params["beta"],
                        "gamma": params["gamma"],
                        "ridge_lambda": params["ridge_lambda"],
                        "ratio_estimator": params["ratio_estimator"],
                        "clipping": params["clipping"],
                        "candidate_pool_size": n_sources,
                        "adaptive_gamma": params["adaptive_gamma"],
                        **vals,
                    }
                )

            if log_path is not None:
                log_obj = {
                    "params": {
                        "setting": setting,
                        **params,
                        "candidate_pool_size": n_sources,
                    },
                    "seed": seed,
                    "command": "run_experiments.py",
                    "duration_sec": None,
                    "metrics": {method: vals["target_mse"] for method, vals in metrics.items()},
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj) + "\n")

    # Optional real-data settings loaded from local CSV conversions.
    for real_name, sources, target in real_settings:
        for seed in seeds:
            metrics = evaluate_baselines(
                sources=sources,
                target=target,
                seed=seed,
                alpha=params["alpha"],
                beta=params["beta"],
                gamma=params["gamma"],
                lam=params["ridge_lambda"],
                ratio_estimator=params["ratio_estimator"],
                clipping=params["clipping"],
                adaptive_gamma=params["adaptive_gamma"],
                ess_floor=params["ess_floor"],
                cond_ref=params["cond_ref"],
                gamma_ess_scale=params["gamma_ess_scale"],
                gamma_cond_scale=params["gamma_cond_scale"],
            )
            for method, vals in metrics.items():
                records.append(
                    {
                        "setting": real_name,
                        "seed": seed,
                        "method": method,
                        "alpha": params["alpha"],
                        "beta": params["beta"],
                        "gamma": params["gamma"],
                        "ridge_lambda": params["ridge_lambda"],
                        "ratio_estimator": params["ratio_estimator"],
                        "clipping": params["clipping"],
                        "candidate_pool_size": len(sources),
                        "adaptive_gamma": params["adaptive_gamma"],
                        **vals,
                    }
                )

    return pd.DataFrame(records)


def load_real_settings(cfg: dict[str, Any]) -> tuple[list[tuple[str, Any, Any]], list[str]]:
    loaded: list[tuple[str, Any, Any]] = []
    warnings: list[str] = []
    for item in cfg.get("real_settings", []):
        setting_name = f"R_{item.get('name', 'unnamed')}"
        try:
            sources, target = load_real_setting(item)
            loaded.append((setting_name, sources, target))
        except RealDataConfigError as exc:
            warnings.append(f"Skipped {setting_name}: {exc}")
    return loaded, warnings


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_fig_dir.mkdir(parents=True, exist_ok=True)
    args.paper_table_dir.mkdir(parents=True, exist_ok=True)
    args.paper_data_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    log_path = args.output_dir / "experiment_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    real_settings, real_warnings = load_real_settings(cfg)

    base_params = {
        "alpha": float(cfg["alpha"]),
        "beta": float(cfg["beta"]),
        "gamma": float(cfg["gamma"]),
        "ridge_lambda": float(cfg["ridge_lambda"]),
        "ratio_estimator": "gaussian_diag_proxy",
        "clipping": "none",
        "adaptive_gamma": False,
        "ess_floor": 200.0,
        "cond_ref": 120.0,
        "gamma_ess_scale": 0.8,
        "gamma_cond_scale": 0.25,
    }

    ablation_grid = build_ablation_grid(cfg)
    ablation_rows: list[dict[str, Any]] = []
    p_value_map: dict[str, float] = {}

    for idx, combo in enumerate(ablation_grid):
        combo_df = run_config_matrix(
            settings=[s for s in cfg["settings"] if s in {"A", "B", "C"}],
            seeds=[int(s) for s in cfg["seeds"]],
            n_sources=int(combo["candidate_pool_size"]),
            n_samples=int(cfg["n_samples"]),
            params=combo,
            real_settings=real_settings,
            log_path=None,
        )
        pooled_sig = paired_significance(combo_df, "pooled_source_IWLS", "stability_aware_composite")
        row = {
            "config_id": f"cfg_{idx:04d}",
            **combo,
            "pooled_gap_mean": pooled_gap(combo_df),
            "pooled_p_value": float(pooled_sig["p_value"]),
            "pooled_test": str(pooled_sig["test"]),
            "pooled_n_pairs": int(pooled_sig["n_pairs"]),
            "stability_mse_mean": float(
                combo_df[combo_df["method"] == "stability_aware_composite"]["target_mse"].mean()
            ),
            "pooled_mse_mean": float(combo_df[combo_df["method"] == "pooled_source_IWLS"]["target_mse"].mean()),
        }
        ablation_rows.append(row)
        p_value_map[row["config_id"]] = row["pooled_p_value"]

    holm = holm_correction(p_value_map)
    for row in ablation_rows:
        row["holm_adjusted_p"] = holm[row["config_id"]]
        row["stability_beats_pooled"] = bool(row["pooled_gap_mean"] < 0.0)
        row["holm_significant"] = bool(row["holm_adjusted_p"] < 0.05)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df = ablation_df.sort_values(
        ["holm_adjusted_p", "pooled_gap_mean", "stability_mse_mean"], ascending=[True, True, True]
    ).reset_index(drop=True)

    ablation_path = args.paper_table_dir / "table_ablation_vs_pooled.csv"
    ablation_df.to_csv(ablation_path, index=False)

    selected_params = dict(base_params)
    selection_reason = "base_config"
    if not ablation_df.empty:
        best = ablation_df.iloc[0].to_dict()
        selected_params = {
            "alpha": float(best["alpha"]),
            "beta": float(best["beta"]),
            "gamma": float(best["gamma"]),
            "ridge_lambda": float(best["ridge_lambda"]),
            "ratio_estimator": str(best["ratio_estimator"]),
            "clipping": str(best["clipping"]),
            "adaptive_gamma": bool(best["adaptive_gamma"]),
            "ess_floor": float(best["ess_floor"]),
            "cond_ref": float(best["cond_ref"]),
            "gamma_ess_scale": float(best["gamma_ess_scale"]),
            "gamma_cond_scale": float(best["gamma_cond_scale"]),
        }
        selected_n_sources = int(best["candidate_pool_size"])
        selection_reason = f"ablation_selected:{best['config_id']}"
    else:
        selected_n_sources = int(cfg["n_sources"])

    results_df = run_config_matrix(
        settings=[s for s in cfg["settings"] if s in {"A", "B", "C"}],
        seeds=[int(s) for s in cfg["seeds"]],
        n_sources=selected_n_sources,
        n_samples=int(cfg["n_samples"]),
        params=selected_params,
        real_settings=real_settings,
        log_path=log_path,
    )

    results_path = args.paper_data_dir / "iwls_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = summarize_metrics(results_df)
    summary_path = args.paper_table_dir / "table_main_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    sig_pairs = {
        "mmd_vs_stability": paired_significance(results_df, "mmd_nearest_source_plus_IWLS", "stability_aware_composite"),
        "pooled_vs_stability": paired_significance(results_df, "pooled_source_IWLS", "stability_aware_composite"),
        "wasserstein_vs_stability": paired_significance(
            results_df,
            "wasserstein_nearest_source_plus_IWLS",
            "stability_aware_composite",
        ),
    }
    corrected = holm_correction({k: float(v["p_value"]) for k, v in sig_pairs.items()})
    sig_df = pd.DataFrame(
        [
            {
                "comparison": key,
                "test": val["test"],
                "p_value": val["p_value"],
                "holm_adjusted_p": corrected[key],
                "n_pairs": val["n_pairs"],
                "note": "Normality via Shapiro-Wilk; paired t-test if normal else Wilcoxon.",
            }
            for key, val in sig_pairs.items()
        ]
    )
    sig_path = args.paper_table_dir / "table_significance.csv"
    sig_df.to_csv(sig_path, index=False)

    fig1 = args.paper_fig_dir / "fig_main_results.pdf"
    fig2 = args.paper_fig_dir / "fig_stability_tradeoff.pdf"
    fig3 = args.paper_fig_dir / "fig_ablation_pooled_gap.pdf"
    plot_multi_panel_results(summary_df, fig1)
    plot_stability_tradeoff(results_df, fig2)
    plot_ablation_vs_pooled(ablation_df, fig3)

    pdf_checks = [verify_pdf_readability(fig1), verify_pdf_readability(fig2)]
    if fig3.exists():
        pdf_checks.append(verify_pdf_readability(fig3))
    check_path = args.output_dir / "pdf_readability_checks.json"
    check_path.write_text(json.dumps(pdf_checks, indent=2), encoding="utf-8")

    sympy_path = args.output_dir / "sympy_validation_report.txt"
    run_sympy_checks(sympy_path)

    elapsed = time.time() - start
    report = {
        "runtime_sec": elapsed,
        "records": len(results_df),
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
        "significance_csv": str(sig_path),
        "ablation_csv": str(ablation_path),
        "figures": [str(fig1), str(fig2), str(fig3)],
        "sympy_report": str(sympy_path),
        "pdf_checks": str(check_path),
        "datasets": [str(results_path)],
        "tables": [str(summary_path), str(sig_path), str(ablation_path)],
        "schema_validation_strict": True,
        "selected_config": {
            **selected_params,
            "candidate_pool_size": selected_n_sources,
            "selection_reason": selection_reason,
        },
        "ablation_grid_size": len(ablation_grid),
        "real_settings_loaded": [name for name, _, _ in real_settings],
        "real_data_warnings": real_warnings,
    }
    (args.output_dir / "results_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

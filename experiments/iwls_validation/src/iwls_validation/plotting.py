from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")


def pretty_method_name(method: str) -> str:
    mapping = {
        "stability_aware_composite": "Stability composite",
        "random_source_plus_IWLS": "Random + IWLS",
        "mmd_nearest_source_plus_IWLS": "MMD-nearest + IWLS",
        "wasserstein_nearest_source_plus_IWLS": "Wasserstein-nearest + IWLS",
        "pooled_source_IWLS": "Pooled IWLS",
        "single_source_unweighted_LS": "Single-source unweighted LS",
        "mixed_shift_gate_plus_composite": "Mixed-shift gate + composite",
        "oracle_best_source_retrospective": "Oracle best source (retro.)",
    }
    return mapping.get(method, method.replace("_", " "))


def _set_external_legend(ax: plt.Axes, *, ncol: int = 1, fontsize: int = 10) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.33),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
    )


def plot_multi_panel_results(summary_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    panel_settings = ["A", "B", "C"]
    for idx, setting in enumerate(panel_settings):
        ax = axes[idx]
        sdf = summary_df[summary_df["setting"] == setting].copy()
        sdf = sdf.sort_values("target_mse_mean", ascending=True)
        x = range(len(sdf))
        y = sdf["target_mse_mean"]
        yerr_low = y - sdf["target_mse_ci_low"]
        yerr_high = sdf["target_mse_ci_high"] - y
        method_labels = [pretty_method_name(m) for m in sdf["method"]]

        ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            capsize=4,
            label="Mean ± 95% CI",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(method_labels, rotation=24, ha="right", fontsize=10)
        ax.set_ylabel("Target MSE (squared error units)")
        ax.set_xlabel("Method")
        ax.set_title(f"Setting {setting}")
        _set_external_legend(ax)
        ax.margins(x=0.03)

    fig.suptitle(
        "Caption: Cross-setting comparison of IWLS source selection methods with 95% CI across five seeds",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def plot_stability_tradeoff(results_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    plot_df = results_df.copy()
    plot_df["method_label"] = plot_df["method"].map(pretty_method_name)

    ax0 = axes[0]
    sns.scatterplot(
        data=plot_df,
        x="ess",
        y="target_mse",
        hue="method_label",
        style="setting",
        ax=ax0,
    )
    ax0.set_xlabel("Effective sample size (ESS)")
    ax0.set_ylabel("Target MSE (squared error units)")
    ax0.set_title("ESS vs Target Error")
    handles0, labels0 = ax0.get_legend_handles_labels()
    if ax0.legend_ is not None:
        ax0.legend_.remove()

    ax1 = axes[1]
    sns.scatterplot(
        data=plot_df,
        x="condition_number",
        y="target_mse",
        hue="method_label",
        style="setting",
        ax=ax1,
    )
    ax1.set_xlabel("Condition number κ(XᵀWX)")
    ax1.set_ylabel("Target MSE (squared error units)")
    ax1.set_title("Conditioning vs Target Error")
    if ax1.legend_ is not None:
        ax1.legend_.remove()

    fig.legend(
        handles0,
        labels0,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=9,
        title="Legend",
        frameon=True,
    )

    fig.suptitle(
        "Caption: Stability diagnostics versus target performance for all settings and baselines",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def plot_ablation_vs_pooled(ablation_df: pd.DataFrame, out_pdf: Path) -> None:
    if ablation_df.empty:
        return

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    ax0 = axes[0]
    sns.scatterplot(
        data=ablation_df,
        x="pooled_gap_mean",
        y="holm_adjusted_p",
        hue="ratio_estimator",
        style="clipping",
        size="gamma",
        ax=ax0,
    )
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax0.axhline(0.05, color="black", linestyle=":", linewidth=1.0)
    ax0.set_xlabel("Mean MSE gap: stability - pooled (lower is better)")
    ax0.set_ylabel("Holm-adjusted p-value")
    ax0.set_title("Ablation significance vs pooled IWLS")
    ax0.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
    )

    ax1 = axes[1]
    top = ablation_df.nsmallest(12, columns=["holm_adjusted_p", "pooled_gap_mean"]).copy()
    top["label"] = "a=" + top["alpha"].astype(str) + ", b=" + top["beta"].astype(str) + ", g=" + top["gamma"].astype(str)
    top["label"] += "\nclip=" + top["clipping"].astype(str) + ", est=" + top["ratio_estimator"].astype(str)
    sns.barplot(data=top, x="pooled_gap_mean", y="label", hue="adaptive_gamma", ax=ax1)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Mean MSE gap: stability - pooled")
    ax1.set_ylabel("Top ablation configurations")
    ax1.set_title("Best pooled-gap configurations")
    ax1.tick_params(axis="y", labelsize=10)
    ax1.legend(loc="upper right", fontsize=10, frameon=True, title="Adaptive gamma")

    fig.suptitle("Caption: Pooled-IWLS-focused ablation over stability calibration controls", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

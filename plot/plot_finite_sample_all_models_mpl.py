from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import FINITE_SAMPLE_RUN, MODEL_ORDER, OUTPUT_DIR, ensure_output_dir, short_model


INPUT_PATH = FINITE_SAMPLE_RUN / "finite_sample_overall_summary.csv"
OUTPUT_PATH = OUTPUT_DIR / "appendix_experiment3_finite_sample_all_models_mpl.png"


def hoeffding_radius(k: np.ndarray, delta: float = 0.05) -> np.ndarray:
    return np.sqrt(np.log(2 / delta) / (2 * k))


def main() -> None:
    ensure_output_dir()
    df = pd.read_csv(INPUT_PATH)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "K"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=180, sharex=True)
    palette = plt.get_cmap("tab10")

    for idx, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model]
        linestyle = "-" if model.endswith("-Instruct") else "--"
        label = short_model(model)
        color = palette(idx)
        axes[0].plot(sub["K"], sub["ci_width_95"] / 2, marker="o", linewidth=1.7, linestyle=linestyle, color=color, label=label)
        axes[1].plot(sub["K"], sub["mu_ci_width_95"] / 2, marker="o", linewidth=1.7, linestyle=linestyle, color=color, label=label)

    k_values = np.array(sorted(df["K"].unique()))
    smooth_k = np.geomspace(k_values.min(), k_values.max(), 240)
    axes[0].plot(smooth_k, hoeffding_radius(smooth_k), color="#111", linewidth=2.2, linestyle=":", label="Hoeffding radius")

    axes[0].set_title(r"Sign statistic: empirical half-width vs. Hoeffding")
    axes[1].set_title(r"Margin statistic: empirical half-width")
    axes[0].set_ylabel("Empirical 95% half-width")
    axes[1].set_ylabel("Empirical 95% half-width")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Number of distinct pairs K")
        ax.grid(True, color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0, 0.20)
    axes[1].set_ylim(0, 0.32)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("Appendix: finite-sample behavior for all models", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

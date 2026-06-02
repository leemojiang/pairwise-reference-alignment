from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_plot import FINITE_SAMPLE_RUN, OUTPUT_DIR, ensure_output_dir, short_model


INPUT_PATH = FINITE_SAMPLE_RUN / "finite_sample_overall_summary.csv"
OUTPUT_PATH = OUTPUT_DIR / "experiment3_finite_sample_representative_mpl.png"
SELECTED_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B-Instruct"]


def hoeffding_radius(k: np.ndarray, delta: float = 0.05) -> np.ndarray:
    return np.sqrt(np.log(2 / delta) / (2 * k))


def main() -> None:
    ensure_output_dir()
    df = pd.read_csv(INPUT_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180, sharex=True)
    palette = plt.get_cmap("tab10")

    for idx, model in enumerate(SELECTED_MODELS):
        sub = df[df["model"] == model].sort_values("K")
        axes[0].plot(sub["K"], sub["ci_width_95"] / 2, marker="o", linewidth=2.0, color=palette(idx), label=short_model(model))
        axes[1].plot(sub["K"], sub["mu_ci_width_95"] / 2, marker="o", linewidth=2.0, color=palette(idx), label=short_model(model))

    k_values = np.array(sorted(df["K"].unique()))
    smooth_k = np.geomspace(k_values.min(), k_values.max(), 240)
    axes[0].plot(smooth_k, hoeffding_radius(smooth_k), color="#111111", linewidth=2.3, linestyle=":", label="Hoeffding radius")

    axes[0].set_title(r"Sign statistic")
    axes[1].set_title(r"Margin statistic")
    axes[0].set_ylabel("Empirical 95% half-width")
    axes[1].set_ylabel("Empirical 95% half-width")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Number of distinct pairs K")
        ax.grid(True, color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Finite-sample stability with Hoeffding reference", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

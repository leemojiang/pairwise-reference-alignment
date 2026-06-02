from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import BOOTSTRAP_RUN, MODEL_ORDER, OUTPUT_DIR, ensure_output_dir, model_kind, model_size


# This file is generated from the plain-prompt raw scores by the bootstrap
# experiment, so it provides both the plain overall estimates and CI bars.
INPUT_PATH = BOOTSTRAP_RUN / "bootstrap_overall_ci.csv"
OUTPUT_PATH = OUTPUT_DIR / "experiment1_overall_bars_mpl.png"


def main() -> None:
    ensure_output_dir()
    df = pd.read_csv(INPUT_PATH)
    df["size"] = df["model"].map(model_size)
    df["kind"] = df["model"].map(model_kind)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model")

    sizes = ["0.5B", "1.5B", "3B", "7B"]
    x = np.arange(len(sizes))
    width = 0.34
    colors = {"Base": "#4C78A8", "Instruct": "#F58518"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=180)
    panels = [
        ("full_A_hat", "bootstrap_ci_low_2p5", "bootstrap_ci_high_97p5", r"$\hat{A}_M$", "Sign agreement"),
        ("full_mu_hat", "bootstrap_mu_ci_low_2p5", "bootstrap_mu_ci_high_97p5", r"$\hat{\mu}_M$", "Signed margin"),
    ]

    for ax, (value_col, low_col, high_col, ylabel, title) in zip(axes, panels):
        for offset, kind in [(-width / 2, "Base"), (width / 2, "Instruct")]:
            sub = df[df["kind"] == kind].set_index("size").loc[sizes]
            values = sub[value_col].to_numpy()
            lows = values - sub[low_col].to_numpy()
            highs = sub[high_col].to_numpy() - values
            ax.bar(x + offset, values, width, label=kind, color=colors[kind], alpha=0.88)
            ax.errorbar(x + offset, values, yerr=[lows, highs], fmt="none", ecolor="#222", elinewidth=1.1, capsize=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, sizes)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0.55, 0.82)
    axes[1].set_ylim(0.04, 0.40)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Overall likelihood-induced alignment on RewardBench", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

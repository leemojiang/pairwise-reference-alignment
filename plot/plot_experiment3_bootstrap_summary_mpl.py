from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import BOOTSTRAP_RUN, MODEL_ORDER, OUTPUT_DIR, ensure_output_dir, short_model


INPUT_PATH = BOOTSTRAP_RUN / "bootstrap_overall_ci.csv"
OUTPUT_PATH = OUTPUT_DIR / "experiment3_bootstrap_summary_mpl.png"


def main() -> None:
    ensure_output_dir()
    df = pd.read_csv(INPUT_PATH)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model")

    labels = [short_model(m) for m in df["model"].astype(str)]
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), dpi=180, sharex=True)
    panels = [
        ("full_A_hat", "bootstrap_ci_low_2p5", "bootstrap_ci_high_97p5", r"$\hat{A}_M$ with bootstrap CI"),
        ("full_mu_hat", "bootstrap_mu_ci_low_2p5", "bootstrap_mu_ci_high_97p5", r"$\hat{\mu}_M$ with bootstrap CI"),
    ]
    colors = ["#4C78A8", "#F58518"]

    for ax, (value_col, low_col, high_col, title), color in zip(axes, panels, colors):
        values = df[value_col].to_numpy()
        lows = values - df[low_col].to_numpy()
        highs = df[high_col].to_numpy() - values
        ax.errorbar(x, values, yerr=[lows, highs], fmt="o", color=color, ecolor="#333", capsize=4, markersize=5)
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("Bootstrap confidence intervals for all models", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

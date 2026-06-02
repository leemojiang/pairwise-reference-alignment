from __future__ import annotations

from plot_subset_family_radar_mpl import GROUPS, grouped_scores

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import CHAT_RUN, MODEL_ORDER, OUTPUT_DIR, ensure_output_dir, short_model


INPUT_PATH = CHAT_RUN / "metrics_by_subset.csv"
OUTPUT_PATH = OUTPUT_DIR / "appendix_chattemplate_subset_family_radar_mpl.png"


def main() -> None:
    ensure_output_dir()
    df = grouped_scores(pd.read_csv(INPUT_PATH))
    labels = [name for name, _ in GROUPS]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10.5, 9), dpi=180)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    palette = plt.get_cmap("tab10")
    for idx, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("group").loc[labels]
        values = sub["A_hat"].tolist()
        values += values[:1]
        linestyle = "-" if model.endswith("-Instruct") else "--"
        ax.plot(angles, values, linewidth=1.8, linestyle=linestyle, color=palette(idx), label=short_model(model))
        ax.fill(angles, values, color=palette(idx), alpha=0.035)

    ax.set_title("Appendix: chat-template subset-family radar", y=1.08, fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.12), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

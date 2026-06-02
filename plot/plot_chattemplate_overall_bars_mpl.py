from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import CHAT_RUN, MODEL_ORDER, OUTPUT_DIR, ensure_output_dir, model_kind, model_size


INPUT_PATH = CHAT_RUN / "metrics_overall.csv"
OUTPUT_PATH = OUTPUT_DIR / "appendix_chattemplate_overall_bars_mpl.png"


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
        ("A_hat", r"$\hat{A}_M$", "Sign agreement"),
        ("mu_hat", r"$\hat{\mu}_M$", "Signed margin"),
    ]

    for ax, (value_col, ylabel, title) in zip(axes, panels):
        for offset, kind in [(-width / 2, "Base"), (width / 2, "Instruct")]:
            sub = df[df["kind"] == kind].set_index("size").loc[sizes]
            ax.bar(x + offset, sub[value_col].to_numpy(), width, label=kind, color=colors[kind], alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, sizes)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0.55, max(0.82, df["A_hat"].max() * 1.08))
    axes[1].set_ylim(0.05, df["mu_hat"].max() * 1.12)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Appendix: chat-template likelihood-induced alignment", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot import MODEL_ORDER, OUTPUT_DIR, PLAIN_RUN, ensure_output_dir, short_model


INPUT_PATH = PLAIN_RUN / "metrics_by_subset.csv"
OUTPUT_PATH = OUTPUT_DIR / "experiment2_subset_family_radar_mpl.png"

GROUPS = [
    ("AlpacaEval", lambda s: s.startswith("alpacaeval")),
    ("LLMBar", lambda s: s.startswith("llmbar")),
    ("MT-Bench", lambda s: s.startswith("mt-bench")),
    ("Safety", lambda s: s.startswith("refusals") or s == "donotanswer" or s.startswith("xstest")),
    ("HEP-Code", lambda s: s.startswith("hep-")),
    ("Math", lambda s: s == "math-prm"),
]


def grouped_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]
        for group_name, predicate in GROUPS:
            group_df = model_df[model_df["subset"].map(predicate)]
            total_k = group_df["K"].sum()
            a_hat = (group_df["A_hat"] * group_df["K"]).sum() / total_k
            rows.append({"model": model, "group": group_name, "A_hat": a_hat})
    return pd.DataFrame(rows)


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

    ax.set_title("Subset-family reference distribution dependence", y=1.08, fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.12), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

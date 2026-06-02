from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path("remote_results/pairwise-reference-alignment-code")
SUBSET_METRICS = DATA_ROOT / "runs/20260528-092001_experiment1and2_qwen_rewardbench_experiment12/metrics_by_subset.csv"
OUTPUT_DIR = Path("../Draft/Experiments-Result/plots")
OUTPUT_PATH = OUTPUT_DIR / "experiment2_subset_radar.svg"

WIDTH = 980
HEIGHT = 820
CX = 430
CY = 390
R = 255

GROUPS = [
    ("AlpacaEval", lambda s: s.startswith("alpacaeval")),
    ("LLMBar", lambda s: s.startswith("llmbar")),
    ("MT-Bench", lambda s: s.startswith("mt-bench")),
    ("Safety", lambda s: s.startswith("refusals") or s == "donotanswer" or s.startswith("xstest")),
    ("HEP-Code", lambda s: s.startswith("hep-")),
    ("Math", lambda s: s == "math-prm"),
]

COLORS = {
    "Qwen2.5-0.5B": "#4C78A8",
    "Qwen2.5-0.5B-Instruct": "#72B7B2",
    "Qwen2.5-1.5B": "#54A24B",
    "Qwen2.5-1.5B-Instruct": "#B279A2",
    "Qwen2.5-3B": "#F58518",
    "Qwen2.5-3B-Instruct": "#FF9DA6",
    "Qwen2.5-7B": "#E45756",
    "Qwen2.5-7B-Instruct": "#9D755D",
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short_model(model: str) -> str:
    return model.split("/")[-1]


def point(axis_idx: int, value: float, n_axes: int) -> tuple[float, float]:
    angle = -math.pi / 2 + 2 * math.pi * axis_idx / n_axes
    radius = R * value
    return CX + radius * math.cos(angle), CY + radius * math.sin(angle)


def read_grouped_scores() -> dict[str, list[float]]:
    weighted: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    with SUBSET_METRICS.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subset = row["subset"]
            model = short_model(row["model"])
            for group_name, predicate in GROUPS:
                if predicate(subset):
                    k = float(row["K"])
                    weighted[model][group_name][0] += float(row["A_hat"]) * k
                    weighted[model][group_name][1] += k
                    break
    result: dict[str, list[float]] = {}
    for model, groups in weighted.items():
        values = []
        for group_name, _ in GROUPS:
            total, k = groups[group_name]
            values.append(total / k if k else 0.0)
        result[model] = values
    return result


def main() -> None:
    scores = read_grouped_scores()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_axes = len(GROUPS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text{font-family:Arial, Helvetica, sans-serif; fill:#222}",
        ".title{font-size:22px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#555}",
        ".label{font-size:13px;font-weight:700}",
        ".tick{font-size:11px;fill:#666}",
        ".grid{fill:none;stroke:#d6d6d6;stroke-width:1}",
        ".axis{stroke:#c8c8c8;stroke-width:1}",
        "</style>",
        f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" class="title">Subset-level reference distribution dependence</text>',
        f'<text x="{WIDTH / 2}" y="58" text-anchor="middle" class="subtitle">Weighted mean A_hat over RewardBench subset families. Larger radius means stronger likelihood-preference agreement.</text>',
    ]

    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = [point(i, level, n_axes) for i in range(n_axes)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polygon points="{path}" class="grid"/>')
        x, y = point(0, level, n_axes)
        parts.append(f'<text x="{x + 6:.1f}" y="{y + 4:.1f}" class="tick">{level:.1f}</text>')

    for i, (label, _) in enumerate(GROUPS):
        x, y = point(i, 1.08, n_axes)
        x0, y0 = point(i, 1.0, n_axes)
        parts.append(f'<line x1="{CX}" y1="{CY}" x2="{x0:.1f}" y2="{y0:.1f}" class="axis"/>')
        anchor = "middle"
        if x < CX - 20:
            anchor = "end"
        elif x > CX + 20:
            anchor = "start"
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" class="label">{esc(label)}</text>')

    for model in sorted(scores, key=lambda name: (name.replace("-Instruct", ""), "Instruct" not in name)):
        vals = scores[model]
        pts = [point(i, value, n_axes) for i, value in enumerate(vals)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        color = COLORS.get(model, "#333")
        width = 2.4 if model.endswith("Instruct") else 1.6
        dash = "" if model.endswith("Instruct") else ' stroke-dasharray="5 4"'
        parts.append(f'<polygon points="{path}" fill="{color}" fill-opacity="0.045" stroke="{color}" stroke-width="{width}"{dash}/>')

    legend_x = 720
    legend_y = 132
    parts.append(f'<text x="{legend_x}" y="{legend_y - 24}" class="label">Models</text>')
    for idx, model in enumerate(sorted(scores, key=lambda name: (name.replace("-Instruct", ""), "Instruct" not in name))):
        y = legend_y + idx * 24
        color = COLORS.get(model, "#333")
        dash = "" if model.endswith("Instruct") else ' stroke-dasharray="5 4"'
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="2.2"{dash}/>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 4}" class="tick">{esc(model)}</text>')

    parts.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

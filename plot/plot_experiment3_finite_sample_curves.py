from __future__ import annotations

import csv
from pathlib import Path


DATA_ROOT = Path("remote_results/pairwise-reference-alignment-code")
FINITE_SAMPLE = DATA_ROOT / "runs/20260528-114143_experiment4_qwen_rewardbench_finite_sample/finite_sample_overall_summary.csv"
OUTPUT_DIR = Path("../Draft/Experiments-Result/plots")
OUTPUT_PATH = OUTPUT_DIR / "experiment3_finite_sample_curves.svg"

WIDTH = 1100
HEIGHT = 640
MARGIN = 76
PANEL_GAP = 70
PANEL_W = (WIDTH - 2 * MARGIN - PANEL_GAP) / 2
PANEL_H = 400
PANEL_TOP = 100
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B-Instruct"]
COLORS = {"Qwen/Qwen2.5-0.5B": "#4C78A8", "Qwen/Qwen2.5-7B-Instruct": "#E45756"}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_rows() -> list[dict[str, str]]:
    with FINITE_SAMPLE.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sx(k: float, left: float) -> float:
    # log10 scale from 50 to 2000.
    import math

    lo, hi = math.log10(50), math.log10(2000)
    return left + (math.log10(k) - lo) / (hi - lo) * PANEL_W


def sy(value: float, ymax: float, top: float) -> float:
    return top + (ymax - value) / ymax * PANEL_H


def draw_panel(rows: list[dict[str, str]], left: float, field: str, title: str, ymax: float) -> list[str]:
    out: list[str] = []
    top = PANEL_TOP
    out.append(f'<text x="{left + PANEL_W / 2:.1f}" y="{top - 30}" text-anchor="middle" class="title">{esc(title)}</text>')
    out.append(f'<line x1="{left}" y1="{top + PANEL_H}" x2="{left + PANEL_W}" y2="{top + PANEL_H}" class="axis"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + PANEL_H}" class="axis"/>')
    for i in range(6):
        value = ymax * i / 5
        y = sy(value, ymax, top)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + PANEL_W}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    for k in [50, 100, 200, 500, 1000, 2000]:
        x = sx(k, left)
        out.append(f'<text x="{x:.1f}" y="{top + PANEL_H + 28}" text-anchor="middle" class="tick">{k}</text>')
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        points = [(sx(float(row["K"]), left), sy(float(row[field]), ymax, top)) for row in model_rows]
        color = COLORS[model]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    return out


def main() -> None:
    rows = read_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text{font-family:Arial, Helvetica, sans-serif; fill:#222}",
        ".title{font-size:19px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#555}",
        ".tick{font-size:12px;fill:#555}",
        ".axis{stroke:#333;stroke-width:1.2}",
        ".grid{stroke:#ddd;stroke-width:0.8}",
        "</style>",
        f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" class="title">Finite-sample stability curves</text>',
        f'<text x="{WIDTH / 2}" y="58" text-anchor="middle" class="subtitle">Empirical 95% subsampling interval widths shrink as the number of distinct pairs K grows.</text>',
    ]
    parts.extend(draw_panel(rows, MARGIN, "ci_width_95", "Sign agreement interval width", 0.30))
    parts.extend(draw_panel(rows, MARGIN + PANEL_W + PANEL_GAP, "mu_ci_width_95", "Margin interval width", 0.60))
    lx = WIDTH / 2 - 170
    ly = HEIGHT - 50
    for idx, model in enumerate(MODELS):
        x = lx + idx * 250
        parts.append(f'<line x1="{x}" y1="{ly}" x2="{x + 30}" y2="{ly}" stroke="{COLORS[model]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{x + 38}" y="{ly + 4}" class="tick">{esc(model.split("/")[-1])}</text>')
    parts.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

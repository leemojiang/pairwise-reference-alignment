from __future__ import annotations

import csv
from pathlib import Path


DATA_ROOT = Path("remote_results/pairwise-reference-alignment-code")
BOOTSTRAP_CI = DATA_ROOT / "runs/20260528-114147_experiment4b_qwen_rewardbench_bootstrap/bootstrap_overall_ci.csv"
OUTPUT_DIR = Path("../Draft/Experiments-Result/plots")
OUTPUT_PATH = OUTPUT_DIR / "experiment3_bootstrap_ci_widths.svg"

WIDTH = 1120
HEIGHT = 620
MARGIN = 82
PANEL_GAP = 68
PANEL_W = (WIDTH - 2 * MARGIN - PANEL_GAP) / 2
PANEL_H = 360
PANEL_TOP = 100


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_rows() -> list[dict[str, str]]:
    with BOOTSTRAP_CI.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_model(model: str) -> str:
    return model.split("/")[-1].replace("Qwen2.5-", "")


def sy(value: float, ymax: float, top: float) -> float:
    return top + (ymax - value) / ymax * PANEL_H


def draw_panel(rows: list[dict[str, str]], left: float, field: str, title: str, ymax: float, color: str) -> list[str]:
    out: list[str] = []
    top = PANEL_TOP
    bar_w = PANEL_W / len(rows) * 0.62
    out.append(f'<text x="{left + PANEL_W / 2:.1f}" y="{top - 30}" text-anchor="middle" class="title">{esc(title)}</text>')
    out.append(f'<line x1="{left}" y1="{top + PANEL_H}" x2="{left + PANEL_W}" y2="{top + PANEL_H}" class="axis"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + PANEL_H}" class="axis"/>')
    for i in range(6):
        value = ymax * i / 5
        y = sy(value, ymax, top)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + PANEL_W}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for idx, row in enumerate(rows):
        x = left + PANEL_W * (idx + 0.5) / len(rows)
        value = float(row[field])
        y = sy(value, ymax, top)
        y0 = sy(0, ymax, top)
        out.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{y0 - y:.1f}" fill="{color}" opacity="0.86"/>')
        out.append(f'<text x="{x:.1f}" y="{top + PANEL_H + 22}" text-anchor="middle" transform="rotate(35 {x:.1f} {top + PANEL_H + 22})" class="tick">{esc(short_model(row["model"]))}</text>')
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
        ".tick{font-size:11px;fill:#555}",
        ".axis{stroke:#333;stroke-width:1.2}",
        ".grid{stroke:#ddd;stroke-width:0.8}",
        "</style>",
        f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" class="title">Bootstrap uncertainty at full RewardBench size</text>',
        f'<text x="{WIDTH / 2}" y="58" text-anchor="middle" class="subtitle">CI widths are small overall, but margin intervals are wider than sign-agreement intervals.</text>',
    ]
    parts.extend(draw_panel(rows, MARGIN, "bootstrap_ci_width_95", "A_hat bootstrap CI width", 0.030, "#4C78A8"))
    parts.extend(draw_panel(rows, MARGIN + PANEL_W + PANEL_GAP, "bootstrap_mu_ci_width_95", "mu_hat bootstrap CI width", 0.065, "#F58518"))
    parts.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

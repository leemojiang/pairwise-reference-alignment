from __future__ import annotations

import csv
from pathlib import Path


DATA_ROOT = Path("remote_results/pairwise-reference-alignment-code")
BOOTSTRAP_CI = DATA_ROOT / "runs/20260528-114147_experiment4b_qwen_rewardbench_bootstrap/bootstrap_overall_ci.csv"
OUTPUT_DIR = Path("../Draft/Experiments-Result/plots")
OUTPUT_PATH = OUTPUT_DIR / "experiment1_overall_bars.svg"


WIDTH = 1180
HEIGHT = 680
MARGIN = 78
PANEL_GAP = 72
PANEL_W = (WIDTH - 2 * MARGIN - PANEL_GAP) / 2
PANEL_H = 430
PANEL_TOP = 92


COLORS = {
    "base": "#4C78A8",
    "instruct": "#F58518",
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_rows() -> list[dict[str, str]]:
    with BOOTSTRAP_CI.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_parts(model: str) -> tuple[str, str]:
    name = model.split("/")[-1]
    is_instruct = name.endswith("-Instruct")
    size = name.replace("Qwen2.5-", "").replace("-Instruct", "")
    return size, "instruct" if is_instruct else "base"


def y_scale(value: float, ymin: float, ymax: float, top: float, height: float) -> float:
    return top + (ymax - value) / (ymax - ymin) * height


def draw_panel(rows: list[dict[str, str]], metric: str, left: float, title: str, y_label: str, ymin: float, ymax: float) -> list[str]:
    out: list[str] = []
    top = PANEL_TOP
    height = PANEL_H
    sizes = ["0.5B", "1.5B", "3B", "7B"]
    by_key = {model_parts(row["model"]): row for row in rows}
    group_w = PANEL_W / len(sizes)
    bar_w = 24
    tick_count = 5

    out.append(f'<text x="{left + PANEL_W / 2:.1f}" y="{top - 34}" text-anchor="middle" class="title">{esc(title)}</text>')
    out.append(f'<text x="{left - 52}" y="{top + height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left - 52} {top + height / 2:.1f})" class="axis-label">{esc(y_label)}</text>')
    out.append(f'<line x1="{left}" y1="{top + height}" x2="{left + PANEL_W}" y2="{top + height}" class="axis"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>')

    for i in range(tick_count + 1):
        value = ymin + (ymax - ymin) * i / tick_count
        y = y_scale(value, ymin, ymax, top, height)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + PANEL_W}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

    for idx, size in enumerate(sizes):
        center = left + group_w * (idx + 0.5)
        out.append(f'<text x="{center:.1f}" y="{top + height + 28}" text-anchor="middle" class="tick">{size}</text>')
        for offset, kind in [(-bar_w * 0.65, "base"), (bar_w * 0.65, "instruct")]:
            row = by_key[(size, kind)]
            if metric == "A":
                value = float(row["full_A_hat"])
                low = float(row["bootstrap_ci_low_2p5"])
                high = float(row["bootstrap_ci_high_97p5"])
            else:
                value = float(row["full_mu_hat"])
                low = float(row["bootstrap_mu_ci_low_2p5"])
                high = float(row["bootstrap_mu_ci_high_97p5"])
            x = center + offset
            y = y_scale(value, ymin, ymax, top, height)
            y0 = y_scale(0 if ymin <= 0 else ymin, ymin, ymax, top, height)
            h = y0 - y
            out.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[kind]}" opacity="0.88"/>')
            y_low = y_scale(low, ymin, ymax, top, height)
            y_high = y_scale(high, ymin, ymax, top, height)
            out.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" class="error"/>')
            out.append(f'<line x1="{x - 7:.1f}" y1="{y_low:.1f}" x2="{x + 7:.1f}" y2="{y_low:.1f}" class="error"/>')
            out.append(f'<line x1="{x - 7:.1f}" y1="{y_high:.1f}" x2="{x + 7:.1f}" y2="{y_high:.1f}" class="error"/>')
    return out


def main() -> None:
    rows = read_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text{font-family:Arial, Helvetica, sans-serif; fill:#222}",
        ".title{font-size:20px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#555}",
        ".tick{font-size:12px;fill:#555}",
        ".axis-label{font-size:14px;font-weight:700;fill:#333}",
        ".axis{stroke:#333;stroke-width:1.2}",
        ".grid{stroke:#ddd;stroke-width:0.8}",
        ".error{stroke:#222;stroke-width:1.2}",
        "</style>",
        f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" class="title">Overall likelihood-induced alignment on RewardBench</text>',
        f'<text x="{WIDTH / 2}" y="58" text-anchor="middle" class="subtitle">Bars show full-sample estimates; whiskers show 95% bootstrap percentile intervals.</text>',
    ]
    parts.extend(draw_panel(rows, "A", MARGIN, "Sign agreement", "A_hat", 0.55, 0.82))
    parts.extend(draw_panel(rows, "mu", MARGIN + PANEL_W + PANEL_GAP, "Signed margin", "mu_hat", 0.04, 0.40))
    legend_y = HEIGHT - 58
    legend_x = WIDTH / 2 - 110
    parts.append(f'<rect x="{legend_x}" y="{legend_y - 13}" width="18" height="18" fill="{COLORS["base"]}" opacity="0.88"/>')
    parts.append(f'<text x="{legend_x + 26}" y="{legend_y + 1}" class="tick">Base</text>')
    parts.append(f'<rect x="{legend_x + 90}" y="{legend_y - 13}" width="18" height="18" fill="{COLORS["instruct"]}" opacity="0.88"/>')
    parts.append(f'<text x="{legend_x + 116}" y="{legend_y + 1}" class="tick">Instruct</text>')
    parts.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

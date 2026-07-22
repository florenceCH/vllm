#!/usr/bin/env python3
"""2026 open-source LLM timeline chart for PPT (15cm x 9cm)."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# fmt: off
MODELS = [
    {"name": "MiniMax M2.1",   "date": "2025-12-23", "params": 229,  "score": 31, "vendor": "MiniMax"},
    {"name": "Kimi K2.5",      "date": "2026-01-27", "params": 1000, "score": 35, "vendor": "Kimi"},
    {"name": "Qwen3.5-35B",    "date": "2026-02-24", "params": 35,   "score": 29, "vendor": "Qwen"},
    {"name": "MiniMax M2.5",   "date": "2026-02-12", "params": 230,  "score": 34, "vendor": "MiniMax"},
    {"name": "Qwen3.5-122B",   "date": "2026-03-04", "params": 122,  "score": 32, "vendor": "Qwen"},
    {"name": "MiniMax M2.7",   "date": "2026-03-18", "params": 230,  "score": 38, "vendor": "MiniMax"},
    {"name": "Qwen3.5-397B",   "date": "2026-03-25", "params": 397,  "score": 34, "vendor": "Qwen"},
    {"name": "Qwen3.6-27B",    "date": "2026-04-22", "params": 27,   "score": 37, "vendor": "Qwen"},
    {"name": "Qwen3.6-35B",    "date": "2026-04-22", "params": 35,   "score": 32, "vendor": "Qwen"},
    {"name": "GLM-5",          "date": "2026-04-14", "params": 744,  "score": 40, "vendor": "GLM"},
    {"name": "GLM-5.1",        "date": "2026-04-29", "params": 744,  "score": 40, "vendor": "GLM"},
    {"name": "DeepSeek V4-Flash","date": "2026-04-29", "params": 284, "score": 40, "vendor": "DeepSeek"},
    {"name": "MiniMax M3",     "date": "2026-06-04", "params": 428,  "score": 44, "vendor": "MiniMax"},
    {"name": "DeepSeek V4-Pro", "date": "2026-06-04", "params": 1600, "score": 44, "vendor": "DeepSeek"},
    {"name": "Kimi K2.7",      "date": "2026-06-12", "params": 1000, "score": 42, "vendor": "Kimi"},
    {"name": "GLM-5.2",        "date": "2026-06-13", "params": 744,  "score": 51, "vendor": "GLM"},
    {"name": "Kimi K3",        "date": "2026-07-16", "params": 2800, "score": 57, "vendor": "Kimi"},
]
# fmt: on

VENDOR_COLORS = {
    "MiniMax": "#E74C3C",
    "DeepSeek": "#3498DB",
    "Qwen": "#9B59B6",
    "Kimi": "#2ECC71",
    "GLM": "#F39C12",
}

LABEL_FONTSIZE = 7.2
LABEL_PAD = 5.0
MAX_OFFSET_PT = 46
NUDGE_STEP = 2
NUDGE_STEPS = 4
AXES_PAD = 10
POINT_PAD = 6.0

# 经碰撞搜索后的最终位置：贴点、不压其它圆点、不出界
FIXED_LABELS = {
    "MiniMax M2.1": (10, -20, "left", "top"),
    "Kimi K2.5": (18, 0, "left", "center"),
    "Qwen3.5-35B": (0, -14, "center", "top"),
    "MiniMax M2.5": (-20, -14, "right", "top"),
    "Qwen3.5-122B": (-18, 14, "right", "bottom"),
    "MiniMax M2.7": (0, 12, "center", "bottom"),
    "Qwen3.5-397B": (18, 0, "left", "center"),
    "Qwen3.6-27B": (-12, 0, "right", "center"),
    "Qwen3.6-35B": (12, 0, "left", "center"),
    "GLM-5": (-18, 0, "right", "center"),
    "GLM-5.1": (10, 8, "left", "bottom"),
    "DeepSeek V4-Flash": (22, -18, "left", "top"),
    "MiniMax M3": (22, -18, "left", "top"),
    "DeepSeek V4-Pro": (-10, 2, "right", "bottom"),
    "Kimi K2.7": (-16, -14, "right", "top"),
    "GLM-5.2": (26, -20, "left", "top"),
    "Kimi K3": (-12, -14, "right", "top"),
}


DISPLAY_X_DAYS = {
    "Qwen3.6-27B": -5,
    "Qwen3.6-35B": 5,
    "MiniMax M3": -6,
    "DeepSeek V4-Pro": 5,
    "Kimi K2.7": -5,
    "GLM-5.2": 7,
}


def display_dates(dates, y_vals):
    """为重叠密集区的模型施加小幅横向偏移，便于标签避让。"""
    out = []
    for i, dt in enumerate(dates):
        shift = DISPLAY_X_DAYS.get(MODELS[i]["name"], 0)
        out.append(dt + timedelta(days=shift))
    return out


def params_to_y(p):
    if p <= 30:
        return 0.06 + (p / 30) * 0.10
    if p <= 200:
        return 0.16 + ((p - 30) / 170) * 0.18
    if p <= 400:
        return 0.34 + ((p - 200) / 200) * 0.28
    if p <= 500:
        return 0.62 + ((p - 400) / 100) * 0.05
    if p <= 1000:
        return 0.67 + ((p - 500) / 500) * 0.11
    return 0.78 + min((p - 1000) / 1800, 1.0) * 0.18


Y_TICK_PARAMS = [30, 50, 100, 200, 400, 500, 750, 1000, 1500, 2500]


def format_y_tick(p):
    if p >= 2500:
        return "2.5T+"
    if p >= 1000:
        v = p / 1000
        return f"{v:.1f}T".replace(".0T", "T")
    return f"{p}B"


def fmt_params(p):
    if p >= 1000:
        v = p / 1000
        return f"{v:.1f}T".replace(".0T", "T")
    return f"{p}B"


def label_text(m):
    return f"{m['name']}\n{fmt_params(m['params'])} · {m['score']}分"


def inflate_bbox(bbox, pad=LABEL_PAD):
    return (bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad)


def bboxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def candidate_offsets():
    return [
        (14, 10, "left", "bottom"),
        (-14, 10, "right", "bottom"),
        (14, -10, "left", "top"),
        (-14, -10, "right", "top"),
        (18, 0, "left", "center"),
        (-18, 0, "right", "center"),
        (0, 14, "center", "bottom"),
        (0, -14, "center", "top"),
        (24, 14, "left", "bottom"),
        (-24, 14, "right", "bottom"),
        (24, -14, "left", "top"),
        (-24, -14, "right", "top"),
        (32, 8, "left", "bottom"),
        (-32, 8, "right", "bottom"),
        (32, -8, "left", "top"),
        (-32, -8, "right", "top"),
        (0, 22, "center", "bottom"),
        (0, -22, "center", "top"),
        (40, 0, "left", "center"),
        (-40, 0, "right", "center"),
        (44, 16, "left", "bottom"),
        (-44, 16, "right", "bottom"),
        (44, -16, "left", "top"),
        (-44, -16, "right", "top"),
        (52, 0, "left", "center"),
        (-52, 0, "right", "center"),
    ]


def approx_text_box(ax, xy, dx, dy, ha, va, text):
    px, py = ax.transData.transform((mdates.date2num(xy[0]), xy[1]))
    scale = ax.figure.dpi / 72.0
    ox = px + dx * scale
    oy = py + dy * scale
    lines = text.split("\n")
    max_chars = max(len(line) for line in lines)
    char_w = LABEL_FONTSIZE * 0.62
    width = max_chars * char_w
    height = len(lines) * LABEL_FONTSIZE * 1.2
    if ha == "left":
        x0, x1 = ox, ox + width
    elif ha == "right":
        x0, x1 = ox - width, ox
    else:
        x0, x1 = ox - width / 2, ox + width / 2
    if va == "bottom":
        y0, y1 = oy, oy + height
    elif va == "top":
        y0, y1 = oy - height, oy
    else:
        y0, y1 = oy - height / 2, oy + height / 2
    pad = LABEL_PAD + 2
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def annotation_bbox_for_offset(
    fig, ax, renderer, xy, text, dx, dy, ha, va, color="#222", size=0.0
):
    bubble_r = np.sqrt(size / np.pi) if size else 0.0
    ann = ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=LABEL_FONTSIZE,
        ha=ha,
        va=va,
        color=color,
        linespacing=1.15,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=0.55,
            alpha=0.65,
            shrinkA=max(3, bubble_r * 0.5),
            shrinkB=2,
        ),
        annotation_clip=False,
    )
    bbox = ann.get_window_extent(renderer)
    ann.remove()
    return bbox


def offset_len(dx, dy):
    return float(np.hypot(dx, dy))


def axes_bbox(ax, pad=AXES_PAD):
    bb = ax.get_window_extent()
    return (bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def bbox_inside(inner, outer):
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def all_point_boxes(ax, points, sizes, pad=POINT_PAD):
    boxes = []
    for xy, size in zip(points, sizes):
        px, py = ax.transData.transform((mdates.date2num(xy[0]), xy[1]))
        r = np.sqrt(size / np.pi) + pad
        boxes.append((px - r, py - r, px + r, py + r))
    return boxes


def label_conflicts(box, point_boxes, placed, axes_box, own_idx=None):
    if not bbox_inside(box, axes_box):
        return True
    for idx, pb in enumerate(point_boxes):
        if idx == own_idx:
            continue
        if bboxes_overlap(box, pb):
            return True
    if any(bboxes_overlap(box, p) for p in placed):
        return True
    return False


def expand_preference(dx, dy, ha, va):
    variants = [(dx, dy, ha, va)]
    for k in range(1, NUDGE_STEPS + 1):
        scale = 1.0 + k * (NUDGE_STEP / max(offset_len(dx, dy), 1.0))
        ndx = dx * scale if dx else 0.0
        ndy = dy * scale if dy else 0.0
        if offset_len(ndx, ndy) > MAX_OFFSET_PT:
            break
        variants.append((ndx, ndy, ha, va))
    for ndx, ndy, nha, nva in list(variants):
        for sdx, sdy in ((NUDGE_STEP, 0), (-NUDGE_STEP, 0), (0, NUDGE_STEP), (0, -NUDGE_STEP)):
            tx, ty = ndx + sdx, ndy + sdy
            if offset_len(tx, ty) <= MAX_OFFSET_PT:
                variants.append((tx, ty, nha, nva))
    return variants


def grid_candidates():
    cands = []
    for dx in range(-46, 47, 2):
        for dy in range(-46, 47, 2):
            dist = offset_len(dx, dy)
            if dist < 12 or dist > MAX_OFFSET_PT:
                continue
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            cands.append((float(dx), float(dy), ha, va))
    return sorted(cands, key=lambda t: offset_len(t[0], t[1]))


def all_candidates(name):
    seen = set()
    cands = []
    if name in FIXED_LABELS:
        dx, dy, ha, va = FIXED_LABELS[name]
        for item in expand_preference(dx, dy, ha, va):
            key = (round(item[0], 1), round(item[1], 1), item[2], item[3])
            if key not in seen:
                seen.add(key)
                cands.append(item)
    for item in grid_candidates():
        key = (item[0], item[1], item[2], item[3])
        if key not in seen:
            seen.add(key)
            cands.append(item)
    return cands


def pick_label_position(
    fig, ax, renderer, xy, size, text, color, point_boxes, placed, axes_box, name, own_idx
):
    approx_hits = []
    for rank, (dx, dy, ha, va) in enumerate(all_candidates(name)):
        box = approx_text_box(ax, xy, dx, dy, ha, va, text)
        if label_conflicts(box, point_boxes, placed, axes_box, own_idx):
            continue
        approx_hits.append((offset_len(dx, dy) + rank * 0.15, dx, dy, ha, va))
        if len(approx_hits) >= 30:
            break

    best = None
    best_score = float("inf")
    for score, dx, dy, ha, va in approx_hits:
        bbox = annotation_bbox_for_offset(
            fig, ax, renderer, xy, text, dx, dy, ha, va, color, size
        )
        box = inflate_bbox(bbox)
        if label_conflicts(box, point_boxes, placed, axes_box, own_idx):
            continue
        if score < best_score:
            best_score = score
            best = (dx, dy, ha, va, box)

    if best:
        return best

    for dx, dy, ha, va in all_candidates(name):
        bbox = annotation_bbox_for_offset(
            fig, ax, renderer, xy, text, dx, dy, ha, va, color, size
        )
        box = inflate_bbox(bbox)
        if not label_conflicts(box, point_boxes, placed, axes_box, own_idx):
            return dx, dy, ha, va, box

    dx, dy, ha, va = all_candidates(name)[0]
    bbox = annotation_bbox_for_offset(
        fig, ax, renderer, xy, text, dx, dy, ha, va, color, size
    )
    return dx, dy, ha, va, inflate_bbox(bbox)


def place_labels(ax, fig, models, points, texts, sizes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_boxes = all_point_boxes(ax, points, sizes)
    axes_box = axes_bbox(ax)
    placed = []
    order = sorted(
        range(len(points)),
        key=lambda i: (points[i][1], mdates.date2num(points[i][0])),
        reverse=True,
    )

    for i in order:
        m = models[i]
        xy = points[i]
        text = texts[i]
        color = VENDOR_COLORS[m["vendor"]]
        dx, dy, ha, va, box = pick_label_position(
            fig,
            ax,
            renderer,
            xy,
            sizes[i],
            text,
            color,
            point_boxes,
            placed,
            axes_box,
            m["name"],
            i,
        )
        placed.append(box)
        bubble_r = np.sqrt(sizes[i] / np.pi)
        ax.annotate(
            text,
            xy=xy,
            xycoords="data",
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=LABEL_FONTSIZE,
            ha=ha,
            va=va,
            color=color,
            linespacing=1.15,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.55,
                alpha=0.65,
                shrinkA=max(3, bubble_r * 0.5),
                shrinkB=2,
            ),
            annotation_clip=True,
            zorder=6,
        )


def main():
    dates = [datetime.strptime(m["date"], "%Y-%m-%d") for m in MODELS]
    y_vals = [params_to_y(m["params"]) for m in MODELS]
    plot_dates = display_dates(dates, y_vals)
    scores = [m["score"] for m in MODELS]
    colors = [VENDOR_COLORS[m["vendor"]] for m in MODELS]

    fig_w, fig_h = 15 / 2.54, 9 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=400)

    sizes = [(s - 26) ** 1.5 * 7 for s in scores]

    for i, m in enumerate(MODELS):
        ax.scatter(
            plot_dates[i],
            y_vals[i],
            s=sizes[i],
            c=colors[i],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

    points = list(zip(plot_dates, y_vals))
    texts = [label_text(m) for m in MODELS]

    ax.set_xlim(datetime(2025, 12, 15), datetime(2026, 7, 25))
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0, interval=2))

    yticks = [params_to_y(p) for p in Y_TICK_PARAMS]
    ylabels = [format_y_tick(p) for p in Y_TICK_PARAMS]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)

    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7)
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)

    ax.set_xlabel("发布日期", fontsize=8.5, labelpad=4)
    ax.set_ylabel("模型总参数量", fontsize=8.5, labelpad=4)
    ax.set_title(
        "2026年开源大模型发布全景图",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    legend_handles = [
        mpatches.Patch(color=c, label=v) for v, c in VENDOR_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    note = (
        "圆点大小 = 综合能力评分 (Artificial Analysis Intelligence Index v4.1)  ·  "
        "数据来源: HuggingFace / 官方发布"
    )
    fig.text(0.5, -0.055, note, ha="center", fontsize=5.5, color="#666")

    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.14)

    place_labels(ax, fig, MODELS, points, texts, sizes)

    out = "/workspace/open_source_models_timeline.png"
    out_latest = "/workspace/open_source_models_timeline_latest.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_latest, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")
    print(f"Saved: {out_latest}")


if __name__ == "__main__":
    main()

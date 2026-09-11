# --- path bootstrap (added during repo restructure: shared code lives in core/ and tools/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while True:
    if _os.path.isdir(_os.path.join(_ROOT, "core")):
        break
    _parent = _os.path.dirname(_ROOT)
    if _parent == _ROOT:
        break
    _ROOT = _parent
for _sub in ("core", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
# -*- coding: utf-8 -*-
"""plot_head2head_8figs.py -- 从 head2head.json 生成 8 张图(绿=champion2, 蓝=opv3)。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_ROOT, "data", "runs", "a2", "head2head.json")
OUT = r"D:\AIs\PINN\reports\figs\head2head_8figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

GREEN = "tab:green"
BLUE = "tab:blue"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
labels = sorted(results, key=lambda k: results[k]["q"])
qs = [results[k]["q"] for k in labels]
heldout = [results[k]["heldout"] for k in labels]
xt = [f"{q:g}" for q in qs]
x = range(len(labels))

C2 = "champion2"
V3 = "opv3"


def bars(vals_c2, vals_v3, title, ylabel, fname, logy=False):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.4
    ax.bar([i - w / 2 for i in x], vals_c2, width=w, color=GREEN,
           label=C2, hatch="//" if True else None, edgecolor="white")
    ax.bar([i + w / 2 for i in x], vals_v3, width=w, color=BLUE,
           label=V3, edgecolor="white")
    for i, h in enumerate(heldout):
        if h:
            ax.axvspan(i - 0.5, i + 0.5, color="0.55", alpha=0.25, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(xt, rotation=45)
    ax.set_xlabel("q")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title + "  (阴影=heldout)")
    ax.legend(loc="best")
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def get(key):
    return [results[k][C2][key] for k in labels], [results[k][V3][key] for k in labels]


written = []
c2, v3 = get("peak_l2re")
written.append(bars(c2, v3, "峰值盒 L2RE (r<2.7)", "L2RE", "fig1_peak_box_l2re.png"))

c2, v3 = get("ring_l2re")
written.append(bars(c2, v3, "环区 L2RE (r<1.0)", "L2RE", "fig2_ring_l2re.png"))

c2, v3 = get("bulk_l2re")
written.append(bars(c2, v3, "整体 bulk L2RE", "L2RE", "fig3_bulk_l2re.png"))

c2, v3 = get("ax_l2re")
written.append(bars(c2, v3, "轴向窗 L2RE [0.05,1.5]", "L2RE", "fig4_axial_l2re.png"))

c2, v3 = get("peak_height_ratio")
written.append(bars(c2, v3, "峰高比 h/h_ref", "ratio", "fig5_peak_height_ratio.png"))

c2, v3 = get("ax_maxerr")
written.append(bars(c2, v3, "轴向最大误差", "abs err", "fig6_ax_maxerr.png"))

# fig7: 散点 bulk vs peak, 两条系列
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter([results[k][C2]["bulk_l2re"] for k in labels],
           [results[k][C2]["peak_l2re"] for k in labels],
           color=GREEN, s=70, label=C2)
ax.scatter([results[k][V3]["bulk_l2re"] for k in labels],
           [results[k][V3]["peak_l2re"] for k in labels],
           color=BLUE, s=70, label=V3)
lim = [4e-3, 4e-2]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("bulk L2RE")
ax.set_ylabel("peak L2RE")
ax.set_title("bulk vs peak (对角线上方=峰值误差更大)")
ax.legend()
fig.tight_layout()
p = os.path.join(OUT, "fig7_bulk_vs_peak_scatter.png")
fig.savefig(p, dpi=150)
plt.close(fig)
written.append(p)

# fig8: peak/bulk 误差比
ratio_c2 = [results[k][C2]["peak_l2re"] / results[k][C2]["bulk_l2re"] for k in labels]
ratio_v3 = [results[k][V3]["peak_l2re"] / results[k][V3]["bulk_l2re"] for k in labels]
written.append(bars(ratio_c2, ratio_v3, "峰值/整体 误差比", "peak / bulk", "fig8_peak_over_bulk.png"))

print("已生成:")
for p in written:
    print(" ", p)

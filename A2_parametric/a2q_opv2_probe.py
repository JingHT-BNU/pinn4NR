"""a2q_opv2_probe.py —— 算子 v2 修正场空间结构探针。

背景(报告 §5.3):v1 算子退化为常数幅度重标定,|corr_eff| p50=p90=p99≈0.21
恒定、零空间结构。v2 的成功判据(其一;另一为 a2q_eval L2RE 突破 3e-2 地板):
  ψ = corr_eff = (u/(κ·u_g) − 1)/w 出现真实空间结构:
    ① 分位数展开:强修正区(w>0.3)内 ψ 的相对 IQR 显著大于 0;
    ② 几何相关:ψ 与 w、与 log10(到最近孔距离) 出现非平凡相关。
输出:每 run 每配置的统计行 + 可选的 x 轴 ψ 剖面图(corr_axis_profile.png)。
"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball
import physics
from a2q_model import load_run, predict_a2q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
R_MAX = 30.0


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def probe_run(run_name, labels, axis_lbs=()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(os.path.join(RUNS, run_name), device)
    print(f"\n===== {run_name} (variant={ck['variant']}) =====")
    rng = np.random.default_rng(5)
    x = sample_ball(50000, R_MAX, rng).astype(np.float32)
    axis_out = {}
    for lb in labels:
        m = ck["meta"][lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
        u = predict_a2q(model, x, cinfo, device)
        ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64)
        xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
        Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
        St = torch.zeros((2, 3), dtype=torch.float64)
        ug = physics.guide_u(torch.from_numpy(x).double(), ma, xs, Ps, St).numpy()
        w = (ug - m["wmin"]) / (m["wmax"] - m["wmin"] + 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = (u / (m["kappa"] * ug + 1e-300) - 1.0) / (w + 1e-12)
        r1 = np.linalg.norm(x - xs[0].numpy(), axis=1)
        r2 = np.linalg.norm(x - xs[1].numpy(), axis=1)
        lrmin = np.log10(np.minimum(r1, r2) + 1e-6)
        ok = np.isfinite(psi) & (np.abs(ug) > 1e-6)
        ps = psi[ok]
        qq = np.percentile(ps, [10, 50, 90, 99, 100])
        strong = ok & (w > 0.3)
        iqr = np.subtract(*np.percentile(psi[strong], [75, 25])) if strong.any() else 0.0
        med = np.percentile(psi[strong], 50) if strong.any() else 0.0
        rel = abs(iqr) / (abs(med) + 1e-9)
        sat = float((np.abs(ps) > 0.9 * 3.0).mean())
        print(f"  {lb:<5} |ψ| p10/50/90/99/max = "
              f"{abs(qq[0]):.3f}/{abs(qq[1]):.3f}/{abs(qq[2]):.3f}/{abs(qq[3]):.3f}/"
              f"{qq[4]:.2f}  强场区相对IQR={rel:.2f}  饱和={sat:.2%}  "
              f"ρ(ψ,w)={pearson(psi[ok], w[ok]):+.2f} "
              f"ρ(ψ,log r)={pearson(psi[ok], lrmin[ok]):+.2f}")
        if lb in axis_lbs:
            axis_out[lb] = (m, cinfo)
    if axis_out:
        _axis_profile(run_name, model, device, axis_out)


def _axis_profile(run_name, model, device, axis_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                                  "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    xa = np.linspace(-25, 25, 2001).astype(np.float32)
    ya = np.zeros_like(xa)
    xax = np.stack([xa, ya, ya], axis=1)
    mask = (np.abs(xa - 3.0) > 0.35) & (np.abs(xa + 3.0) > 0.35)
    fig, axes = plt.subplots(1, len(axis_out), figsize=(6 * len(axis_out), 4.2),
                             squeeze=False)
    for ax, (lb, (m, cinfo)) in zip(axes[0], axis_out.items()):
        u = predict_a2q(model, xax, cinfo, device)
        ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64)
        xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
        Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
        St = torch.zeros((2, 3), dtype=torch.float64)
        ug = physics.guide_u(torch.from_numpy(xax).double(), ma, xs, Ps, St).numpy()
        w = (ug - m["wmin"]) / (m["wmax"] - m["wmin"] + 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = (u / (m["kappa"] * ug + 1e-300) - 1.0) / (w + 1e-12)
        pn = np.percentile(np.abs(psi[mask & np.isfinite(psi)]), 90)
        ax.plot(xa[mask], np.clip(psi[mask], -5 * pn, 5 * pn), lw=1.2)
        ax.plot(xa[mask], w[mask], lw=0.9, ls="--", alpha=0.6, label="w(x)")
        ax.axhline(0.0, color="k", lw=0.5)
        for xp in (3.0, -3.0):
            ax.axvline(xp, color="gray", lw=0.6, alpha=0.5)
        ax.set_title(f"{run_name} {lb} (q={m['q']:g}): x 轴 ψ 剖面", fontsize=11)
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
    fd = os.path.join(RUNS, run_name, "figs")
    os.makedirs(fd, exist_ok=True)
    out = os.path.join(fd, "corr_axis_profile.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  剖面图: {out}")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["a2q_opv2"]
    for rn in runs:
        if os.path.isdir(os.path.join(RUNS, rn)):
            probe_run(rn, ["q10", "q15", "q20", "q54"],
                      axis_lbs=("q10", "q20") if rn.endswith("opv2") else ())
        else:
            print(f"跳过(不存在): {rn}")

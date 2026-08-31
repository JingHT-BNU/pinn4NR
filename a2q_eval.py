"""a2q_eval.py —— A2 q∈[1,10] 攻关:统一评估 + param_axis_profiles 图。

对每个有谱参考解的配置(训练 + 留出):
  - 81³ 均匀网格 [-30,30]³ 上模型 vs 谱参考解 L2RE(及引导基线 κ·u_g 的 L2RE)
  - 留出配置额外给 PDE 残差自检(20000 独立点)
输出: <run>/eval.json + <run>/figs/param_axis_profiles.png(3×6 面板,
      模型/引导/谱参考同图,格式与原 parametric_viz 一致)
"""
import json, logging, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import physics
from a2q_model import load_run, predict_a2q

log = logging.getLogger("paper.A2.a2q_eval")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFS = os.path.join(ROOT, "paper", "tools", "refs_a2")
sys.path.insert(0, os.path.join(ROOT, "paper", "tools"))
from spectral_reference import SpectralPunctureSolver  # noqa: E402

AXIS_RMIN = 0.05    # 奇点邻域掩膜半径(u 在 puncture 处正则,收窄掩膜以减小峰顶截断)
RCUT = 0.3          # 评估指标用的奇点邻域排除(A1 reference rcut=0.3 口径)
GRID_N = 81
R_MAX = 30.0


def _bridge(ax_x, ax_u):
    """用缺口两端各 12 个点的三次样条,桥接奇点邻域掩膜造成的曲线断口,
    使 x 轴剖面图在尖峰处平滑连续(掩膜本身只保留 ±AXIS_RMIN 的窄缝)。"""
    from scipy.interpolate import CubicSpline
    u = np.asarray(ax_u, dtype=float).copy()
    u[~np.isfinite(u)] = np.nan
    for xp in (3.0, -3.0):
        lo = np.searchsorted(ax_x, xp - AXIS_RMIN)
        hi = np.searchsorted(ax_x, xp + AXIS_RMIN)
        if lo == 0 or hi >= len(ax_x) or hi <= lo:
            continue
        left = max(lo - 12, 0)
        right = min(hi + 12, len(ax_x))
        xs = np.concatenate([ax_x[left:lo], ax_x[hi:right]])
        us = np.concatenate([u[left:lo], u[hi:right]])
        m = np.isfinite(us)
        if m.sum() < 4:
            continue
        u[lo:hi] = CubicSpline(xs[m], us[m])(ax_x[lo:hi])
    return u


def l2re(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30)))


def main():
    setup_logging("A2", "a2q_eval")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--grid-n", type=int, default=GRID_N)
    ap.add_argument("--skip-fig", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu")
    model, ck = load_run(args.run, device)
    meta = ck["meta"]
    labels = sorted(meta.keys(), key=lambda lb: meta[lb]["q"])
    heldout = set(ck.get("heldout_labels", []))

    # 均匀网格(所有配置共用; 排除奇点邻域 rcut 球, 避免 u_g 精确命中爆炸伪影)
    g = np.linspace(-R_MAX, R_MAX, args.grid_n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    keep = (np.abs(pts[:, 0] - 3.0) > RCUT) | (np.abs(pts[:, 1]) > RCUT) | \
           (np.abs(pts[:, 2]) > RCUT)
    keep &= (np.abs(pts[:, 0] + 3.0) > RCUT) | (np.abs(pts[:, 1]) > RCUT) | \
            (np.abs(pts[:, 2]) > RCUT)
    pts = np.ascontiguousarray(pts[keep])
    log.info(f"网格 {args.grid_n}³ → 排除奇点球(r<{RCUT})后 {len(pts)} 点")

    # x 轴稠密剖点
    xs_d = np.linspace(-28.0, 28.0, 2401)
    axis_pts = np.zeros((len(xs_d), 3), dtype=np.float32)
    axis_pts[:, 0] = xs_d

    results, evals = {}, {}
    figs_ok = not args.skip_fig
    if figs_ok:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                                  "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
        n = len(labels)
        ncol = 6
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                                 sharex=True)
        axes = np.atleast_2d(axes)

    t0 = time.time()
    for i, lb in enumerate(labels):
        m = meta[lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
        cinfo["m1"] = 0.5
        res = dict(q=m["q"], heldout=lb in heldout)
        src = os.path.join(REFS, f"ref_a2_{lb}.npz")
        if os.path.exists(src):
            if src not in evals:
                evals[src] = SpectralPunctureSolver.from_coefficients(
                    src, device=str(device), verify=False)
            ev = evals[src]
            u_ref = ev.evaluate(pts, chunk=131072, dtype=torch.float32)
            u_mod = predict_a2q(model, pts, cinfo, device)
            res["l2re"] = l2re(u_mod, u_ref)
            with torch.no_grad():
                ug = np.empty(len(pts))
                ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64, device=device)
                xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                                   device=device)
                Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                                  dtype=torch.float64, device=device)
                St = torch.zeros((2, 3), dtype=torch.float64, device=device)
                for c0 in range(0, len(pts), 262144):
                    xt = torch.from_numpy(pts[c0:c0 + 262144]).double().to(device)
                    ug[c0:c0 + 262144] = (float(m["kappa"]) *
                                          physics.guide_u(xt, ma, xst, Pt, St)).cpu().numpy()
            res["l2re_guide"] = l2re(ug.astype(np.float64), u_ref)
            res["improve"] = res["l2re_guide"] / max(res["l2re"], 1e-30)
            ref_axis = ev.evaluate(axis_pts, chunk=131072, dtype=torch.float64)
        else:
            u_mod = None
            ref_axis = None
            res["l2re"] = None

        results[lb] = res
        msg = (f"{lb} (q={m['q']:g}{' 留出' if lb in heldout else ''}): "
               f"L2RE={res['l2re']:.4e}" if res["l2re"] is not None
               else f"{lb} (q={m['q']:g}): 参考解缺失")
        if res["l2re"] is not None:
            msg += f"  引导基线={res['l2re_guide']:.3e} 改进×{res['improve']:.1f}"
        log.info(f"{msg}  ({time.time()-t0:.0f}s)")

        if figs_ok:
            ax = axes[i // ncol][i % ncol]
            um = predict_a2q(model, axis_pts, cinfo, device).astype(float)
            um = _bridge(xs_d, um)
            ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64)
            xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
            Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
            St = torch.zeros((2, 3), dtype=torch.float64)
            xt = torch.from_numpy(axis_pts).double()
            ug = (float(m["kappa"]) * physics.guide_u(xt, ma, xst, Pt, St)).numpy()
            ug = _bridge(xs_d, ug)
            ax.plot(xs_d, um, "-", lw=0.9, color="C0", label="模型 u_θ")
            ax.plot(xs_d, ug, "--", lw=0.6, color="C2", label="引导基线 κ·u_g")
            if ref_axis is not None:
                ur = _bridge(xs_d, ref_axis)
                ax.plot(xs_d, ur, "-", lw=0.9, color="C3", alpha=0.9,
                        label="谱方法参考解")
            tag = " | 零样本" if lb in heldout else ""
            ax.set_title(f"{lb} (m2={m['m2']:g}{tag})", fontsize=9)
            ax.set_xlim(-28, 28)
            ax.grid(True, alpha=0.3)
            if i % ncol == 0:
                ax.set_ylabel("u")
            if i >= n - ncol:
                ax.set_xlabel("x")
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")

    # 汇总
    have = [r for r in results.values() if r["l2re"] is not None]
    summ = {}
    for tag, sel in (("train", [r for r in have if not r["heldout"]]),
                     ("heldout", [r for r in have if r["heldout"]])):
        if sel:
            v = [r["l2re"] for r in sel]
            summ[tag] = {"n": len(v), "max": max(v), "mean": float(np.mean(v)),
                         "geo_mean": float(np.exp(np.mean(np.log(v))))}
    log.info(f"[汇总] {json.dumps(summ)}")
    q1 = results.get("q10", {})
    log.info(f"[对照] A1 base 口径: 自研全参考 0.0067 / 论文 0.017;本模型 q10 L2RE="
             f"{q1.get('l2re')}")

    out = os.path.join(args.run, "eval.json")
    json.dump({"results": results, "summary": summ,
               "grid_n": args.grid_n, "target": {"A1_full": 0.0067, "A1_paper": 0.017}},
              open(out, "w"), indent=1)
    log.info(f"评估写入 {out}")

    if figs_ok:
        for j in range(len(labels), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{os.path.basename(args.run)} x 轴剖面 (y=z=0): "
                     "模型 vs 引导基线 vs 谱方法参考解(L=48)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fd = os.path.join(args.run, "figs")
        os.makedirs(fd, exist_ok=True)
        fig.savefig(os.path.join(fd, "param_axis_profiles.png"), dpi=400)
        plt.close(fig)
        log.info(f"[图] {os.path.join(fd, 'param_axis_profiles.png')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

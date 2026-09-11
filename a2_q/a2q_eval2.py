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
"""a2q_eval2.py -- 新评估口径(2026-09-01 无人值守指令 #1/#2)。

旧口径(81³,[-30,30]³,rcut=0.3,格距 0.75)由体积主导,对峰区几乎失明
(r<1.0 邻域仅 ~20 格点),不能代表"可视化贴合度"。新口径:

  - 窗口: [-10,10]³ 均匀 N³ 网格(默认 N=161,格距 0.125)
  - 掩膜: 仅剔除奇点球 r<rcut(默认 0.1)—— 只消除 1/r 爆炸本身,
    峰顶环带完整进入指标
  - 子指标: global / peak(双孔 r<1.5)/ ring(r<0.5)/ valley(原点 r<1.5)/
    far(r0>5) 的 L2RE;轴向稠密线:峰窗 L2RE、峰高比 h/h_ref、
    谷深比 d/d_ref、谷窗 L2RE
  - 参考解网格缓存: 每配置一次性求值缓存到
    data/refs/a2v2/evalcache/w10_n<N>/ref_<lb>.npz
  - 可视化面板: 每配置一行 [全轴剖面, 峰区放大, z=0 符号误差图]
  - 参考解目录优先 data/refs/a2v2(新参考解),回退 data/refs/a2(旧 L48)

用法:
  python a2q_eval2.py --run data/runs/a2/a2q_opv3 [--grid-n 161]
                      [--configs q10,q20,...] [--skip-fig]
  python a2q_eval2.py --a1 data/runs/a1/base_a1     # A1 base 校准阈值
输出: <run>/eval2[_a1].json + <run>/figs/eval2_panels.png
注意: a1 模式与 a2q 模式分别懒加载 physics 模块(paper 与 pinn4NR 的
physics 同名),不要在同进程混用。
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from logutil import setup_logging

log = logging.getLogger("paper.A2.a2q_eval2")

R_MAX2 = 10.0


def l2re(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30)))


def build_grid(n, rmax, rcut):
    g = np.linspace(-rmax, rmax, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
    r0 = np.linalg.norm(pts, axis=1)
    keep = (r1 > rcut) & (r2 > rcut)
    return (pts[keep].astype(np.float32), r1[keep], r2[keep], r0[keep],
            g, np.where(keep)[0])


def _find_ref(lb):
    for d in ("a2v2", "a2"):
        for pat in (f"ref_{lb}.npz", f"ref_a2_{lb}.npz"):
            p = os.path.join(_ROOT, "data", "refs", d, pat)
            if os.path.exists(p):
                return p
    return None


def _load_solver(src, device):
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    import spectral_reference as sr
    return sr.SpectralPunctureSolver.from_coefficients(src, device=str(device),
                                                       verify=False)


def ref_grid_u(lb, n, rmax, rcut, device):
    """每配置参考解在评估网格上的求值(带缓存)。返回 (u_ref_keep 或 None, src)。"""
    src = _find_ref(lb)
    if src is None:
        return None, None
    cache_dir = os.path.join(_ROOT, "data", "refs", "a2v2", "evalcache",
                             f"w{rmax:g}_n{n}")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, f"ref_{lb}.npz")
    if os.path.exists(dst):
        u_full = np.load(dst)["u_ref"]
        g = np.linspace(-rmax, rmax, n)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        ptsf = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        rr1 = np.linalg.norm(ptsf - np.array([3.0, 0, 0]), axis=1)
        rr2 = np.linalg.norm(ptsf - np.array([-3.0, 0, 0]), axis=1)
        return u_full[(rr1 > rcut) & (rr2 > rcut)], src
    solver = _load_solver(src, device)
    full_n = n ** 3
    g = np.linspace(-rmax, rmax, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    u_full = np.empty(full_n, dtype=np.float32)
    B = 32768   # 谱求值显存 ~chunk×K×16B(2401 模态 complex128);262144 会 OOM
    t0 = time.time()
    for i in range(0, full_n, B):
        u_full[i:i + B] = solver.evaluate(pts[i:i + B], chunk=B,
                                          dtype=torch.float64).astype(
                                              np.float32)
        if (i // B) % 4 == 0:
            log.info("  ref %s: %d/%d (%.0fs)", lb, min(i + B, full_n),
                     full_n, time.time() - t0)
    np.savez(dst, u_ref=u_full, grid_n=n, rmax=rmax, rcut=rcut,
             src=os.path.relpath(src, HERE))
    log.info("  ref cache %s -> %s (%.0fs)", lb, dst, time.time() - t0)
    r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
    keep = (r1 > rcut) & (r2 > rcut)
    return u_full[keep], src


def axial_ref_u(lb, device, n_ax=2001, rmax=10.0):
    src = _find_ref(lb)
    if src is None:
        return None, None
    solver = _load_solver(src, device)
    xs = np.linspace(-rmax, rmax, n_ax)
    ax = np.zeros((len(xs), 3))
    ax[:, 0] = xs
    return xs, solver.evaluate(ax, chunk=32768,
                               dtype=torch.float64).astype(np.float64)


def axial_metrics(xs, u_mod, u_ref):
    dx = np.minimum(np.abs(xs - 3.0), np.abs(xs + 3.0))
    fin = np.isfinite(u_mod) & np.isfinite(u_ref)
    pk = fin & (dx >= 0.1) & (dx <= 1.5)
    vy = fin & (np.abs(xs) <= 1.2)
    return {
        "peak_win_l2re": l2re(u_mod[pk], u_ref[pk]),
        "peak_maxabs": float(np.max(np.abs(u_mod[pk] - u_ref[pk]))),
        "peak_height_ratio": float(u_mod[pk].max() /
                                   max(u_ref[pk].max(), 1e-30)),
        "valley_l2re": l2re(u_mod[vy], u_ref[vy]),
        # 两峰间为正的局部极小(非负值),比较极小值之比
        "valley_depth_ratio": float(u_mod[vy].min() /
                                    max(u_ref[vy].min(), 1e-30)),
        "axis_l2re": l2re(u_mod[fin], u_ref[fin]),
    }


def _bridge(xs, u, rcut=0.1):
    """奇点邻域断口样条桥接(绘图用;与 a2q_eval._bridge 同法,rcut 参数化)。"""
    from scipy.interpolate import CubicSpline
    u = np.asarray(u, dtype=float).copy()
    u[~np.isfinite(u)] = np.nan
    for xp in (3.0, -3.0):
        lo = np.searchsorted(xs, xp - rcut)
        hi = np.searchsorted(xs, xp + rcut)
        if lo == 0 or hi >= len(xs) or hi <= lo:
            continue
        left, right = max(lo - 12, 0), min(hi + 12, len(xs))
        xseg = np.concatenate([xs[left:lo], xs[hi:right]])
        useg = np.concatenate([u[left:lo], u[hi:right]])
        m = np.isfinite(useg)
        if m.sum() < 4:
            continue
        u[lo:hi] = CubicSpline(xseg[m], useg[m])(xs[lo:hi])
    return u


def main():
    setup_logging("A2", "a2q_eval2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--a1", default=None, help="A1 base run dir (single cfg)")
    ap.add_argument("--grid-n", type=int, default=161)
    ap.add_argument("--rcut", type=float, default=0.1)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--fig-configs", default="q10,q20,q50,q100")
    ap.add_argument("--skip-fig", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "auto" else "cpu")

    a1_mode = args.a1 is not None
    if a1_mode:
        run_dir = args.a1
        labels = ["q10"]
        ckpt = torch.load(os.path.join(run_dir, "model.pt"),
                          map_location=device, weights_only=False)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "a1_single"))
        from model import GuidedPINN
        from config import PINNConfig
        pcfg = PINNConfig(hidden_layers=ckpt["hidden_layers"],
                          hidden_neurons=ckpt["hidden_neurons"])
        sd = ckpt["model_state"]
        a1_model = GuidedPINN(pcfg, kappa=ckpt["kappa"], c=ckpt["c"],
                              masses=sd["masses"].cpu().numpy(),
                              xs=sd["xs"].cpu().numpy(),
                              Ps=sd["Ps"].cpu().numpy(),
                              Ss=sd["Ss"].cpu().numpy(),
                              u_min=ckpt["u_min"], u_max=ckpt["u_max"]
                              ).to(device)
        a1_model.load_state_dict(sd)
        a1_model.double()
        a1_model.eval()
        meta = {"q10": {"q": 1.0}}
        train_labels, heldout = set(labels), set()

        def predict(pp, cinfo):
            out = []
            with torch.no_grad():
                for i in range(0, len(pp), 65536):
                    xb = torch.from_numpy(pp[i:i + 65536]).double().to(device)
                    out.append(a1_model(xb).cpu().numpy())
            return np.concatenate(out)
    else:
        run_dir = args.run
        sys.path.insert(0, os.path.dirname(HERE))
        import a2q_model as A2
        model, ck = A2.load_run(run_dir, device)
        meta = ck["meta"]
        train_labels = set(ck.get("train_labels", meta.keys()))
        heldout = set(ck.get("heldout_labels", []))
        labels = sorted(meta.keys(), key=lambda lb: meta[lb]["q"])
        if args.configs:
            want = set(args.configs.split(","))
            labels = [lb for lb in labels if lb in want]

        def predict(pp, cinfo):
            return A2.predict_a2q(model, pp, cinfo, device, chunk=65536)

    pts, r1, r2, r0, grid_g, _ = build_grid(args.grid_n, R_MAX2, args.rcut)
    n = args.grid_n
    log.info("eval2 网格 %d³ (±%g, rcut=%g) -> %d 点",
             n, R_MAX2, args.rcut, len(pts))

    results = {}
    t0 = time.time()
    for lb in labels:
        m = meta[lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")
                 if k in m}
        cinfo.setdefault("m1", 0.5)
        u_ref_keep, src = ref_grid_u(lb, n, R_MAX2, args.rcut, device)
        if u_ref_keep is None:
            log.warning("%s: 参考解缺失,跳过", lb)
            continue
        u_ref = u_ref_keep.astype(np.float64)
        u_mod = predict(pts, cinfo).astype(np.float64)
        res = {"q": float(m["q"]),
               "group": ("heldout" if lb in heldout else
                         "train" if lb in train_labels else "zero_shot"),
               "ref": os.path.relpath(src, HERE)}
        res["global"] = l2re(u_mod, u_ref)
        pk = np.minimum(r1, r2) < 1.5
        rg = np.minimum(r1, r2) < 0.5
        vy = r0 < 1.5
        fr = r0 > 5.0
        res["peak"] = l2re(u_mod[pk], u_ref[pk])
        res["ring"] = l2re(u_mod[rg], u_ref[rg])
        res["valley"] = l2re(u_mod[vy], u_ref[vy])
        res["far"] = l2re(u_mod[fr], u_ref[fr])
        xs, u_ref_ax = axial_ref_u(lb, device, rmax=R_MAX2)
        if u_ref_ax is not None:
            ax_pts = np.zeros((len(xs), 3))
            ax_pts[:, 0] = xs
            u_mod_ax = predict(ax_pts.astype(np.float32), cinfo).astype(
                np.float64)
            res.update(axial_metrics(xs, u_mod_ax, u_ref_ax))
        results[lb] = res
        log.info("%-6s q=%-5g g=%.3e pk=%.3e rg=%.3e vy=%.3e far=%.3e "
                 "axw=%.3e h/href=%.3f d/dref=%.3f (%.0fs)",
                 lb, m["q"], res["global"], res["peak"], res["ring"],
                 res["valley"], res["far"], res.get("peak_win_l2re", -1),
                 res.get("peak_height_ratio", -1),
                 res.get("valley_depth_ratio", -1), time.time() - t0)

    summ = {}
    for grp in ("train", "heldout", "zero_shot"):
        sel = [r for r in results.values() if r["group"] == grp]
        if sel:
            summ[grp] = {k: {"mean": float(np.mean([r[k] for r in sel])),
                             "max": float(np.max([r[k] for r in sel]))}
                         for k in ("global", "peak", "ring", "valley",
                                   "peak_win_l2re") if k in sel[0]}
            hr = [r["peak_height_ratio"] for r in sel
                  if "peak_height_ratio" in r]
            if hr:
                summ[grp]["peak_height_ratio"] = {
                    "min": float(min(hr)), "max": float(max(hr))}
    log.info("[汇总] %s", json.dumps(summ))

    suffix = "_a1" if a1_mode else ""
    out_p = os.path.join(run_dir, f"eval2{suffix}.json")
    json.dump({"protocol": {"window": [-R_MAX2, R_MAX2], "grid_n": n,
                            "rcut": args.rcut,
                            "note": "new metric 2026-09-01: peak-inclusive"},
               "results": results, "summary": summ},
              open(out_p, "w"), indent=1)
    log.info("写入 %s", out_p)

    if args.skip_fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig_lbs = [lb for lb in args.fig_configs.split(",") if lb in results]
    if not fig_lbs:
        return
    xs, _ = axial_ref_u(fig_lbs[0], device, rmax=R_MAX2)
    ax_pts = np.zeros((len(xs), 3))
    ax_pts[:, 0] = xs
    xx, yy = np.meshgrid(grid_g, grid_g, indexing="ij")
    px = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)
    pr1 = np.sum((px - np.array([3.0, 0, 0])) ** 2, axis=1)
    pr2 = np.sum((px - np.array([-3.0, 0, 0])) ** 2, axis=1)
    pkeep = (pr1 > args.rcut ** 2) & (pr2 > args.rcut ** 2)
    z0 = (n // 2) * n * n
    cache_dir = os.path.join(_ROOT, "data", "refs", "a2v2", "evalcache",
                             f"w{R_MAX2:g}_n{n}")

    nrow = len(fig_lbs)
    fig, axes = plt.subplots(nrow, 3, figsize=(15.5, 3.1 * nrow),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    axes = np.atleast_2d(axes)
    for i, lb in enumerate(fig_lbs):
        m = meta[lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")
                 if k in m}
        cinfo.setdefault("m1", 0.5)
        _, u_ref_ax = axial_ref_u(lb, device, rmax=R_MAX2)
        u_mod_ax = predict(ax_pts.astype(np.float32), cinfo).astype(np.float64)
        u_full = np.load(os.path.join(cache_dir, f"ref_{lb}.npz"))["u_ref"]
        u_ref_plane = u_full[z0:z0 + n * n].reshape(n, n)
        um = predict(px[pkeep].astype(np.float32), cinfo).astype(np.float64)
        err = np.full(n * n, np.nan)
        err[np.where(pkeep)[0]] = um - u_ref_plane.ravel()[pkeep]
        err = err.reshape(n, n)

        axf, axz, axe = axes[i]
        axf.plot(xs, _bridge(xs, u_ref_ax, args.rcut), "-", lw=1.5,
                 color="C3", label="spectral reference")
        axf.plot(xs, _bridge(xs, u_mod_ax, args.rcut), "-", lw=0.9,
                 color="C0", label="model")
        axf.set_xlim(-10, 10)
        axf.set_title(f"{lb} (q={m['q']:g}) full axis", fontsize=9)
        axf.legend(fontsize=6)
        axf.grid(alpha=0.3)
        mz = (xs >= 1.3) & (xs <= 4.7)
        axz.plot(xs[mz], _bridge(xs, u_ref_ax, args.rcut)[mz], "-", lw=1.5,
                 color="C3")
        axz.plot(xs[mz], _bridge(xs, u_mod_ax, args.rcut)[mz], "-", lw=0.9,
                 color="C0")
        axz.set_xlim(1.3, 4.7)
        axz.set_title("zoom @ x=+3", fontsize=9)
        axz.grid(alpha=0.3)
        fin = np.abs(err[np.isfinite(err)])
        vn = float(np.median(fin)) * 10 if fin.size else 1e-4
        pcm = axe.pcolormesh(xx, yy, err, shading="auto", cmap="RdBu_r",
                             norm=SymLogNorm(linthresh=max(vn, 1e-7),
                                             vmin=-vn * 100, vmax=vn * 100))
        axe.set_aspect("equal")
        axe.set_title("z=0 signed error", fontsize=9)
        if i == nrow - 1:
            fig.colorbar(pcm, ax=axe, fraction=0.046)
    fig.suptitle(f"{os.path.basename(run_dir)} — eval2 "
                 f"(window ±10, grid {n}³, rcut {args.rcut})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fd = os.path.join(run_dir, "figs")
    os.makedirs(fd, exist_ok=True)
    fp = os.path.join(fd, "eval2_panels.png")
    fig.savefig(fp, dpi=250)
    plt.close(fig)
    log.info("[图] %s", fp)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("eval2 failed")
        raise

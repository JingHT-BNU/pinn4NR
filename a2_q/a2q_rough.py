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
"""a2q_rough.py -- u 剖面粗糙度诊断(量化"凸起/凹陷"不稳定行为)。

对给定 run,沿若干射线采样 u 剖面,计算:
  - far_wiggle: 远场(沿射线 s>2)二阶导 RMS / 剖面 max|u|(无量纲粗糙度)
  - far_signflip: 远场二阶导符号翻转次数(真实光滑解应 ≤1)
  - near_wiggle: 近场(s<1.5)二阶导 RMS / max|u|
射线:每孔 12 条斐波那契方向。输出 <run>/rough.json 与 figs/rough.png。

用法: python a2q_rough.py --run data/runs/a2/a2q_v4b [--configs q10,q100]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools"))
from logutil import setup_logging
import a2q_model as A2

log = logging.getLogger("paper.A2.a2q_rough")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_ROOT, "data", "runs", "a2")


def fib_dirs(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi),
                     np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)


def ray_metrics(u, s):
    u2 = np.gradient(np.gradient(u, s), s)
    far = s > 2.0
    scale = max(np.abs(u).max(), 1e-30)
    u2f = u2[far]
    sign = np.sign(u2f)
    flips = int(np.sum(sign[1:] * sign[:-1] < 0))
    return dict(far_wiggle=float(np.sqrt(np.mean(u2f ** 2)) / scale),
                far_signflip=flips,
                near_wiggle=float(np.sqrt(np.mean(u2[s < 1.5] ** 2)) / scale))


def main():
    setup_logging("A2", "a2q_rough")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--configs", default="q10,q20,q50,q100")
    ap.add_argument("--n-ray", type=int, default=12)
    ap.add_argument("--n-pts", type=int, default=400)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.run if os.path.isabs(args.run) else os.path.join(RUNS, args.run)
    model, ck = A2.load_run(run_dir, device)
    meta = ck["meta"]

    dirs = fib_dirs(args.n_ray)
    out = {}
    prof = {}
    for lb in args.configs.split(","):
        m = meta[lb]
        cinfo = {k: float(m[k]) for k in
                 ("q", "m2", "kappa", "sq", "wmin", "wmax")}
        s = np.linspace(0.1, 10.0, args.n_pts)
        v_far, v_near, flips = [], [], []
        for c in ((3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)):
            for i, e in enumerate(dirs):
                pts = np.asarray(c)[None, :] + s[:, None] * e[None, :]
                u = A2.predict_a2q(model, pts.astype(np.float64), cinfo,
                                   device, chunk=65536)
                met = ray_metrics(u, s)
                v_far.append(met["far_wiggle"])
                v_near.append(met["near_wiggle"])
                flips.append(met["far_signflip"])
                if c[0] > 0 and i < 4 and lb in ("q10", "q100"):
                    prof.setdefault(lb, []).append(u.copy())
        agg = dict(far_wiggle=float(np.mean(v_far)),
                   near_wiggle=float(np.mean(v_near)),
                   far_signflip_mean=float(np.mean(flips)),
                   far_signflip_max=int(np.max(flips)))
        out[lb + ":SUMMARY"] = agg
        log.info("[%s] far_wiggle=%.3e near_wiggle=%.3e signflip mean/max=%d/%d",
                 lb, agg["far_wiggle"], agg["near_wiggle"],
                 agg["far_signflip_mean"], agg["far_signflip_max"])

    dst = os.path.join(run_dir, "rough.json")
    json.dump(out, open(dst, "w"), indent=1)
    log.info("写入 %s", dst)

    if prof:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, axes = plt.subplots(1, len(prof),
                                     figsize=(6 * len(prof), 4),
                                     squeeze=False)
            for j, (lb, rays_u) in enumerate(sorted(prof.items())):
                ax = axes[0][j]
                for i, u in enumerate(rays_u):
                    ax.plot(s, u, lw=1, alpha=0.8, label=f"dir{i}")
                ax.set_yscale("log")
                ax.set_xlabel("s (距 +3 孔)")
                ax.set_title(f"{lb} 射线剖面(前4方向)")
                ax.legend(fontsize=7)
            fig.tight_layout()
            os.makedirs(os.path.join(run_dir, "figs"), exist_ok=True)
            fig.savefig(os.path.join(run_dir, "figs", "rough.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("rough.png 生成失败: %s", e)


if __name__ == "__main__":
    main()

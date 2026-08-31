"""
tools/make_refs_batch.py —— 批量生成 A3 代表配置的谱方法参考解
================================================================

选取规则与 multi_param_viz.plot_axis_profiles 一致: base 配置 + 按 κ 分位数
取 5 个代表配置, 共 6 个。对每个配置调用 SpectralPunctureSolver 求解并保存
paper/tools/refs/ref_<label>.npz (与 reference_u.npz 同格式 + 谱系数)。

分辨率默认 N_r=512, L=48 (2026-08-29 起; L=32 在轻质量+自旋奇点针尖欠收敛
~25-35%, 见 verify_ref_tip.py; base L=48 vs TwoPunctures L2RE=6.35e-5)。
npz 含 radial_coeffs 谱系数, 可用 SpectralPunctureSolver.from_coefficients
免重解任意分辨率求值。

用法:
    python paper/tools/make_refs_batch.py                 # 6 个代表配置
    python paper/tools/make_refs_batch.py --force         # 覆盖已有文件
    python paper/tools/make_refs_batch.py --n-r 384 --lmax 32   # 低分辨率(快)
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PAPER, "tools"))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "A3_multi_param"))
from logutil import setup_logging  # noqa: E402

log = logging.getLogger("paper.tools.make_refs_batch")
from spectral_reference import SpectralPunctureSolver  # noqa: E402


def main():
    setup_logging("tools", "make_refs_batch")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(
        PAPER, "runs", "multi_param_a2", "configs.json"))
    p.add_argument("--out-dir", default=os.path.join(PAPER, "tools", "refs"))
    p.add_argument("--n-r", type=int, default=512)
    p.add_argument("--lmax", type=int, default=48)
    p.add_argument("--n-theta", type=int, default=72)
    p.add_argument("--n-phi", type=int, default=144)
    p.add_argument("--grid-n", type=int, default=197)
    p.add_argument("--R", type=float, default=30.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="覆盖已存在的参考解")
    args = p.parse_args()

    info = json.load(open(args.config))
    cfgs = info["train"]
    BASE_RAW = np.array([0.5, 0.5, 3.0, -3.0, 0.2, -0.2, 0.0, 0.0])
    base_cfg = min(cfgs, key=lambda c: np.linalg.norm(np.array(c["raw"]) - BASE_RAW))
    order = np.argsort([c["kappa"] for c in cfgs])
    picks = [cfgs[order[q]] for q in np.linspace(0, len(cfgs) - 1, 5).astype(int)]
    selected = []
    for c in [base_cfg] + picks:
        if all(c["label"] != s["label"] for s in selected):
            selected.append(c)
    selected = selected[:6]

    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"待生成 {len(selected)} 个配置: {[c['label'] for c in selected]}")
    n = args.grid_n
    axis = np.linspace(-args.R, args.R, n)
    X, Yc, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack([X, Yc, Z], axis=-1).reshape(-1, 3)

    for ci, cfg in enumerate(selected):
        out_path = os.path.join(args.out_dir, f"ref_{cfg['label']}.npz")
        if os.path.exists(out_path) and not args.force:
            log.info(f"[{ci + 1}/{len(selected)}] {cfg['label']} 已存在, 跳过 (--force 覆盖)")
            continue
        raw = np.array(cfg["raw"], dtype=np.float64)
        log.info(f"[{ci + 1}/{len(selected)}] {cfg['label']} (κ={cfg['kappa']:.4f}): "
                 f"求解 N_r={args.n_r}, L={args.lmax}...")
        t0 = time.time()
        solver = SpectralPunctureSolver(raw, N_r=args.n_r, L=args.lmax,
                                        N_th=args.n_theta, N_ph=args.n_phi,
                                        device=args.device)
        it, stats = solver.solve()
        log.info(f"  迭代 {it}, 谱残差 max={stats['res_max']:.2e} "
                 f"rms={stats['res_rms']:.2e}, modal_max={stats['res_modal_max']:.2e}")
        u = solver.evaluate(pts, dtype=torch.float32)
        log.info(f"  网格求值完成 ({time.time() - t0:.0f}s), u ∈ [{u.min():.4e}, {u.max():.4e}]")
        meta = {"params": raw.tolist(), "label": cfg["label"], "kappa": cfg["kappa"],
                "N_r": args.n_r, "L": args.lmax, "N_theta": args.n_theta,
                "N_phi": args.n_phi, "R0": 15.0, "grid_n": n, "R": args.R,
                "iterations": it, "res_max": stats["res_max"],
                "res_rms": stats["res_rms"],
                "method": "spherical-harmonic spectral, continuation+Anderson"}
        np.savez(out_path,
                 x_ref=pts.astype(np.float32),
                 u_ref=u.astype(np.float32),
                 radial_coeffs=solver.export_coefficients(),
                 raw=raw, xc=solver.xc.astype(np.float64),
                 meta=json.dumps(meta))
        log.info(f"  已保存: {out_path}")
    log.info("全部完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

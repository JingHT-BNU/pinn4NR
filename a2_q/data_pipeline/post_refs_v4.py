#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成含**远场**的参考解子集(refsub),供 A2 训练直接监督 r>10 区域。

为什么需要
----------
a2q_data_v3 的 refsub 只覆盖 r<=10 的球(x_int 的 r0 上界恰为 10.000,
x_bnd 落在 r0=10.000 的球面上)。也就是说 `L_ref` 对 r>10 **没有任何监督**,
而物理型指标(以及 eval2)的 far 区是 r0>5 / r0>=10 —— 模型在那里是纯外推,
只被 PDE 残差项这一项弱约束。这正是远场 Hamilton 残差比参考解差 4~5 个
数量级的根源。

谱参考解本身**没有这个限制**:spectral_reference 用径向紧化
r = R0(1+s)/(1-s)(s∈[-1,1] -> r∈[0,inf)) + 球谐展开 l<=48 表示,
`SpectralPunctureSolver.from_coefficients(...).evaluate(pts)` 可在任意半径
求值。r=28 对应 s=0.302(R0=15),落在 s 区间中部,是分辨率最好的区域。

做法
----
保留 v3 的全部内部点(不改动近场行为),追加一层远场壳 r∈(10, R_far],
并按**参照归一化质量**自动定权:使远场对 Σ w·u_ref² 的贡献占比 ≈ --far-share。
因为训练损失是

    L_ref = Σ w (u-u_ref)² / Σ w u_ref²

分母若全被内区占据,远场的相对误差就几乎不影响损失;定权后远场获得
与之相称的权重。

用法::

    # 先单配置验证
    python a2_q/data_pipeline/post_refs_v4.py --config q10 --dry-run
    # 全量生成到新数据集(不触碰 v3)
    python a2_q/data_pipeline/post_refs_v4.py --out-dir data/datasets/a2q_data_v4
"""
# ---- path bootstrap ----------------------------------------------------
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE,):
    while True:
        if _os.path.isdir(_os.path.join(_p, "core")) and \
           _os.path.isdir(_os.path.join(_p, "a2_q")):
            break
        _np = _os.path.dirname(_p)
        if _np == _p:
            break
        _p = _np
    if _os.path.isdir(_os.path.join(_p, "core")):
        import sys as _sys
        for _sub in ("core", "tools", "a2_q"):
            _q = _os.path.join(_p, _sub)
            if _os.path.isdir(_q) and _q not in _sys.path:
                _sys.path.insert(0, _q)
        _ROOT = _p
        break
# ------------------------------------------------------------------------

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

SRC_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v3")
REF_DIRS = ("a2v2", "a2")


def find_ref(lb):
    for d in REF_DIRS:
        for pat in (f"ref_{lb}.npz", f"ref_a2_{lb}.npz"):
            p = os.path.join(_ROOT, "data", "refs", d, pat)
            if os.path.exists(p):
                return p
    return None


def sample_shell(n, r_in, r_out, rng):
    """在球壳 r∈(r_in, r_out] 内均匀采样(体均匀:r³ 均匀)。"""
    u = rng.random(n)
    r = (r_in ** 3 + u * (r_out ** 3 - r_in ** 3)) ** (1.0 / 3.0)
    ct = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    return np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct],
                    axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        _ROOT, "data", "datasets", "a2q_data_v4"))
    ap.add_argument("--n-far", type=int, default=16384,
                    help="每个配置追加的远场点数")
    ap.add_argument("--r-in", type=float, default=10.0)
    ap.add_argument("--r-far", type=float, default=28.0)
    ap.add_argument("--far-share", type=float, default=0.25,
                    help="远场在 Σ w·u_ref² (损失分母)中占的目标份额")
    ap.add_argument("--config", default=None, help="只处理单个配置(验证用)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告统计,不写文件")
    ap.add_argument("--dtype", choices=("float32", "float64"),
                    default="float32",
                    help="谱求值精度。float32 相对误差 ~1e-7,提速约 50x")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import spectral_reference as sr

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = torch.float32 if args.dtype == "float32" else torch.float64

    labels = sorted(f[4:-4] for f in os.listdir(SRC_DIR)
                    if f.startswith("cfg_") and f.endswith(".npz"))
    if args.config:
        labels = [lb for lb in labels if lb == args.config]
        if not labels:
            sys.exit("无此配置: %s" % args.config)

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
    log = []

    for i_lb, lb in enumerate(labels):
        rp = os.path.join(SRC_DIR, f"refsub_{lb}.npz")
        if not os.path.exists(rp):
            print("!! 缺 refsub_%s.npz,跳过" % lb)
            continue
        src = find_ref(lb)
        if src is None:
            print("!! %s 无谱参考解,跳过" % lb)
            continue
        z = np.load(rp)
        x0, u0 = z["x"].astype(np.float64), z["u"].astype(np.float64)
        w0 = z["w"].astype(np.float64) if "w" in z.files \
            else np.ones(len(x0))
        r0 = np.linalg.norm(x0, axis=1)

        rng = np.random.default_rng(args.seed + i_lb)
        xf = sample_shell(args.n_far, args.r_in, args.r_far, rng)

        ev = sr.SpectralPunctureSolver.from_coefficients(
            src, device=str(device), verify=False)
        t0 = time.time()
        uf = np.asarray(ev.evaluate(xf, chunk=32768, dtype=dt),
                        dtype=np.float64)
        el = time.time() - t0

        # 按"参照归一化质量"定权:使远场占 Σ w u_ref² 的份额 ≈ far_share
        #   Σ_in w0 u0²  vs  Σ_far wf uf²
        mass_in = float(np.sum(w0 * u0 ** 2))
        mass_far_raw = float(np.sum(uf ** 2))
        if mass_far_raw > 0 and args.far_share > 0:
            wf = args.far_share / (1.0 - args.far_share) * mass_in / mass_far_raw
        else:
            wf = 0.0
        w_far = np.full(len(xf), wf)

        x = np.concatenate([x0, xf])
        u = np.concatenate([u0, uf])
        w = np.concatenate([w0, w_far])

        m_frac = float(np.sum(w * u ** 2))
        share = float(np.sum(w_far * uf ** 2)) / max(m_frac, 1e-300)
        rec = dict(lb=lb, n_in=len(x0), n_far=len(xf), sec=round(el, 1),
                   r0_in_max=float(r0.max()), far_rms=float(
                       np.sqrt(np.mean(uf ** 2))),
                   w_far=wf, share=share,
                   u_in_rms=float(np.sqrt(np.mean(u0 ** 2))))
        log.append(rec)
        print("[%2d/%2d] %-6s in=%-6d(r0max %5.2f) far=%-6d w_far=%.3e "
              "far 质量份额 %.3f  u_rms in=%.4f far=%.5f  (%.1fs)"
              % (i_lb + 1, len(labels), lb, len(x0), r0.max(), len(xf),
                 wf, share, rec["u_in_rms"], rec["far_rms"], el))

        if args.dry_run:
            continue
        np.savez(os.path.join(args.out_dir, f"refsub_{lb}.npz"),
                 x=x, u=u, w=w)
        shutil.copyfile(os.path.join(SRC_DIR, f"cfg_{lb}.npz"),
                        os.path.join(args.out_dir, f"cfg_{lb}.npz"))

    if log and not args.dry_run:
        with open(os.path.join(args.out_dir, "_summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(dict(n_far=args.n_far, r_in=args.r_in,
                           r_far=args.r_far, far_share=args.far_share,
                           dtype=args.dtype, per_config=log), fh,
                      ensure_ascii=False, indent=1)
        print("\n写出 %d 个配置 -> %s" % (len(log), args.out_dir))


if __name__ == "__main__":
    main()

"""tp_validate_ref.py —— 验证 TP 子午面参考解与自研 L48 谱参考解的一致性。

对同一配置(如 q=1.5)取两条参考路径在共同点集上对比:
  - TP:data/tp_opv3/ref_tq*.npz(x_ref, u_ref,子午面 y≥0 薄层)
  - 谱:paper/tools/refs_a2/ref_a2_qXX.npz(SpectralPunctureSolver 重建)
在 TP 子午面点上重建谱解 u,报 L2RE 与分带。要求子午面点 z=0 平面(y≥0)
与谱解 3D 场比较 —— 轴对称下 u(x,y,0) 即子午面 u(x,ρ)。

用法: python tp_validate_ref.py --tp data/tp_opv3/ref_tq1p5.npz --spec q15
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
SPECS = os.path.join(ROOT, "data", "refs", "a2")
_TOOLS = os.path.join(os.path.dirname(ROOT), "paper", "tools")
if not os.path.exists(os.path.join(_TOOLS, "spectral_reference.py")):
    _TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, _TOOLS)
from spectral_reference import SpectralPunctureSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", required=True)
    ap.add_argument("--spec", required=True, help="谱参考 label,如 q15")
    ap.add_argument("--q", type=float, default=1.5, help="psi 文本模式的 q")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.tp.endswith(".npz"):
        tpz = np.load(args.tp)
        xt, ut = tpz["x_ref"], tpz["u_ref"].astype(np.float64)
        q = float(tpz["q"])
        m1 = float(tpz["m1"]) if "m1" in tpz.files else 0.5
        m2 = float(tpz["m2"]) if "m2" in tpz.files else 0.5 * q
    else:  # 原始 psi 文本(x y z psi)
        arr = np.loadtxt(args.tp)
        xt = arr[:, :3]
        psi = arr[:, 3]
        q = float(args.q)
        m1, m2 = 0.5, 0.5 * q
        r1_ = np.linalg.norm(xt - np.array([3.0, 0, 0]), axis=1)
        r2_ = np.linalg.norm(xt - np.array([-3.0, 0, 0]), axis=1)
        ut = psi - 1.0 - m1 / (2 * r1_) - m2 / (2 * r2_)
    # 只保留 z=0 平面附近且 r>0.3 的点(与评估口径一致)
    r1 = np.linalg.norm(xt - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(xt - np.array([-3.0, 0, 0]), axis=1)
    keep = (np.abs(xt[:, 2]) < 1e-6) & (np.minimum(r1, r2) > 0.3)
    xt, ut, rmin = xt[keep], ut[keep], np.minimum(r1, r2)[keep]
    print(f"TP 点 {len(xt)}(z=0, r>0.3)")

    src = os.path.join(SPECS, f"ref_a2_{args.spec}.npz")
    ev = SpectralPunctureSolver.from_coefficients(src, device=dev, verify=False)
    us = ev.evaluate(xt, chunk=131072, dtype=torch.float64).astype(np.float64)
    d = us - ut
    l2re = float(np.sqrt((d ** 2).sum() / (ut ** 2).sum()))
    print(f"q={q:g} 谱 vs TP 子午面: L2RE={l2re:.3e}")
    for lo, hi in ((0.3, 2), (2, 6), (6, 1e9)):
        m = (rmin >= lo) & (rmin < hi)
        if m.any():
            lr = float(np.sqrt((d[m] ** 2).sum() / (ut[m] ** 2).sum()))
            print(f"  r∈[{lo:g},{hi:g}): {lr:.3e} ({m.sum()} pts)")


if __name__ == "__main__":
    main()

"""a2q_kappa_probe.py —— 诊断 κ 非单调跳变:solve_kappa 是否选错根。

对给定配置,在 κ∈[0.05,1.2] 网格上计算:
  f(κ) = −κ·S_b − (1/8)·V(κ),  V(κ)=∫ KK (ψ_sing+κ·u_g)^{-7} dV(QMC)
以及每个 κ 处 min(ψ_sing + κ·u_g)(物理解要求 ψ>0)。
若 f 有多个根且 solve_kappa 选中的根使 ψ 触 0/变号 → κ 错根。
"""
import logging, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import physics
from data import sobol_volume
from a2q_prep import DATA_DIR, R_MAX
from config import TrainConfig

log = logging.getLogger("paper.A2.a2q_kappa_probe")


def probe(lb):
    z = np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz"))
    ma = torch.tensor(z["masses"]).double()
    xs = torch.tensor(z["xs"]).double()
    Ps = torch.tensor(z["Ps"]).double()
    Ss = torch.tensor(z["Ss"]).double()
    cfg = TrainConfig(n_qmc_vol=200000, n_qmc_surf=20000, R_max=R_MAX)
    x_vol = sobol_volume(200000, R_MAX, seed=12345)  # 固定种子,便于横向比较
    xt = torch.from_numpy(x_vol).double()
    with torch.no_grad():
        ug = physics.guide_u(xt, ma, xs, Ps, Ss)
        kk = physics.bowen_york_KK(xt, ma, xs, Ps, Ss)
        ps = physics.psi_sing(xt, ma, xs)
    # 边界通量 S_b
    n_surf = 20000
    rng = np.random.default_rng(1)
    # 均匀球面采样
    v = rng.normal(size=(n_surf, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    xs_s = (v * R_MAX).astype(np.float64)
    xst = torch.from_numpy(xs_s).double()
    with torch.no_grad():
        ug_b = physics.guide_u(xst, ma, xs, Ps, Ss)
        ug_b2 = physics.guide_u(xst * 1.001, ma, xs, Ps, Ss)
        dudr = (ug_b2 - ug_b) / (0.001 * R_MAX)
    S_b = 4 * np.pi * R_MAX ** 2 * dudr.mean().item()

    print(f"\n===== {lb} (q={float(z['q']):g}) cache κ={float(z['kappa']):.6f} "
          f"S_b={S_b:.4e} =====")
    print(f"{'κ':>8} {'f(κ)':>12} {'min(ψ)':>10} {'V(κ)/8':>12}")
    for k in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
              0.65, 0.7, 0.8, 0.9, 1.0, 1.1]:
        with torch.no_grad():
            psi = ps + k * ug
            vmin = psi.min().item()
            # 数值保护:ψ<1e-3 处截断,标记为无效
            v = (kk / torch.clamp(psi, min=1e-3) ** 7).mean().item() * (4/3)*np.pi*R_MAX**3
        f = -k * S_b - v / 8.0
        flag = " <-- ψ 触 0!" if vmin < 0.02 else ""
        print(f"{k:>8.2f} {f:>12.4e} {vmin:>10.4f} {v/8.0:>12.4e}{flag}")


if __name__ == "__main__":
    setup_logging("A2", "a2q_kappa_probe")
    for lb in sys.argv[1:] or ["q20", "q17", "q24", "q63", "q10"]:
        try:
            probe(lb)
        except Exception as e:
            log.exception(f"{lb} 失败: {e}")

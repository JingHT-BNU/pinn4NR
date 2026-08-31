"""a2q_check_fast.py —— 校验快路径残差与全图自动微分残差数值等价。

对若干配置:同一批点、同一(未训练)模型,分别用
  快路径: 预计算 u_g/∇u_g/Δu_g 常数 + 仅对 MLP 部分自动微分
  全路径: u 对 x 全图自动微分(compute_parametric_pde_residual)
计算 PDE 残差与 Robin 残差,报告最大相对偏差。预期 ~1e-6 以下(快路径的
u_g 导数经 float32 存储,引入 ~1e-7 相对噪声)。
"""
import logging, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import physics
from a2q_prep import DATA_DIR, R_MAX
from a2q_model import make_model, param_vec
from a2q_train import Trainer
import argparse

log = logging.getLogger("paper.A2.a2q_check_fast")


def pde_residual(u, x, ma, xs, Ps, Ss):
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(3):
        g2 = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0]
        lap = lap + g2[:, i]
    ps = physics.psi_sing(x, ma, xs)
    kk = physics.bowen_york_KK(x, ma, xs, Ps, Ss)
    psi = torch.clamp(ps + u, min=1e-4)
    return lap + kk / (8.0 * psi ** 7)


def robin_residual(u, x, r_max):
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return ((x * g).sum(dim=1) + u) / r_max


class A:
    variant = "base"; steps = 10; lr = 3e-4; cfgs_per_step = 1; n_int_step = 8000
    noise_sigma_max = 0.06; rar_every = 250; seed = 0; exp_name = "smoke"


def main():
    setup_logging("A2", "a2q_check_fast")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = A()
    tr = Trainer.__new__(Trainer)
    import a2q_train as T
    Trainer.__init__(tr, args, dev)  # 需要 cfg/ref;refweight 不用,base 即可
    model = tr.model.to(dev)
    model.double()
    torch.manual_seed(3)

    for lb in ("q10", "q54", "q100"):
        d = tr.cfgs[lb]
        sqv = float(d["sq"]); wmin, wmax = float(d["wmin"]), float(d["wmax"])
        k = float(d["kappa"])
        p = param_vec(float(d["q"]), float(d["m2"]), dev)
        ma = torch.from_numpy(d["masses"]).double().to(dev)
        xs = torch.from_numpy(d["xs"]).double().to(dev)
        Ps = torch.from_numpy(d["Ps"]).double().to(dev)
        Ss = torch.from_numpy(d["Ss"]).double().to(dev)

        idx = np.arange(3000)
        xi_np = d["x_int"][idx]
        ps = torch.from_numpy(d["ps_int"][idx]).double().to(dev)
        kk = torch.from_numpy(d["kk_int"][idx]).double().to(dev)
        ug = torch.from_numpy(d["ug_int"][idx]).double().to(dev)
        gug = torch.from_numpy(d["grad_ug"][idx]).double().to(dev)
        lug = torch.from_numpy(d["lap_ug"][idx]).double().to(dev)
        w = (ug - wmin) / (wmax - wmin + 1e-8)

        # 快路径
        x1 = torch.from_numpy(xi_np).double().to(dev)
        x1.requires_grad_(True)
        R_fast, _, _ = tr._fast_residual(x1, p, k, sqv, ug, w, gug, lug, ps, kk,
                                         wmin, wmax)
        R_fast = R_fast.detach()

        # 全路径(模型 forward 内部现算 u_g,残差对 u 全图微分)
        x2 = torch.from_numpy(xi_np).double().to(dev)
        x2.requires_grad_(True)
        u2 = model(x2, ma, xs, Ps, Ss, p, k, wmin, wmax, sqv)
        R_full = pde_residual(u2, x2, ma, xs, Ps, Ss).detach()

        rel = ((R_fast - R_full).abs().max() / R_full.abs().max()).item()
        med = ((R_fast - R_full).abs().median() / R_full.abs().median()).item()
        log.info(f"{lb}: 残差快vs全 |ΔR|max/|R|max = {rel:.3e}  "
                 f"median比 = {med:.3e}  "
                 f"(|R|范围 {R_full.abs().min():.2e}~{R_full.abs().max():.2e})")

        # Robin
        xb_np = d["x_bnd"][:2000]
        ug_b = torch.from_numpy(d["ug_bnd"][:2000]).double().to(dev)
        gug_b = torch.from_numpy(d["grad_ug_b"][:2000]).double().to(dev)
        span = wmax - wmin + 1e-8
        w_b = (ug_b - wmin) / span
        x3 = torch.from_numpy(xb_np).double().to(dev)
        x3.requires_grad_(True)
        ub, phib, psib = model.forward_from_parts(x3, p, k, ug_b, w_b, sq=sqv)
        gpsib = torch.autograd.grad(psib.sum(), x3, create_graph=True)[0]
        gwb = gug_b / span
        gphib = gwb * psib.unsqueeze(1) + w_b.unsqueeze(1) * gpsib
        gub = k * (gug_b * (1.0 + phib).unsqueeze(1) + ug_b.unsqueeze(1) * gphib)
        rob_fast = (((x3 * gub).sum(1) + ub) / R_MAX).detach()
        x4 = torch.from_numpy(xb_np).double().to(dev)
        x4.requires_grad_(True)
        u4 = model(x4, ma, xs, Ps, Ss, p, k, wmin, wmax, sqv)
        rob_full = robin_residual(u4, x4, R_MAX).detach()
        relr = ((rob_fast - rob_full).abs().max() / rob_full.abs().max()).item()
        log.info(f"{lb}: Robin 快vs全 |Δ|max/|max| = {relr:.3e}")
    log.info("校验完成")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

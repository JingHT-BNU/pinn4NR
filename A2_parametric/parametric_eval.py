"""
parametric_eval.py —— 参数化 PINN 严格审查评估
===============================================

审查维度:
    1. base(q10) L2RE:全 47.7M 参考 + 均匀 101³ 参考(与 A1 报告口径一致)
    2. PDE 残差自检:所有训练/验证配置的残差均值/最大值
    3. 与引导解 κ·u_g 的对比:网络是否真的优于引导解(否则没意义)
    4. Robin 边界残差:u 在边界处行为
    5. 物理合理性:
       - u 幅值随 q 单调变化(质量越大,源项越强,u 越大? 需检查)
       - x 轴剖面平滑无锯齿
       - 远场 u→0
    6. 零样本泛化:未见过的 q 值(插值)解的连续性与物理合理性
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BBHConfig, TrainConfig
from data import sample_ball, sample_sphere_surface
from logutil import setup_logging
import physics
from parametric_model import ParamGuidedPINN, compute_parametric_pde_residual, compute_parametric_robin_residual
from parametric_train import build_bbh, TRAIN_PARAMS, VAL_PARAMS, predict, l2re

log = logging.getLogger("paper.A2.parametric_eval")

KAPPA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa_cache.json")


def load_model(path: str, device) -> ParamGuidedPINN:
    ckpt = torch.load(path, map_location=device)
    model = ParamGuidedPINN(n_params=2, c_init=ckpt["c"],
                            hidden_layers=4, hidden_neurons=128, n_freq=8).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.set_u_range(ckpt["u_min"], ckpt["u_max"])
    model.eval()
    return model  # 保持训练后的 float32 权重(与 predict 一致)


def pde_selfcheck(model, m1, m2, kappa, cfg, device, n=20000):
    """在独立采样点上检查 PDE 残差。"""
    rng = np.random.default_rng(7)
    x_int = sample_ball(n, cfg.R_max, rng).astype(np.float32)
    bb = build_bbh(m1, m2)
    ma = torch.tensor([m1, m2], dtype=torch.float64, device=device)
    xs = torch.tensor(list(bb.x_plus)+list(bb.x_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ps = torch.tensor(list(bb.P_plus)+list(bb.P_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ss = torch.tensor(list(bb.S_plus)+list(bb.S_minus), dtype=torch.float64, device=device).reshape(2,3)
    p = torch.tensor([[m1, m2]], dtype=torch.float64, device=device)

    res = []
    for i in range(0, n, 512):
        xi = torch.from_numpy(x_int[i:i+512]).float().to(device)
        xi.requires_grad_(True)
        u = model(xi, ma, xs, Ps, Ss, p, kappa)
        R = compute_parametric_pde_residual(u, xi, ma, xs, Ps, Ss)
        res.append(R.detach().abs().cpu().numpy())
    res = np.concatenate(res)
    return float(res.mean()), float(res.max())


def robin_selfcheck(model, m1, m2, kappa, cfg, device, n=5000):
    rng = np.random.default_rng(7)
    x_bnd = sample_sphere_surface(n, cfg.R_max, rng).astype(np.float32)
    bb = build_bbh(m1, m2)
    ma = torch.tensor([m1, m2], dtype=torch.float64, device=device)
    xs = torch.tensor(list(bb.x_plus)+list(bb.x_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ps = torch.tensor(list(bb.P_plus)+list(bb.P_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ss = torch.tensor(list(bb.S_plus)+list(bb.S_minus), dtype=torch.float64, device=device).reshape(2,3)
    p = torch.tensor([[m1, m2]], dtype=torch.float64, device=device)
    res = []
    for i in range(0, n, 512):
        xb = torch.from_numpy(x_bnd[i:i+512]).float().to(device)
        xb.requires_grad_(True)
        u = model(xb, ma, xs, Ps, Ss, p, kappa)
        R = compute_parametric_robin_residual(u, xb, cfg.R_max)
        res.append(R.detach().abs().cpu().numpy())
    res = np.concatenate(res)
    return float(res.mean()), float(res.max())


def guide_l2re(x_ref, u_ref, m1, m2, kappa, device):
    """引导解 κ·u_g 的 L2RE(基线对比)。"""
    bb = build_bbh(m1, m2)
    ma = torch.tensor([m1, m2], device=device)
    xs = torch.tensor(list(bb.x_plus)+list(bb.x_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ps = torch.tensor(list(bb.P_plus)+list(bb.P_minus), dtype=torch.float64, device=device).reshape(2,3)
    Ss = torch.tensor(list(bb.S_plus)+list(bb.S_minus), dtype=torch.float64, device=device).reshape(2,3)
    out = []
    with torch.no_grad():
        for i in range(0, x_ref.shape[0], 16384):
            xt = torch.from_numpy(x_ref[i:i+16384]).double().to(device)
            ug = physics.guide_u(xt, ma, xs, Ps, Ss)
            out.append((kappa * ug).cpu().numpy())
    return l2re(np.concatenate(out), u_ref)


def axis_profile(model, m1, m2, kappa, device, R=30.0, n=601):
    """x 轴剖面(y=z=0),验证平滑性和对称性。"""
    xs = np.linspace(-R, R, n)
    x_line = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1).astype(np.float32)
    u = predict(model, x_line, m1, m2, kappa, device)
    return xs, u


def check_monotone(profiles):
    """检查 x 轴剖面在奇点两侧是否单调(物理上 u 在奇点处最大,向两端衰减)。"""
    results = {}
    for label, (xs, u) in profiles.items():
        # 找两个峰(x=±3 附近)
        i_plus = np.argmax(u[xs > 0])
        i_minus = np.argmax(u[xs < 0])
        # 检查从峰向外是否递减(粗略:左峰向左、右峰向右)
        left_ok = u[:i_minus].max() <= u[i_minus] * 1.01
        right_ok = u[i_plus:].max() <= u[i_plus] * 1.01
        results[label] = (left_ok, right_ok)
    return results


def main():
    setup_logging("A2", "parametric_eval")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "parametric_a1", "model.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "reference_u.npz"))
    parser.add_argument("--reference-uniform", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "reference_u_uniform101_rcut0.3.npz"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    log.info(f"设备: {device}")
    log.info(f"加载模型: {args.model}")
    model = load_model(args.model, device)

    kc = json.load(open(KAPPA_CACHE))
    kc = {k: v["kappa"] for k, v in kc.items()}
    cfg = TrainConfig(R_max=30.0)

    log.info("\n" + "=" * 70)
    log.info("1. base(q10) L2RE —— 严格评估(与 A1 报告双口径一致)")
    log.info("=" * 70)
    ref = np.load(args.reference)
    u_pinn = predict(model, ref["x_ref"], 0.5, 0.5, kc["q10"], device)
    l2re_full = l2re(u_pinn, ref["u_ref"])
    ug = guide_l2re(ref["x_ref"], ref["u_ref"], 0.5, 0.5, kc["q10"], device)
    log.info(f"  [全 47.7M 参考] 参数化 PINN L2RE = {l2re_full:.4e}")
    log.info(f"  [全 47.7M 参考] 引导解 κ·u_g  L2RE = {ug:.4e}  (基线)")
    log.info(f"  → 网络相对引导解改进: {(1 - l2re_full/ug)*100:.1f}%")

    refu = np.load(args.reference_uniform)
    u_pinn2 = predict(model, refu["x_ref"], 0.5, 0.5, kc["q10"], device)
    l2re_uniform = l2re(u_pinn2, refu["u_ref"])
    log.info(f"  [均匀 101^3 参考] 参数化 PINN L2RE = {l2re_uniform:.4e}")
    log.info(f"  [参考] 单配置 base_a1(全参考) L2RE = 0.0067")
    log.info(f"  [参考] 论文 Table I L2RE = 0.017")

    log.info("\n" + "=" * 70)
    log.info("2. PDE 残差自检(独立采样点)—— 所有配置")
    log.info("=" * 70)
    log.info(f"  {'配置':<6}{'q':<6}{'PDE残差均值':<14}{'PDE残差最大':<14}{'Robin均值':<12}{'Robin最大'}")
    for m1, m2, label in TRAIN_PARAMS + VAL_PARAMS:
        k = kc.get(label)
        if k is None:
            continue
        pm, pM = pde_selfcheck(model, m1, m2, k, cfg, device)
        rm, rM = robin_selfcheck(model, m1, m2, k, cfg, device)
        tag = "训练" if label in [l for _, _, l in TRAIN_PARAMS] else "零样本"
        log.info(f"  {label:<6}{m2/m1:<6.2f}{pm:<14.3e}{pM:<14.3e}{rm:<12.3e}{rM:<10.3e} {tag}")

    log.info("\n" + "=" * 70)
    log.info("3. 物理合理性:x 轴剖面")
    log.info("=" * 70)
    profiles = {}
    for m1, m2, label in TRAIN_PARAMS + VAL_PARAMS:
        k = kc.get(label)
        if k is None:
            continue
        xs, u = axis_profile(model, m1, m2, k, device)
        profiles[label] = (xs, u)
        ug_peak = k * 0.02  # 引导解峰值量级(粗略)
        log.info(f"  {label}: u范围[{u.min():.4e},{u.max():.4e}] "
              f"u(0)={u[np.argmin(np.abs(xs))]:.4e} "
              f"峰={u.max():.4e}")

    mon = check_monotone(profiles)
    log.info(f"  单调性检查(峰向外递减): {mon}")

    # 保存剖面供可视化
    np.savez("runs/parametric_a1/figs/axis_profiles.npz",
             **{k: np.stack(v) for k, v in profiles.items()})
    log.info(f"\n剖面已保存: runs/parametric_a1/figs/axis_profiles.npz")

    log.info("\n" + "=" * 70)
    log.info("4. 零样本泛化连续性检查(q 从 0.5 到 2.0,网络输出应平滑变化)")
    log.info("=" * 70)
    # 在 x=0 处检查 u 随 m2 的变化(应为平滑函数)
    x0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    us = []
    m2s = np.linspace(0.25, 1.0, 16)
    for m2 in m2s:
        # κ 用附近缓存值插值(简化)
        k = np.interp(m2, [0.25, 0.35, 0.5, 0.65, 0.8, 1.0],
                      [kc["q05"], kc["q07"], kc["q10"], kc["q13"], kc["q16"], kc["q20"]])
        u = predict(model, x0, 0.5, m2, float(k), device)[0]
        us.append(u)
    us = np.array(us)
    # 平滑性:一阶差分的变化率不应突兀
    du = np.diff(us)
    rel_change = np.abs(du[1:] - du[:-1]) / (np.abs(us[1:-1]) + 1e-12)
    log.info(f"  u(x=0) 随 m2 变化: {['%.4e' % v for v in us]}")
    log.info(f"  相对变化率的波动(max): {rel_change.max():.4f} (应远小于 1,说明平滑)")

    log.info("审查完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
"""
evaluate.py —— 评估模块:推理 + L2RE + 残差自检
================================================

评估指标:
    - L2RE(Eq.18):相对 L2 误差,需要 TwoPunctures 参考解(npz)。
        如果没有参考解,退化为"残差自检"指标:
            - PDE 残差均值/最大值(越小越接近方程解)
            - 边界残差均值/最大值
    - 推理:对任意坐标批量计算 u_θ(训练好的模型)。

论文 Eq.(18):
    L2RE = sqrt( Σ_i |u_PINN(x_i) − u_TP(x_i)|² / Σ_i |u_TP(x_i)|² )
    其中 TP = TwoPunctures 参考解,在 TP 谱网格节点上计算。
"""

import logging
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import physics
from config import TrainConfig
from model import GuidedPINN

log = logging.getLogger("paper.A1.evaluate")


def predict_u(model: GuidedPINN, x: np.ndarray, device) -> np.ndarray:
    """推理:对任意坐标批量计算修正项 u_θ(训练后的模型)。

    Args:
        model: 训练好的 GuidedPINN
        x    : 场点坐标, shape (M, 3)
        device: torch 设备

    Returns:
        u_theta: shape (M,),模型输出的修正项解
    """
    model.eval()
    out = []
    batch = 8192
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.from_numpy(x[i:i+batch]).float().to(device)
            u = model(xb)                       # 引导解与窗函数在模型内部计算
            out.append(u.cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_l2re(u_pinn: np.ndarray, u_ref: np.ndarray) -> float:
    """相对 L2 误差 L2RE(Eq.18)。

    Args:
        u_pinn: PINN 预测, shape (M,)
        u_ref : TwoPunctures 参考, shape (M,)(与 u_pinn 同网格)

    Returns:
        L2RE: 浮点数(论文 Table I/II 的指标)
    """
    num = np.sum((u_pinn - u_ref) ** 2)
    den = np.sum(u_ref ** 2)
    if den < 1e-30:
        return float("nan")
    return float(np.sqrt(num / den))


def residual_selfcheck(model: GuidedPINN, data, cfg: TrainConfig,
                       device, n_check: int = 10000) -> Dict[str, float]:
    """残差自检:在独立采样点上计算 PDE/边界残差统计(无参考解时的评估)。

    Args:
        model : 训练好的模型
        data  : DataBundle
        cfg   : TrainConfig
        device: torch 设备
        n_check: 自检采样点数

    Returns:
        {'pde_res_mean': ..., 'pde_res_max': ..., 'bnd_res_mean': ...,
         'bnd_res_max': ...} 等统计量
    """
    import data as data_mod
    rng = np.random.default_rng(cfg.seed + 1)
    x_int = data_mod.sample_ball(n_check, cfg.R_max, rng).astype(np.float32)
    x_bnd = data_mod.sample_sphere_surface(n_check // 4, cfg.R_max, rng).astype(np.float32)

    # 预计算解析量(psi_sing, K̄K̄)为张量(不含 u,不需要梯度),搬到与模型相同设备
    m_t = torch.from_numpy(data.masses).double().to(device)
    xs_t = torch.from_numpy(data.xs).double().to(device)
    P_t = torch.from_numpy(data.Ps).double().to(device)
    S_t = torch.from_numpy(data.Ss).double().to(device)

    def _res(x_np: np.ndarray, is_bnd: bool):
        model.eval()
        xb = torch.from_numpy(x_np).float().to(device)
        xb = xb.requires_grad_(True)
        u = model(xb)                                   # (N,) u_θ(含可微引导解)
        xd = xb.double()
        if is_bnd:
            R = physics.robin_residual(u, xb, cfg.R_max)
        else:
            ps = physics.psi_sing(xd, m_t, xs_t).float()
            kk = physics.bowen_york_KK(xd, m_t, xs_t, P_t, S_t).float()
            R = physics.pde_residual(u, xb, ps, kk)
        return R.detach().abs().cpu().numpy()

    r_int = _res(x_int, is_bnd=False)
    r_bnd = _res(x_bnd, is_bnd=True)
    return {
        "pde_res_mean": float(r_int.mean()),
        "pde_res_max": float(r_int.max()),
        "bnd_res_mean": float(r_bnd.mean()),
        "bnd_res_max": float(r_bnd.max()),
    }


def evaluate_on_grid(model: GuidedPINN, data, cfg: TrainConfig,
                     device, reference_path: Optional[str] = None) -> Dict:
    """完整评估:优先 L2RE(有参考解),否则残差自检。

    Args:
        model         : 训练好的模型
        data          : DataBundle
        cfg           : TrainConfig
        device        : torch 设备
        reference_path: TwoPunctures 参考解 npz 路径(可选)

    Returns:
        {'metric': 'L2RE' 或 'residual', 以及各项指标}
    """
    import data as data_mod
    ref = data_mod.load_reference(reference_path)
    if ref is not None:
        x_ref, u_ref = ref
        u_pinn = predict_u(model, x_ref, device)
        l2re = compute_l2re(u_pinn, u_ref)
        log.info(f"[评估] 参考解网格 {x_ref.shape[0]} 点, L2RE = {l2re:.4e}")
        return {"metric": "L2RE", "l2re": l2re}
    stats = residual_selfcheck(model, data, cfg, device)
    log.info("[评估] 未找到参考解,使用残差自检:")
    for k, v in stats.items():
        log.info(f"  {k} = {v:.3e}")
    return {"metric": "residual", **stats}

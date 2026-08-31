"""
train.py —— 训练循环:Adam + 复合损失 + EMA 损失平衡
=====================================================

对应论文:
    - Eq.(11-12) : L = w2·L2 + w∞·soft-L∞ + wrob·LBC
    - Eq.(13)    : L2 = (1/NΩ) Σ R_i²
    - Eq.(14)    : soft-L∞ = (1/β)ln(Σ exp(β|R_i|)) − (1/β)ln NΩ
    - Eq.(15)    : LBC = (wrob/N∂Ω) Σ R_B,i²
    - Eq.(16-17) : EMA 损失平衡 L̃_k = L_k / L̄_k, L̄_k(t)=αL̄_k(t-1)+(1-α)L_k(t)

训练流程(两阶段):
    阶段一(数据准备时完成):由散度定理求 κ
    阶段二(本模块):固定 κ,训练网络 h_θ,最小化损失
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import physics
from config import TrainConfig
from model import GuidedPINN

log = logging.getLogger("paper.A1.train")


class Trainer:
    """PINN 训练器:封装损失计算、EMA 平衡、Adam 优化。

    Args:
        cfg   : TrainConfig
        model : GuidedPINN 模型
        device: torch 设备
    """

    def __init__(self, cfg: TrainConfig, model: GuidedPINN, device):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.ema: Dict[str, float] = {}        # 各损失项的 EMA(用于归一化)
        self.history: Dict[str, List[float]] = {"L2": [], "softLinf": [],
                                                "LBC": [], "total": []}

    # ------------------------------------------------------------------
    def _ema_balance(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        """EMA 损失平衡(Eq.16-17):L̃ = L / L̄。

        第一步用 L 本身初始化 L̄(等价于 α 取 0 的 EMA)。

        Args:
            name: 损失项名('L2' / 'softLinf' / 'LBC')
            loss: 原始损失值(标量张量)

        Returns:
            归一化损失 L̃(量级约 O(1),不会独大)
        """
        l = loss.item()
        if name not in self.ema:
            self.ema[name] = l
        else:
            self.ema[name] = (self.cfg.ema_alpha * self.ema[name]
                              + (1 - self.cfg.ema_alpha) * l)
        return loss / (self.ema[name] + 1e-12)

    # ------------------------------------------------------------------
    def _compute_losses(self, d: Dict[str, torch.Tensor]):
        """计算三个损失分量(L2, soft-L∞, LBC)。

        Args:
            d: DataBundle.to_torch() 返回的字典(含 x_int/x_bnd 与解析量)

        Returns:
            (l2, soft_linf, lbc) 三个标量张量
        """
        x_int = d["x_int"].clone().requires_grad_(True)      # (NΩ,3) 内部点
        x_bnd = d["x_bnd"].clone().requires_grad_(True)      # (N∂Ω,3) 边界点

        # ---- 内部点:PDE 残差(模型内部实时算 u_g/W,保证 Δu 解析贡献完整) ----
        u_int = self.model(x_int)                            # (NΩ,) u_θ
        R = physics.pde_residual(u_int, x_int, d["ps_int"], d["kk_int"])  # (NΩ,) 残差

        # ---- 峰值区域加权:奇点附近残差权重更大,抑制锯齿 ----
        if self.cfg.peak_weighting and self.cfg.peak_weight_power > 0:
            # 计算每个内部点到最近奇点的距离
            masses = self.model.masses
            xs = self.model.xs
            min_dist = torch.full((x_int.shape[0],), float("inf"),
                                  device=x_int.device, dtype=x_int.dtype)
            for n in range(xs.shape[0]):
                dist = (x_int - xs[n]).norm(dim=1)
                min_dist = torch.min(min_dist, dist)
            # 权重 ∝ 1/r^power,在 1e-2 处截断(避免奇点处除零)
            r_safe = torch.clamp(min_dist, min=1e-2)
            w = 1.0 / (r_safe ** self.cfg.peak_weight_power)
            # 归一化权重(均值=1,保持整体损失量级不变)
            w = w / (w.mean() + 1e-12)
            l2 = (w * (R ** 2)).mean()                       # Eq.(13) 加权 L2
        else:
            l2 = (R ** 2).mean()                             # Eq.(13) L2

        # ---- soft-L∞(Eq.14):log-sum-exp 光滑 max ----
        beta = self.cfg.beta
        if beta > 0:
            soft_linf = (1.0 / beta) * torch.logsumexp(beta * R.abs(), dim=0) \
                        - (1.0 / beta) * np.log(R.shape[0])
            soft_linf = torch.clamp(soft_linf, min=0.0)      # 数值噪声保护
        else:
            soft_linf = torch.zeros_like(l2)

        # ---- 边界点:Robin 残差 ----
        u_bnd = self.model(x_bnd)                            # (N∂Ω,) u_θ
        R_B = physics.robin_residual(u_bnd, x_bnd, self.cfg.R_max)  # (N∂Ω,)
        lbc = (R_B ** 2).mean()                              # Eq.(15) 内部平方项

        return l2, soft_linf, lbc

    # ------------------------------------------------------------------
    def train_step(self, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练:计算损失 → EMA 平衡 → 反向传播 → Adam 更新。

        Args:
            d: 数据字典

        Returns:
            {损失名: 值} 字典(记录用)
        """
        self.optimizer.zero_grad()
        # 将模型和数据转为 float64 以消除 softLinf 的 FP32 数值噪声
        # (logsumexp 对精度敏感;FP32→~1e-7, FP64→~1e-16)
        self.model.double()
        d64 = {k: v.double() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
                for k, v in d.items()}
        l2, soft_linf, lbc = self._compute_losses(d64)

        # ---- EMA 平衡(Eq.16-17)后加权组合 ----
        l2_b, sl_b, lbc_b = (self._ema_balance("L2", l2),
                             self._ema_balance("softLinf", soft_linf),
                             self._ema_balance("LBC", lbc))
        total = (self.cfg.w2 * l2_b
                 + self.cfg.w_inf * sl_b
                 + self.cfg.w_rob * lbc_b)                   # Eq.(11-12)

        total.backward()
        self.optimizer.step()
        self.model.float()                                   # 恢复 float32

        # ---- 记录(原始量,未归一化) ----
        out = {"L2": l2.item(), "softLinf": soft_linf.item(),
               "LBC": lbc.item(), "total": total.item()}
        for k, v in out.items():
            self.history[k].append(v)
        return out

    # ------------------------------------------------------------------
    def train(self, d: Dict[str, torch.Tensor], n_steps: Optional[int] = None,
              log_every: int = 100) -> Dict[str, List[float]]:
        """完整训练循环。

        Args:
            d       : 数据字典
            n_steps : 训练步数(默认取 cfg.n_steps)
            log_every: 每多少步打印一次日志

        Returns:
            history: 各损失的历史曲线 {'L2': [...], ...}
        """
        n_steps = n_steps or self.cfg.n_steps
        t0 = time.time()
        for step in range(1, n_steps + 1):
            out = self.train_step(d)
            if step % log_every == 0 or step == 1:
                log.info(f"[step {step:6d}/{n_steps}] L2={out['L2']:.3e} "
                      f"softLinf={out['softLinf']:.3e} LBC={out['LBC']:.3e} "
                      f"total={out['total']:.3e}  ({time.time()-t0:.0f}s)")
        log.info(f"训练完成,用时 {time.time()-t0:.1f}s")
        return self.history

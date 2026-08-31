"""
multi_param_model.py —— 多参数参数化 PINN 模型
================================================

架构设计:
    输入: (x, params) — 坐标 + 8维物理参数
    输出: u_θ(x; params) — 修正项解

参数向量 (8维):
    [m_plus, m_minus, x_plus_x, x_minus_x, P_plus_y, P_minus_y, S_plus_z, S_minus_z]

参数范围:
    m ∈ [0.25, 3.0]      质量
    x ∈ [-8.0, 8.0]      位置(x轴)
    P ∈ [-0.5, 0.5]      线动量(y分量)
    S ∈ [0.0, 0.4]       自旋(z分量)

核心改进(vs 单参数版):
    1. 8维参数空间(质量+位置+动量+自旋)
    2. 更大网络: 6×256 + FiLM 调制
    3. 更高频正弦编码(n_freq_coord=10, n_freq_param=12)
    4. 参数归一化到 [-1,1] 后编码
    5. ~1.15M 可训练参数
"""

import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

import physics

# ── 参数范围定义 ──────────────────────────────────────────────
# 8维参数: [m_plus, m_minus, x_plus, x_minus, P_plus_y, P_minus_y, S_plus_z, S_minus_z]
PARAM_NAMES = ["m_plus", "m_minus", "x_plus", "x_minus",
               "P_plus_y", "P_minus_y", "S_plus_z", "S_minus_z"]

PARAM_LO = np.array([0.25, 0.25,  2.0, -8.0,  0.05, -0.5, 0.0, 0.0])
PARAM_HI = np.array([3.0,  3.0,   8.0, -2.0,   0.5,  -0.05, 0.4, 0.4])
PARAM_RANGE = PARAM_HI - PARAM_LO

# base case 在归一化参数空间中的坐标
BASE_RAW = np.array([0.5, 0.5, 3.0, -3.0, 0.2, -0.2, 0.0, 0.0])
BASE_NORM = (BASE_RAW - PARAM_LO) / PARAM_RANGE * 2.0 - 1.0


def normalize_params(raw: np.ndarray) -> np.ndarray:
    """原始参数 → [-1, 1] 归一化。raw: (..., 8) → (..., 8)"""
    return (raw - PARAM_LO) / PARAM_RANGE * 2.0 - 1.0


def denormalize_params(norm: np.ndarray) -> np.ndarray:
    """[-1, 1] 归一化 → 原始参数。norm: (..., 8) → (..., 8)"""
    return (norm + 1.0) / 2.0 * PARAM_RANGE + PARAM_LO


def build_bbh_from_params(raw_params: np.ndarray):
    """从8维原始参数构建 BBHConfig 所需的物理量。

    Args:
        raw_params: shape (8,), 原始物理参数

    Returns:
        masses: (2,), xs: (2,3), Ps: (2,3), Ss: (2,3)
    """
    m_plus, m_minus, x_plus, x_minus, P_plus_y, P_minus_y, S_plus_z, S_minus_z = raw_params
    masses = np.array([m_plus, m_minus])
    xs = np.array([[x_plus, 0.0, 0.0], [x_minus, 0.0, 0.0]])
    Ps = np.array([[0.0, P_plus_y, 0.0], [0.0, P_minus_y, 0.0]])
    Ss = np.array([[0.0, 0.0, S_plus_z], [0.0, 0.0, S_minus_z]])
    return masses, xs, Ps, Ss


# ── 正弦位置编码 ─────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """正弦位置编码: 将标量输入映射到高频特征。"""

    def __init__(self, n_freq: int = 8, scale: float = 1.0):
        super().__init__()
        self.n_freq = n_freq
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(torch.linspace(0, self.n_freq - 1, self.n_freq,
                                         device=x.device, dtype=x.dtype) * self.scale)
        angles = x.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ── FiLM 调制 ────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, in_features: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_features)
        self.beta = nn.Linear(cond_dim, in_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.gamma(cond)) + self.beta(cond)


# ── 主网络 ───────────────────────────────────────────────────

class MultiParamConditionMLP(nn.Module):
    """多参数条件 MLP。

    架构:
        坐标 3D → 正弦编码(10 freq) → coord_net → 63→256→256
        参数 8D → 归一化 → 正弦编码(12 freq) → param_net → 33→256→256
        FiLM(x_enc, p_enc) → 6×256 共享层 → 1
    """

    def __init__(self, n_params: int = 8, hidden_layers: int = 6,
                 hidden_neurons: int = 256, n_freq_coord: int = 10,
                 n_freq_param: int = 12):
        super().__init__()
        self.n_params = n_params

        self.coord_embed = SinusoidalEmbedding(n_freq=n_freq_coord, scale=0.5)
        self.param_embed = SinusoidalEmbedding(n_freq=n_freq_param, scale=0.35)

        coord_enc_dim = 3 + 3 * 2 * n_freq_coord      # 3 + 60 = 63
        param_enc_dim = n_params + n_params * 2 * n_freq_param  # 8 + 192 = 200

        self.coord_net = nn.Sequential(
            nn.Linear(coord_enc_dim, hidden_neurons),
            nn.SiLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.SiLU(),
        )
        self.param_net = nn.Sequential(
            nn.Linear(param_enc_dim, hidden_neurons),
            nn.SiLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.SiLU(),
        )

        self.film = FiLM(hidden_neurons, hidden_neurons)

        layers = []
        in_dim = hidden_neurons
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_neurons))
            layers.append(nn.SiLU())
            in_dim = hidden_neurons
        layers.append(nn.Linear(in_dim, 1))
        self.shared = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _embed_coord(self, x: torch.Tensor) -> torch.Tensor:
        emb = [x]
        for i in range(self.coord_embed.n_freq):
            freq = np.exp(i * self.coord_embed.scale)
            emb.append(torch.sin(x * freq))
            emb.append(torch.cos(x * freq))
        return torch.cat(emb, dim=-1)

    def _embed_param(self, p: torch.Tensor) -> torch.Tensor:
        emb = [p]
        for i in range(self.param_embed.n_freq):
            freq = np.exp(i * self.param_embed.scale)
            emb.append(torch.sin(p * freq))
            emb.append(torch.cos(p * freq))
        return torch.cat(emb, dim=-1)

    def forward(self, x: torch.Tensor, params_norm: torch.Tensor) -> torch.Tensor:
        """前向: h_θ(x, params_norm)。

        Args:
            x          : 场点坐标, shape (N, 3)
            params_norm: 归一化参数, shape (N, 8) 或 (1, 8)

        Returns:
            h: shape (N, 1)
        """
        x_enc = self.coord_net(self._embed_coord(x))
        if params_norm.shape[0] == 1 and x.shape[0] > 1:
            params_norm = params_norm.expand(x.shape[0], -1)
        p_enc = self.param_net(self._embed_param(params_norm))
        x_mod = self.film(x_enc, p_enc)
        return self.shared(x_mod)


class MultiParamGuidedPINN(nn.Module):
    """多参数引导式硬约束 PINN。

    u_θ(x; params) = κ(params) · u_g(x; params) · [1 + c · W(x) · tanh(h_θ(x, params_norm))]

    窗函数 W = tanh(ug / u_scale), u_scale 默认 0.05(与中场 ug 量级匹配):
    - 远场/边界 ug→0 → W→0, 修正自动消失(Robin BC 不被破坏);
    - 近奇点 ug≫u_scale → W→1, 修正全力工作;
    - 逐点确定, 与采样/配置统计无关, 训练与评估严格一致。

    历史教训: 旧版 W = (ug-u_min)/(u_max-u_min) 用全部 300 配置的全局
    ug 范围([7e-5, 1.45], 跨 4 个数量级), 对 base 等大多数配置
    W≈0.001~0.016, 修正头被压到 1% 量级, 30000 步 L_ref 纹丝不动
    (A2 只有 6 个近邻配置, 全局范围≈单配置范围, W≈O(0.1-1) 所以能学)。
    """

    def __init__(self, n_params: int = 8, c_init: float = 0.2,
                 hidden_layers: int = 6, hidden_neurons: int = 256,
                 n_freq_coord: int = 10, n_freq_param: int = 12,
                 u_scale: float = 0.05,
                 amp_mode: str = "raw", c_max: float = 1.0):
        super().__init__()
        self.mlp = MultiParamConditionMLP(n_params, hidden_layers, hidden_neurons,
                                          n_freq_coord, n_freq_param)
        self.amp_mode = amp_mode
        self.c_max = float(c_max)
        if amp_mode == "sigmoid":
            # v5: c = c_max·σ(c_raw), 有界于 (0, c_max)。v4 固定幅值 c=0.3351 时
            # 修正因子 1±c·W·tanh(h) 只能覆盖 [0.665, 1.335]× 引导解, 而谱参考解
            # 表明 0059/0164 针尖需要压到 0.5×/0.38× —— 幅值上限不够。
            # sigmoid 参数化既放开上限又保证 c>0(防止修正因子变号)。
            c0 = min(max(c_init / self.c_max, 1e-4), 1.0 - 1e-4)
            rho0 = float(np.log(c0 / (1.0 - c0)))
            self.c_raw = nn.Parameter(torch.tensor(rho0, dtype=torch.float64))
        else:
            self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float64))
        self.register_buffer("u_scale", torch.tensor(u_scale, dtype=torch.float64))

    def effective_c(self) -> float:
        """当前有效幅值 c(日志/保存用)。"""
        if self.amp_mode == "sigmoid":
            return float(self.c_max * torch.sigmoid(self.c_raw).item())
        return float(self.c.item())

    def set_u_scale(self, u_scale: float):
        self.u_scale = torch.tensor(u_scale, dtype=torch.float64,
                                    device=self.u_scale.device)

    def forward(self, x: torch.Tensor, masses: torch.Tensor,
                xs: torch.Tensor, Ps: torch.Tensor, Ss: torch.Tensor,
                params_norm: torch.Tensor, kappa: float) -> torch.Tensor:
        """前向: u_θ(x; params)。

        Args:
            x           : 场点坐标, shape (N, 3)
            masses      : 奇点质量, shape (2,)
            xs          : 奇点位置, shape (2, 3)
            Ps          : 奇点线动量, shape (2, 3)
            Ss          : 奇点自旋, shape (2, 3)
            params_norm : 归一化参数, shape (1, 8) 或 (N, 8)
            kappa       : 预计算全局尺度因子

        Returns:
            u_theta: shape (N,)
        """
        ug = physics.guide_u(x, masses, xs, Ps, Ss).to(dtype=x.dtype)
        w = torch.tanh(ug / self.u_scale.to(dtype=x.dtype))
        h = self.mlp(x, params_norm.to(dtype=x.dtype))
        h = torch.tanh(h.squeeze(-1))
        if self.amp_mode == "sigmoid":
            c_val = (self.c_max * torch.sigmoid(self.c_raw)).to(dtype=x.dtype)
        else:
            c_val = self.c.to(dtype=x.dtype)
        return kappa * ug * (1.0 + c_val * w * h)


def compute_pde_residual(u, x, masses, xs, Ps, Ss):
    """PDE 残差: R = Δu + (1/8)ψ^{-7} K̄K̄"""
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(3):
        g2 = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0]
        lap = lap + g2[:, i]
    psi_s = physics.psi_sing(x, masses, xs)
    kk = physics.bowen_york_KK(x, masses, xs, Ps, Ss)
    psi = torch.clamp(psi_s + u, min=1e-4)
    return lap + (1.0 / 8.0) * kk / (psi ** 7)


def compute_robin_residual(u, x, R_max):
    """Robin 边界残差: R_B = (x·∇u + u)/r"""
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    xg = (x * g).sum(dim=1)
    return (xg + u) / R_max

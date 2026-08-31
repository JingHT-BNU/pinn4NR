"""
parametric_model.py —— 参数化 PINN 模型(改进版)
=================================================

架构设计:
    输入: (x, m1, m2) — 坐标 + 物理参数
    输出: u_θ(x; m1, m2) — 修正项解

核心思路:将论文的引导式硬约束 ansatz 参数化:
    u_θ(x; params) = κ(params) · u_g(x; params) · [1 + c · W(x; params) · tanh(h_θ(x, params))]

改进点(v2):
    1. κ 使用预计算值(查表+线性插值),而非网络预测(避免 QMC 噪声)
    2. 条件 MLP 使用 FiLM 调制(Feature-wise Linear Modulation),更高效地注入参数信息
    3. 增加网络容量:4×128
    4. 参数编码使用正弦位置编码(类似 NeRF),增强对参数变化的敏感性
"""

import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

import physics


class SinusoidalEmbedding(nn.Module):
    """正弦位置编码:将标量输入映射到高频特征。"""

    def __init__(self, n_freq: int = 8, scale: float = 1.0):
        super().__init__()
        self.n_freq = n_freq
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 1) → (..., 2*n_freq)"""
        freqs = torch.exp(torch.linspace(0, self.n_freq - 1, self.n_freq,
                                         device=x.device, dtype=x.dtype) * self.scale)
        angles = x.unsqueeze(-1) * freqs  # (..., 1, n_freq)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., 2*n_freq)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation:用条件向量调制特征。"""

    def __init__(self, in_features: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_features)
        self.beta = nn.Linear(cond_dim, in_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (N, in_features), cond: (N, cond_dim) → (N, in_features)"""
        return x * (1 + self.gamma(cond)) + self.beta(cond)


class ParamConditionMLP(nn.Module):
    """条件 MLP:输入 (x, params) → 输出 h_θ。

    架构:
        - 坐标:3 维 → 正弦编码 → 坐标编码网络
        - 参数:2 维 → 正弦编码 → 参数编码网络
        - 拼接后通过 FiLM 调制的共享隐藏层
    """

    def __init__(self, n_params: int = 2, hidden_layers: int = 4,
                 hidden_neurons: int = 128, n_freq: int = 8):
        super().__init__()
        # 正弦编码
        self.coord_embed = SinusoidalEmbedding(n_freq=n_freq, scale=0.5)
        self.param_embed = SinusoidalEmbedding(n_freq=n_freq, scale=1.0)

        coord_enc_dim = 3 + 3 * 2 * n_freq  # 原始坐标 + 正弦编码
        param_enc_dim = n_params + n_params * 2 * n_freq  # 原始参数 + 正弦编码

        # 坐标编码网络
        self.coord_net = nn.Sequential(
            nn.Linear(coord_enc_dim, hidden_neurons),
            nn.SiLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.SiLU(),
        )
        # 参数编码网络
        self.param_net = nn.Sequential(
            nn.Linear(param_enc_dim, hidden_neurons),
            nn.SiLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.SiLU(),
        )
        # FiLM 调制的共享隐藏层
        self.film = FiLM(hidden_neurons, hidden_neurons)
        layers = []
        in_dim = hidden_neurons
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_neurons))
            layers.append(nn.SiLU())
            in_dim = hidden_neurons
        layers.append(nn.Linear(in_dim, 1))
        self.shared = nn.Sequential(*layers)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _embed_coord(self, x: torch.Tensor) -> torch.Tensor:
        """坐标正弦编码。"""
        emb = [x]
        for i in range(self.coord_embed.n_freq):
            freq = np.exp(i * self.coord_embed.scale)
            emb.append(torch.sin(x * freq))
            emb.append(torch.cos(x * freq))
        return torch.cat(emb, dim=-1)

    def _embed_param(self, p: torch.Tensor) -> torch.Tensor:
        """参数正弦编码。"""
        emb = [p]
        for i in range(self.param_embed.n_freq):
            freq = np.exp(i * self.param_embed.scale)
            emb.append(torch.sin(p * freq))
            emb.append(torch.cos(p * freq))
        return torch.cat(emb, dim=-1)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """前向:h_θ(x, params)。

        Args:
            x     : 场点坐标, shape (N, 3)
            params: 物理参数, shape (N, n_params) 或 (1, n_params)

        Returns:
            h: shape (N, 1)
        """
        # 编码
        x_enc = self.coord_net(self._embed_coord(x))  # (N, hidden)
        if params.shape[0] == 1 and x.shape[0] > 1:
            params = params.expand(x.shape[0], -1)
        p_enc = self.param_net(self._embed_param(params))  # (N, hidden)
        # FiLM 调制
        x_mod = self.film(x_enc, p_enc)  # (N, hidden)
        return self.shared(x_mod)  # (N, 1)


class ParamGuidedPINN(nn.Module):
    """参数化引导式硬约束 PINN。

    u_θ(x; params) = κ(params) · u_g(x; params) · [1 + c · W(x; params) · tanh(h_θ(x, params))]
    """

    def __init__(self, n_params: int = 2, c_init: float = 0.2,
                 hidden_layers: int = 4, hidden_neurons: int = 128,
                 n_freq: int = 8):
        super().__init__()
        self.mlp = ParamConditionMLP(n_params, hidden_layers, hidden_neurons, n_freq)
        # c 作为可学习参数
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float64))
        # κ 使用预计算值(查表),不通过网络学习
        self.register_buffer("u_min", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("u_max", torch.tensor(1.0, dtype=torch.float64))

    def set_u_range(self, u_min: float, u_max: float):
        self.u_min = torch.tensor(u_min, dtype=torch.float64, device=self.u_min.device)
        self.u_max = torch.tensor(u_max, dtype=torch.float64, device=self.u_max.device)

    def forward(self, x: torch.Tensor, masses: torch.Tensor,
                xs: torch.Tensor, Ps: torch.Tensor, Ss: torch.Tensor,
                params: torch.Tensor, kappa: float) -> torch.Tensor:
        """前向:u_θ(x; params)。

        Args:
            x     : 场点坐标, shape (N, 3)(需要梯度)
            masses: 奇点质量, shape (2,)
            xs    : 奇点位置, shape (2, 3)
            Ps    : 奇点线动量, shape (2, 3)
            Ss    : 奇点自旋, shape (2, 3)
            params: 物理参数, shape (1, n_params)
            kappa : 预计算的全局尺度因子(标量)

        Returns:
            u_theta: shape (N,)
        """
        ug = physics.guide_u(x, masses, xs, Ps, Ss)  # (N,) 可微
        ug = ug.to(dtype=x.dtype)
        w = (ug - self.u_min.to(dtype=x.dtype)) / \
            (self.u_max.to(dtype=x.dtype) - self.u_min.to(dtype=x.dtype) + 1e-8)  # (N,)
        h = self.mlp(x, params.to(dtype=x.dtype))  # (N, 1)
        h = torch.tanh(h.squeeze(-1))  # (N,)
        c_val = self.c.to(dtype=x.dtype)
        return kappa * ug * (1.0 + c_val * w * h)  # (N,)


def compute_parametric_pde_residual(u, x, masses, xs, Ps, Ss):
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


def compute_parametric_robin_residual(u, x, R_max):
    """Robin 边界残差: R_B = (x·∇u + u)/r"""
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    xg = (x * g).sum(dim=1)
    return (xg + u) / R_max
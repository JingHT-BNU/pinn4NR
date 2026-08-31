"""
model.py —— PINN 模型:MLP 网络 + 引导式硬约束 ansatz
=====================================================

论文 Eq.(5) 的引导式硬约束 ansatz:

    u_θ(x) = κ u_g(x) [1 + c W(x) tanh(h_θ(x))]

其中:
    - u_g : 解析引导解 physics.guide_u(x)(Lousto-Zlochower 公式)
    - κ   : 全局尺度因子(两阶段训练阶段一确定)
    - c   : 修正幅度超参数(论文基准算例 c=0.2)
    - W   : 窗函数 Eq.(7),取值 [0,1]
    - h_θ : 唯一的网络输出(3 层 × 64 神经元,SiLU 激活)

重要设计点:u_g 与 W 在 forward 中**实时从 x 计算**(而不是预缓存常数),
因为残差 Δu_θ 需要 Δu_g、ΔW 的解析贡献(它们都是 x 的可微函数)。
psi_sing 与 K̄K̄ 是已知源项(不含 u),可以预缓存以提速。

网络只学"修正比例场" h_θ,最终解 u_θ 的结构、量级、符号都被 ansatz 强制保证:
    - tanh 把修正压到 (-1,1),修正项永不爆炸
    - 乘法形式保证修正与局部解值成比例
"""

import os, sys
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn

import physics
from config import PINNConfig


class SiLUMLP(nn.Module):
    """全连接 MLP:输入 3 维坐标,输出 1 维修正场 h_θ。

    论文 §II.B.4:3 个隐藏层,每层 64 神经元,SiLU 激活。

    Args:
        cfg: PINNConfig(隐藏层数、每层神经元数、激活函数)
    """

    def __init__(self, cfg: PINNConfig):
        super().__init__()
        layers = []
        in_dim = 3
        for _ in range(cfg.hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_neurons))
            layers.append(self._act(cfg.activation))
            in_dim = cfg.hidden_neurons
        layers.append(nn.Linear(in_dim, 1))       # 输出层:标量 h_θ
        self.net = nn.Sequential(*layers)
        # 合理的初始化:让 h_θ 初始接近 0 → u_θ 初始 ≈ κ u_g(引导解本身)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _act(name: str):
        if name == "silu":
            return nn.SiLU()                      # SiLU(x) = x/(1+e^{-x})
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"未知激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向:h_θ(x)。

        Args:
            x: 场点坐标, shape (N, 3)

        Returns:
            h: shape (N, 1),网络原始输出(未经 tanh)
        """
        return self.net(x)


class GuidedPINN(nn.Module):
    """完整 PINN:输入坐标 → 输出修正项 u_θ(应用 Eq.5 硬约束 ansatz)。

    Args:
        pcfg  : PINNConfig
        kappa : 全局尺度 κ(阶段一确定)
        c     : 修正幅度系数 c(超参数)
        masses: 奇点质量, numpy shape (2,)
        xs    : 奇点位置, numpy shape (2, 3)
        Ps    : 奇点线动量, numpy shape (2, 3)
        Ss    : 奇点自旋, numpy shape (2, 3)
        u_min : 引导解在域内最小值(窗函数归一化)
        u_max : 引导解在域内最大值(窗函数归一化)
    """

    def __init__(self, pcfg: PINNConfig, kappa: float, c: float,
                 masses, xs, Ps, Ss, u_min: float, u_max: float):
        super().__init__()
        self.mlp = SiLUMLP(pcfg)
        self.kappa = kappa
        self.c = c
        # 物理参数存为 buffer(不参与训练,但随模型保存/加载)
        self.register_buffer("masses", torch.tensor(masses, dtype=torch.float64))
        self.register_buffer("xs", torch.tensor(xs, dtype=torch.float64))
        self.register_buffer("Ps", torch.tensor(Ps, dtype=torch.float64))
        self.register_buffer("Ss", torch.tensor(Ss, dtype=torch.float64))
        self.u_min = u_min
        self.u_max = u_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向:u_θ = κ u_g [1 + c W tanh(h_θ)]。

        u_g 与 W 实时从 x 解析计算(可微,贡献 Δu 的解析部分)。

        Args:
            x: 场点坐标, shape (N, 3)(需要梯度:残差要求二阶导数)

        Returns:
            u_theta: shape (N,),满足结构约束的修正项解
        """
        ug = physics.guide_u(x, self.masses, self.xs, self.Ps, self.Ss)  # (N,) 可微引导解
        ug = ug.to(dtype=x.dtype)
        w = (ug - self.u_min) / (self.u_max - self.u_min + 1e-8)         # (N,) 窗函数
        h = self.mlp(x)                                                  # (N,1) h_θ
        h = torch.tanh(h.squeeze(-1))                                    # (N,) 有界修正比例
        return self.kappa * ug * (1.0 + self.c * w * h)                  # (N,) Eq.(5)

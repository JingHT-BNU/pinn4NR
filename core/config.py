"""
config.py —— 配置模块:集中管理物理参数与训练超参数
=====================================================

论文:Solving Hamiltonian Constraint Equation with Physics-Informed Neural Networks
    (arXiv:2607.06002v1, 2026)

本文件用 dataclass 定义三类配置:
    1. BBHConfig    : 双黑洞物理参数(质量、位置、动量、自旋)
    2. PINNConfig   : 网络结构参数
    3. TrainConfig  : 训练超参数(与论文 §II 对齐)

全部参数都有默认值 = 论文 §II.B.4 的"基准算例"
(等质量 m±=0.5, x=±3, P_+y=-P_-y=0.2, 无自旋)。
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class BBHConfig:
    """双黑洞物理参数(论文 Eq.2 中的 m_n, x_n, P_n, S_n)。

    约定:
        - 下标 + 对应 x 轴正方向的奇点,下标 - 对应负方向;
        - 所有量均为几何单位制(G=c=1),无量纲。
    """
    m_plus: float = 0.5            # 奇点 + 的质量参数
    m_minus: float = 0.5           # 奇点 - 的质量参数
    x_plus: Tuple[float, float, float] = (3.0, 0.0, 0.0)     # 奇点 + 的位置
    x_minus: Tuple[float, float, float] = (-3.0, 0.0, 0.0)   # 奇点 - 的位置
    P_plus: Tuple[float, float, float] = (0.0, 0.2, 0.0)     # 奇点 + 的线动量
    P_minus: Tuple[float, float, float] = (0.0, -0.2, 0.0)   # 奇点 - 的线动量
    S_plus: Tuple[float, float, float] = (0.0, 0.0, 0.0)     # 奇点 + 的自旋
    S_minus: Tuple[float, float, float] = (0.0, 0.0, 0.0)    # 奇点 - 的自旋

    def as_arrays(self):
        """把 dataclass 转成 numpy 数组,便于向量化计算。

        Returns:
            masses: ndarray (2,), 两个奇点的质量
            xs    : ndarray (2,3), 两个奇点的位置
            Ps    : ndarray (2,3), 两个奇点的线动量
            Ss    : ndarray (2,3), 两个奇点的自旋
        """
        masses = np.array([self.m_plus, self.m_minus])
        xs = np.array([self.x_plus, self.x_minus], dtype=float)
        Ps = np.array([self.P_plus, self.P_minus], dtype=float)
        Ss = np.array([self.S_plus, self.S_minus], dtype=float)
        return masses, xs, Ps, Ss

    @staticmethod
    def from_case(name: str) -> "BBHConfig":
        """按论文 Table II 的三个典型算例构造配置。

        Args:
            name: 取值
                - "base"         : 等质量无自旋(论文 §II.B.4 基准算例)
                - "spin_eq"      : 等质量自旋(Table II 第一行)
                - "uneq_nospin"  : 不等质量无自旋(Table II 第二行)
                - "uneq_spin"    : 不等质量自旋(Table II 第三行)

        Returns:
            BBHConfig: 对应算例的物理参数
        """
        if name == "base":
            return BBHConfig()
        if name == "spin_eq":
            return BBHConfig(m_plus=0.5, m_minus=0.5,
                             x_plus=(3.0, 0, 0), x_minus=(-3.0, 0, 0),
                             P_plus=(0, 0.2, 0), P_minus=(0, -0.2, 0),
                             S_plus=(0, 0, 0.2), S_minus=(0, 0, 0.2))
        if name == "uneq_nospin":
            return BBHConfig(m_plus=3.0, m_minus=2.0,
                             x_plus=(8.0, 0, 0), x_minus=(-8.0, 0, 0),
                             P_plus=(0.1, 0.25, 0), P_minus=(0.2, -0.25, 0),
                             S_plus=(0, 0, 0), S_minus=(0, 0, 0))
        if name == "uneq_spin":
            return BBHConfig(m_plus=3.0, m_minus=1.0,
                             x_plus=(5.0, 0, 0), x_minus=(-5.0, 0, 0),
                             P_plus=(0, 0.3, 0), P_minus=(0, -0.2, 0),
                             S_plus=(0, 0, 0.1), S_minus=(0, 0, 0.2))
        raise ValueError(f"未知算例名: {name}")


@dataclass
class PINNConfig:
    """网络结构(论文 §II.B.4:3 层隐藏层,每层 64 神经元,SiLU 激活)。"""
    hidden_layers: int = 3                 # 隐藏层层数
    hidden_neurons: int = 64               # 每层神经元数
    activation: str = "silu"               # 激活函数:silu(x) = x/(1+e^{-x})


@dataclass
class TrainConfig:
    """训练超参数(论文 §II 与 Table I 的基准设置)。"""
    # ---- 计算域 ----
    R_max: float = 30.0                    # 球域半径 Rmax(论文 §II.B.1)
    N_Omega: int = 20000                   # 内部配置点数 NΩ(论文 §II.B.2)
    N_boundary: int = 8000                 # 边界配置点数 N∂Ω

    # ---- 采样与加权(默认=论文的均匀配置,A1 复现验证) ----
    # 注意:A1 复现实验(E1 vs E2 vs base_full)证明,自适应采样与峰值加权把训练点
    # 集中到奇点邻域,系统性欠加权 bulk 区,导致整体 L2RE 从 ~1e-2 恶化到 ~1e-1。
    # 论文原配置为均匀采样 + 均匀 L2 损失,故默认关闭二者(可手动重新开启)。
    adaptive_sampling: bool = False        # 是否启用自适应采样
    adaptive_inner_radius: float = 2.0     # 每个奇点周围内层球半径
    adaptive_inner_fraction: float = 0.5   # 分配给内层区域的点数比例(0=全部均匀)

    # ---- 峰值区域加权(提高奇点附近的损失权重) ----
    peak_weighting: bool = False           # 是否对奇点附近残差加权
    peak_weight_power: float = 1.0         # 权重 ∝ 1/r^power(0=关闭,建议 0.5-1.0)

    # ---- 引导解 ansatz(Eq.5) ----
    c_guide: float = 0.2                   # 修正幅度系数 c(基准算例)

    # ---- 损失权重(Eq.11-15) ----
    w2: float = 1.0                        # L2 权重
    w_inf: float = 0.0                     # soft-L∞ 权重(基准算例为 0)
    w_rob: float = 1.0                     # Robin 边界损失权重
    beta: float = 2.5                      # soft-L∞ 的尖锐度 β

    # ---- EMA 损失平衡(Eq.16-17) ----
    ema_alpha: float = 0.9                 # EMA 衰减系数 α ∈ [0,1)

    # ---- 优化 ----
    lr: float = 3e-4                       # Adam 学习率(基准算例)
    n_steps: int = 5000                    # 训练步数
    seed: int = 42                         # 随机种子(采样与初始化)

    # ---- κ 求解的 QMC 积分点数(论文 §II.B.1) ----
    n_qmc_vol: int = 5_000_000             # 体积积分点数(论文取值 5e6)
    n_qmc_surf: int = 50_000               # 边界积分点数(论文取值 5e4)

    # ---- 冒烟测试(本机 CPU 快速验证用) ----
    @staticmethod
    def smoke() -> "TrainConfig":
        """小规模配置:本机 CPU 冒烟验证用,几秒~几分钟跑完。"""
        return TrainConfig(
            N_Omega=1500, N_boundary=600,
            n_qmc_vol=20_000, n_qmc_surf=2_000,
            n_steps=200, seed=42,
        )


# 论文 Table II 三个算例对应的训练超参数(§IV)
CASE_HYPER: dict = {
    "base":       dict(c_guide=0.2, w2=1.0, w_inf=0.0, w_rob=1.0, beta=2.5,  lr=3e-4, n_steps=5000),
    "spin_eq":    dict(c_guide=1.0, w2=1.0, w_inf=0.5, w_rob=1.0, beta=2.5,  lr=5e-4, n_steps=5000),
    "uneq_nospin":dict(c_guide=1.0, w2=1.0, w_inf=0.5, w_rob=1.0, beta=10.0, lr=5e-4, n_steps=5000),
    "uneq_spin":  dict(c_guide=1.0, w2=1.0, w_inf=1.0, w_rob=1.0, beta=10.0, lr=5e-4, n_steps=10000),
}


def build_train_config(case: str = "base", smoke: bool = False) -> TrainConfig:
    """按算例名构造 TrainConfig。

    Args:
        case : 算例名(见 BBHConfig.from_case)
        smoke: True 时用冒烟小规模配置(仅用于本机 CPU 验证)

    Returns:
        TrainConfig
    """
    cfg = TrainConfig.smoke() if smoke else TrainConfig()
    for k, v in CASE_HYPER.get(case, {}).items():
        if smoke and k == "n_steps":
            continue                       # 冒烟模式保持小步数(200),不被算例步数覆盖
        setattr(cfg, k, v)
    return cfg

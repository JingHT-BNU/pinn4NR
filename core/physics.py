"""
physics.py —— 物理模块:哈密顿约束方程所需的全部解析量
=======================================================

对应论文公式:
    - Eq.(1)  : Lichnerowicz 方程:  Δψ + (1/8)ψ^{-7} K̄_ij K̄^ij = 0
    - Eq.(2)  : Bowen-York 解 K̄_ij(动量约束的解析解)
    - Eq.(3)  : 奇点分解 ψ = 1 + Σ_n m_n/(2r_n) + u
    - Eq.(4)  : 修正项方程       Δu + (1/8)ψ^{-7} K̄_ij K̄^ij = 0   ← 本模块求解对象
    - Eq.(5)  : 引导式硬约束 ansatz u_θ = κ u_g [1 + c W(x) tanh(h_θ)]
    - Eq.(6)  : 引导解 u_g = u_P + u_J + u_c(Lousto-Zlochower 2008)
    - Eq.(7)  : 窗函数 W(x) = (u_g - u_min)/(u_max - u_min)
    - Eq.(8)  : Robin 边界条件 n̂·∇u + (1/r)(∂r/∂n) u = 0 (球面上等价 ∂u/∂r + u/r = 0)
    - Eq.(9-10): 由散度定理确定全局尺度 κ 的标量方程

所有物理量都实现为 torch 函数(可与自动微分配合,求二阶导数 Δu)。
坐标为几何单位制(G=c=1),无量纲。
"""

from typing import Tuple
import numpy as np
import torch
from scipy.optimize import brentq

from config import BBHConfig

# 数值保护:避免除零/奇异点处的 NaN(奇点位置 r=0 不在采样与评估点中)
_EPS = 1e-8


# ----------------------------------------------------------------------
# 一、奇点分解中的"已知解析量"
# ----------------------------------------------------------------------

def psi_sing(x: torch.Tensor, masses: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """共形因子的奇异(已知)部分: ψ_sing = 1 + Σ_n m_n/(2 r_n)   [Eq.3 前两项]

    Args:
        x     : 场点坐标, shape (N, 3)
        masses: 两个奇点质量, shape (2,)
        xs    : 两个奇点位置, shape (2, 3)

    Returns:
        psi_sing: shape (N,), ψ 中已知的 1 + Σ m/(2r) 部分(不含待求的 u)
    """
    psi = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    # 统一 dtype 与 device:参数可能与 x(GPU 训练 / CPU 积分)不在同一设备
    masses = masses.to(dtype=x.dtype, device=x.device)
    xs = xs.to(dtype=x.dtype, device=x.device)
    for n in range(2):
        rn = (x - xs[n]).norm(dim=1) + _EPS          # (N,),到第 n 个奇点的距离
        psi = psi + masses[n] / (2.0 * rn)           # 每个奇点贡献 m/(2r)
    return psi


def bowen_york_KK(x: torch.Tensor, masses: torch.Tensor,
                  xs: torch.Tensor, Ps: torch.Tensor, Ss: torch.Tensor) -> torch.Tensor:
    """Bowen-York 共形外曲率的范数平方 K̄_ij K̄^ij   [Eq.2, 平直度规下 = Σ_ij K̄_ij²]

    单个奇点的 Bowen-York 解(相对坐标 rel = x - xs[n]):
        K̄_ij = (3/(2 r³)) [ P_i rel_j + P_j rel_i − (δ_ij − rel_i rel_j/r²)(P·rel) ]
              + (3/r⁵) [ ε_{ikl} S_k rel_l rel_j + ε_{jkl} S_k rel_l rel_i ]
    双奇点:两个解线性叠加(动量约束线性)。

    Args:
        x     : 场点坐标, shape (N, 3)
        masses: 奇点质量, shape (2,)(本函数未用,保留接口一致性)
        xs    : 奇点位置, shape (2, 3)
        Ps    : 奇点线动量, shape (2, 3)
        Ss    : 奇点自旋, shape (2, 3)

    Returns:
        KK: shape (N,), K̄_ij K̄^ij ≥ 0(哈密顿约束源项的"分子"部分)
    """
    N = x.shape[0]
    device = x.device
    Kbar = torch.zeros(N, 3, 3, dtype=x.dtype, device=device)   # (N,3,3) 外曲率张量
    eye = torch.eye(3, dtype=x.dtype, device=device)            # (3,3) 单位阵 δ_ij
    xs = xs.to(dtype=x.dtype, device=device)
    Ps = Ps.to(dtype=x.dtype, device=device)
    Ss = Ss.to(dtype=x.dtype, device=device)

    for n in range(2):
        rel = x - xs[n]                          # (N,3) 相对坐标(场点 - 奇点 n)
        r = rel.norm(dim=1) + _EPS               # (N,) 到奇点 n 的距离
        P = Ps[n]                                # (3,) 该奇点的动量
        S = Ss[n]                                # (3,) 该奇点的自旋

        # ---- 动量项: (3/(2r³)) [P_i rel_j + P_j rel_i − (δ_ij − rel_i rel_j/r²)(P·rel)] ----
        Pdot = (P[None, :] * rel).sum(dim=1)                     # (N,) P·rel
        proj = eye[None, :, :] - (rel[:, :, None] * rel[:, None, :]) / (r[:, None, None] ** 2)  # (N,3,3) 投影算子 δ − rel⊗rel/r²
        asym = P[None, :, None] * rel[:, None, :]                # (N,3,3) P_i rel_j
        momentum = (3.0 / (2.0 * r ** 3))[:, None, None] * (asym + asym.transpose(1, 2) - proj * Pdot[:, None, None])

        # ---- 自旋项: (3/r⁵)[ε_{ikl} S_k rel_l rel_j + ε_{jkl} S_k rel_l rel_i] ----
        Sd = S                                 # (3,) 已统一 dtype
        cross = torch.linalg.cross(Sd[None, :].expand(N, 3), rel)  # (N,3) ε_{ikl} S_k rel_l = (S × rel)_i
        spin = (3.0 / r ** 5)[:, None, None] * (cross[:, :, None] * rel[:, None, :]
                                               + cross[:, None, :] * rel[:, :, None])  # (N,3,3)

        Kbar = Kbar + momentum + spin                            # 线性叠加两个奇点的贡献

    KK = (Kbar ** 2).sum(dim=(1, 2))                             # (N,) K̄_ij K̄^ij(度规平直,上下标等同)
    return KK


# ----------------------------------------------------------------------
# 二、引导解 u_g = u_P + u_J + u_c(Lousto-Zlochower 2008, PRD 77, 024034)
# ----------------------------------------------------------------------

def _lz_boost_part(rel: torch.Tensor, m: float, P: torch.Tensor) -> torch.Tensor:
    """单个 boosted(带动量)奇点的引导解 u_P  [Eq.3-4 of LZ2008]

    变量:
        R  = 2r/m,  ℓ = 1/(1+R),  μ_P = P̂·r̂,  P̃ = 2P/m,  P₂(x)=(3x²-1)/2
        u_P = P̃² (u_P0 + u_P2 P₂(μ_P))
        (32/5) u_P0 = ℓ − 2ℓ² + 2ℓ³ − ℓ⁴ + ℓ⁵/5
        80R u_P2   = 15ℓ + 132ℓ² + 53ℓ³ + 96ℓ⁴ + 82ℓ⁵ + 84ℓ⁵/R + 84 ln(ℓ)/R²

    Args:
        rel: 相对坐标 x − x_n, shape (N,3)
        m  : 该奇点质量(标量)
        P  : 该奇点线动量, shape (3,)

    Returns:
        u_P: shape (N,),动量贡献的解析引导解
    """
    r = rel.norm(dim=1) + _EPS
    R = 2.0 * r / m
    ell = 1.0 / (1.0 + R)
    pnorm = P.norm()
    if pnorm < _EPS:                       # 无动量 → 该项为 0
        return torch.zeros_like(r)
    mu_P = (rel @ P) / (r * pnorm)         # (N,) μ_P = P̂·r̂
    P2 = 0.5 * (3.0 * mu_P ** 2 - 1.0)     # 勒让德多项式 P₂

    uP0 = (5.0 / 32.0) * (ell - 2*ell**2 + 2*ell**3 - ell**4 + ell**5/5.0)
    # 注:84ℓ⁵/R 与 84ln(ℓ)/R² 两项在 R→0 时各自发散但互相抵消(解析上),故整体有限
    uP2 = (15*ell + 132*ell**2 + 53*ell**3 + 96*ell**4 + 82*ell**5
           + 84*ell**5/R + 84*torch.log(ell)/R**2) / (80.0 * R)

    Ptil = 2.0 * pnorm / m                 # P̃ = 2P/m(标量)
    return (Ptil ** 2) * (uP0 + uP2 * P2)  # (N,)


def _lz_spin_part(rel: torch.Tensor, m: float, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """单个自旋奇点的引导解 u_J 与交叉项 u_c  [Eq.1-2,5 of LZ2008]

    变量:
        J̃ = 4J/m²,  μ_J = Ĵ·r̂
        u_J = J̃² (u_J0 + u_J2 R² P₂(μ_J))
        40 u_J0 = ℓ + ℓ² + ℓ³ − 4ℓ⁴ + 2ℓ⁵
        20 u_J2 = −ℓ⁵
    交叉项(动量×自旋): u_c = [(P̃ × J̃)·R⃗] (1+5R+10R²) ℓ⁵ / 80, R⃗ = 2r⃗/m

    Args:
        rel: 相对坐标 x − x_n, shape (N,3)
        m  : 该奇点质量(标量)
        S  : 该奇点自旋(=角动量 J), shape (3,)

    Returns:
        u_J: shape (N,),自旋贡献的引导解
        u_c: shape (N,),动量-自旋交叉项(无动量时恒为 0)
    """
    r = rel.norm(dim=1) + _EPS
    R = 2.0 * r / m
    ell = 1.0 / (1.0 + R)
    snorm = S.norm()
    u_c = torch.zeros_like(r)
    if snorm < _EPS:                       # 无自旋 → 两项都为 0
        return u_c, u_c
    mu_J = (rel @ S) / (r * snorm)         # (N,) μ_J = Ĵ·r̂
    P2 = 0.5 * (3.0 * mu_J ** 2 - 1.0)

    uJ0 = (ell + ell**2 + ell**3 - 4*ell**4 + 2*ell**5) / 40.0
    uJ2 = -ell**5 / 20.0
    Jtil = 4.0 * snorm / m ** 2            # J̃ = 4J/m²
    u_J = (Jtil ** 2) * (uJ0 + uJ2 * R**2 * P2)   # (N,)
    return u_J, u_c                        # u_c 需动量,在引导解主函数中计算


def guide_u(x: torch.Tensor, masses: torch.Tensor, xs: torch.Tensor,
            Ps: torch.Tensor, Ss: torch.Tensor) -> torch.Tensor:
    """解析引导解 u_g = u_P + u_J + u_c   [Eq.6; LZ2008]

    每个奇点独立计算 (u_P, u_J, u_c),再线性叠加(多奇点叠加原理)。

    Args:
        x     : 场点坐标, shape (N, 3)
        masses: 奇点质量, shape (2,)
        xs    : 奇点位置, shape (2, 3)
        Ps    : 奇点线动量, shape (2, 3)
        Ss    : 奇点自旋, shape (2, 3)

    Returns:
        u_g: shape (N,),引导解(解析近似,修正项 u 的"第一猜测")
    """
    u = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    # 统一 dtype 与 device:物理参数可能与 x(GPU 训练 / CPU 积分)不在同一设备
    masses = masses.to(dtype=x.dtype, device=x.device)
    xs = xs.to(dtype=x.dtype, device=x.device)
    Ps = Ps.to(dtype=x.dtype, device=x.device)
    Ss = Ss.to(dtype=x.dtype, device=x.device)
    for n in range(2):
        m = float(masses[n])
        rel = x - xs[n]                    # (N,3) 相对坐标
        P = Ps[n]
        S = Ss[n]

        uP = _lz_boost_part(rel, m, P)     # (N,) 动量项
        uJ, _ = _lz_spin_part(rel, m, S)   # (N,) 自旋项

        # ---- 交叉项 u_c = [(P̃ × J̃)·R⃗] (1+5R+10R²)ℓ⁵/80 ----
        u_c = torch.zeros_like(uP)
        if P.norm() > _EPS and S.norm() > _EPS:
            r = rel.norm(dim=1) + _EPS
            R = 2.0 * r / m
            ell = 1.0 / (1.0 + R)
            Ptil_vec = 2.0 * P / m         # (3,) P̃ = 2P/m(矢量)
            Jtil_vec = 4.0 * S / m ** 2    # (3,) J̃ = 4J/m²(矢量)
            Rvec = 2.0 * rel / m           # (N,3) R⃗ = 2r⃗/m(矢量)
            cross = torch.linalg.cross(Ptil_vec[None, :].expand(x.shape[0], 3),
                                       Jtil_vec[None, :].expand(x.shape[0], 3))  # (N,3) P̃×J̃
            u_c = (cross * Rvec).sum(dim=1) * (1 + 5*R + 10*R**2) * ell**5 / 80.0

        u = u + uP + uJ + u_c
    return u


def window_function(u_g: torch.Tensor, u_min: float, u_max: float) -> torch.Tensor:
    """窗函数 W(x) = (u_g − u_min)/(u_max − u_min)   [Eq.7]

    把引导解线性映射到 [0,1]:u_g 大(奇点附近强场区)的地方 W≈1,网络有完全修正权;
    u_g 小的地方修正权被抑制。W 是固定的,训练前由解析 u_g 算好。

    Args:
        u_g : 引导解, shape (N,)
        u_min: 引导解在计算域内的最小值
        u_max: 引导解在计算域内的最大值

    Returns:
        W: shape (N,),取值在 [0,1]
    """
    return (u_g - u_min) / (u_max - u_min + _EPS)


# ----------------------------------------------------------------------
# 三、κ 的确定:散度定理 + 拟蒙特卡洛积分   [Eq.9-10]
# ----------------------------------------------------------------------

def solve_kappa(cfg, masses, xs, Ps, Ss,
                x_vol: np.ndarray, x_surf: np.ndarray, R_max: float) -> float:
    """求解全局尺度 κ(论文 Eq.10,两阶段训练的阶段一)。

    方程:  −κ ∮_{∂Ω} ∇u_g·n̂ dS = (1/8) ∫_Ω K̄K̄ (ψ_sing + κ u_g)^{-7} dV
    令:
        S_b(κ 无关) = ∮ ∇u_g·n̂ dS = 4πR_max² · mean(∂u_g/∂r)(边界 QMC 采样)
        V(κ) = ∫ K̄K̄ (ψ_sing + κu_g)^{-7} dV = (4/3)πR_max³ · mean(...)(体积 QMC 采样)
    解标量方程 f(κ) = −κ·S_b − (1/8)·V(κ) = 0。

    Args:
        cfg    : TrainConfig
        masses : 奇点质量, shape (2,)
        xs     : 奇点位置, shape (2, 3)
        Ps     : 奇点线动量, shape (2, 3)
        Ss     : 奇点自旋, shape (2, 3)
        x_vol  : 球内 QMC 采样点(Cartesian 坐标), shape (n_qmc_vol, 3)
        x_surf : 球面 QMC 采样点(Cartesian 坐标), shape (n_qmc_surf, 3)
        R_max  : 球域半径

    Returns:
        kappa: 浮点数,全局尺度因子(≈ O(1))
    """
    # ---- 边界积分:∂u_g/∂r 在球面 r=R_max 上(中心差分) ----
    # x_surf 是球面上笛卡尔坐标点,直接取径向差分(缩放因子 (R±ε)/R)
    x_t = torch.from_numpy(x_surf).double()
    m_t = torch.from_numpy(masses).double()
    xs_t = torch.from_numpy(xs).double()
    P_t = torch.from_numpy(Ps).double()
    S_t = torch.from_numpy(Ss).double()
    eps_r = 1e-3 * R_max                   # 中心差分步长(沿径向)
    x_plus = x_t * (1.0 + eps_r / R_max)   # 沿径向向外
    x_minus = x_t * (1.0 - eps_r / R_max)  # 沿径向向内
    ug_p = guide_u(x_plus, m_t, xs_t, P_t, S_t)
    ug_m = guide_u(x_minus, m_t, xs_t, P_t, S_t)
    dudr = (ug_p - ug_m) / (2.0 * eps_r)   # (n_surf,) ∂u_g/∂r
    S_b = 4.0 * np.pi * R_max ** 2 * dudr.mean().item()          # ∮∇u_g·n̂ dS

    # ---- 体积积分:x_vol 是球内笛卡尔坐标点,直接代入 ----
    xv_t = torch.from_numpy(x_vol).double()
    ps_v = psi_sing(xv_t, m_t, xs_t)                              # (n_vol,) ψ_sing
    kk_v = bowen_york_KK(xv_t, m_t, xs_t, P_t, S_t)               # (n_vol,) K̄K̄
    ug_v = guide_u(xv_t, m_t, xs_t, P_t, S_t)                     # (n_vol,) u_g
    vol = (4.0 / 3.0) * np.pi * R_max ** 3                        # 球体积

    def f(kappa: float) -> float:
        """f(κ) = −κ·S_b − (1/8)·V(κ),求 f=0 的根。"""
        psi7 = torch.clamp(ps_v + kappa * ug_v, min=1e-4)         # ψ = ψ_sing + κu_g, 下限保护
        V = (kk_v / (psi7 ** 7)).mean().item() * vol               # ∫K̄K̄ψ^{-7}dV
        return -kappa * S_b - V / 8.0

    # ---- 二分法求根:f(0) = −V(0)/8 < 0; 找 f>0 的右端点 ----
    kappa_max = 1.0
    while f(kappa_max) < 0 and kappa_max < 1e3:
        kappa_max *= 2.0
    try:
        kappa = brentq(f, 0.0, kappa_max, xtol=1e-8)
    except ValueError:
        # 无根(理论不该发生):退回线性估计 −(1/8)V(0)/S_b
        kappa = -(1.0 / 8.0) * f(0.0) / max(abs(S_b), 1e-12) if abs(S_b) > 0 else 1.0
    return float(kappa)


# ----------------------------------------------------------------------
# 四、残差计算(训练损失的核心)
# ----------------------------------------------------------------------

def pde_residual(u: torch.Tensor, x: torch.Tensor,
                 psi_s: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    """方程残差 R = Δu + (1/8)ψ^{-7} K̄K̄   [Eq.4]

    Args:
        u    : 网络输出的修正项 u_θ, shape (N,)(requires_grad=True)
        x    : 场点坐标, shape (N,3)(requires_grad=True)
        psi_s: ψ_sing, shape (N,)(已知部分)
        kk   : K̄_ijK̄^ij, shape (N,)

    Returns:
        R: shape (N,),方程残差(理想解处为 0)
    """
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]      # (N,3) ∇u
    lap = torch.zeros_like(u)
    for i in range(3):                                             # Δu = Σ ∂²u/∂x_i²
        g2 = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0]
        lap = lap + g2[:, i]
    psi = psi_s + u                                                # (N,) ψ = ψ_sing + u
    psi = torch.clamp(psi, min=1e-4)                               # ψ>0 的物理下限保护
    return lap + (1.0 / 8.0) * kk / (psi ** 7)


def robin_residual(u: torch.Tensor, x: torch.Tensor, R_max: float) -> torch.Tensor:
    """Robin 边界残差 R_B = n̂·∇u + (1/r)(∂r/∂n) u   [Eq.8, 球面上 ∂r/∂n=1]

    在球面 r=R_max 上:n̂ = x̂,故 R_B = (x·∇u)/r + u/r。

    Args:
        u    : 网络输出的修正项, shape (N,)(requires_grad=True)
        x    : 球面上的场点坐标, shape (N,3)(requires_grad=True)
        R_max: 球域半径

    Returns:
        R_B: shape (N,),边界残差(理想解处为 0,模拟 u~1/r 的远场衰减)
    """
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]      # (N,3) ∇u
    xg = (x * g).sum(dim=1)                                        # (N,) x·∇u
    return (xg + u) / R_max                                        # (N,) (x·∇u + u)/r

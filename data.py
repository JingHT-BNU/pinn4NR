"""
data.py —— 数据获取模块:采样配置点 + 拟蒙特卡洛积分点 + 参考解加载
==================================================================

PINN 是"无网格"方法:训练数据 = 计算域内的随机采样点(配置点)。
本模块负责:
    1. 球域内部配置点 NΩ(均匀随机)—— 计算 PDE 残差
    2. 球面边界配置点 N∂Ω —— 计算 Robin 边界残差
    3. κ 求解用的 QMC(Sobol)积分点 —— 两阶段训练阶段一
    4. (可选)TwoPunctures 参考解 npz 的加载,用于 L2RE 评估

论文 §II.B.2:训练前均匀随机选取 NΩ=20000 内部点, N∂Ω=8000 边界点。
κ 积分(§II.B.1):体积 5×10⁶ 点,边界 5×10⁴ 点,用 Sobol 序列(低差异)。
"""

from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from scipy.stats.qmc import Sobol

# Sobol 序列对非 2 的幂点数只提示"平衡性质略降",不影响积分收敛性(论文亦用非 2 幂点数)
warnings.filterwarnings("ignore", message="The balance properties of Sobol")

from config import BBHConfig, TrainConfig


def sample_ball(n: int, R_max: float, rng: np.random.Generator) -> np.ndarray:
    """在半径为 R_max 的球内均匀采样 n 个点。

    均匀球内采样的标准做法:方向均匀(球面上) + 半径 r = R_max·u^{1/3}(u~U[0,1])。
    这样体积元 dV = r² dr dΩ 被均匀覆盖(小 r 处点更密,但每体积点数相同)。

    Args:
        n    : 采样点数
        R_max: 球半径
        rng  : numpy 随机数生成器(可复现)

    Returns:
        points: shape (n, 3),球内均匀分布的坐标
    """
    u = rng.random(n)                       # (n,) 半径均匀变量
    r = R_max * np.cbrt(u)                  # (n,) r = R_max·u^{1/3}
    # 高斯向量归一化得到球面均匀方向(比三角函数更快更稳)
    dir_vec = rng.standard_normal((n, 3))
    dir_vec /= np.linalg.norm(dir_vec, axis=1, keepdims=True)     # (n,3) 单位方向
    return dir_vec * r[:, None]


def sample_ball_adaptive(n: int, R_max: float, singularity_positions: np.ndarray,
                         rng: np.random.Generator,
                         inner_radius: float = 2.0,
                         inner_fraction: float = 0.5) -> np.ndarray:
    """自适应球内采样:在奇点附近(峰值区域)加密采样点。

    问题:均匀球内采样在奇点附近每单位体积点数不变,但解在 r~0.3-2 范围
    变化剧烈(~1/r 发散),导致 PINN 在峰值区域缺乏足够的监督信号,
    评估时出现锯齿状误差。

    策略:将球域分成两个区域:
        - 内层(r <= inner_radius 围绕每个奇点):分配 inner_fraction 的点,
          使用更密集的径向采样(对数间隔 r = inner_radius·u^α, α<1/3)
        - 外层(剩余球域):分配 (1-inner_fraction) 的点,均匀体积采样

    Args:
        n                    : 总采样点数
        R_max                : 球域半径
        singularity_positions: 奇点位置, shape (k, 3)
        rng                  : 随机数生成器
        inner_radius         : 每个奇点周围内层球半径(默认 2.0)
        inner_fraction       : 分配给内层区域的点数比例(默认 0.5)

    Returns:
        points: shape (n, 3),自适应分布的采样点
    """
    n_inner = int(n * inner_fraction)
    n_outer = n - n_inner
    k = singularity_positions.shape[0]  # 奇点数量(通常 2)

    points = []

    # ---- 内层:每个奇点周围 r <= inner_radius 的球内 ----
    if n_inner > 0 and k > 0:
        n_per_sing = n_inner // k
        n_remainder = n_inner % k
        for idx in range(k):
            n_i = n_per_sing + (1 if idx < n_remainder else 0)
            if n_i == 0:
                continue
            # 对数半径分布:让小 r 处有更密的点
            # r = inner_radius * u^(1/5), u~U[0,1]
            # 指数 1/5 < 1/3 使得小 r 处比均匀体积分布更密
            u_r = rng.random(n_i)
            r = inner_radius * np.power(u_r, 1.0 / 5.0)
            # 方向均匀(球面)
            dir_vec = rng.standard_normal((n_i, 3))
            dir_vec /= np.linalg.norm(dir_vec, axis=1, keepdims=True)
            pts_i = dir_vec * r[:, None] + singularity_positions[idx]
            points.append(pts_i)

    # ---- 外层:球域内但不属于任何内层球的点 ----
    if n_outer > 0:
        outer_pts = []
        # 用拒绝采样:先生成均匀球内点,再排除落入内层球的点
        # 为效率,一次性多生成一些点再筛选
        batch_factor = 2  # 超额生成因子(外层体积通常远大于内层)
        while len(outer_pts) < n_outer:
            n_gen = (n_outer - len(outer_pts)) * batch_factor
            n_gen = max(n_gen, 1000)
            cand = sample_ball(n_gen, R_max, rng)
            # 排除落入任何奇点内层球的点
            mask = np.ones(n_gen, dtype=bool)
            for idx in range(k):
                dist = np.linalg.norm(cand - singularity_positions[idx], axis=1)
                mask &= (dist > inner_radius)
            outer_pts.append(cand[mask])
        outer_pts = np.concatenate(outer_pts, axis=0)[:n_outer]
        points.append(outer_pts)

    result = np.concatenate(points, axis=0)
    # 确保最终点数正确(可能因浮点误差差 1-2 个)
    if result.shape[0] > n:
        result = result[:n]
    elif result.shape[0] < n:
        extra = sample_ball(n - result.shape[0], R_max, rng)
        result = np.concatenate([result, extra], axis=0)
    return result.astype(np.float32)


def sample_sphere_surface(n: int, R_max: float, rng: np.random.Generator) -> np.ndarray:
    """在半径为 R_max 的球面(边界 ∂Ω)上均匀采样 n 个点。

    Args:
        n    : 采样点数
        R_max: 球半径
        rng  : numpy 随机数生成器

    Returns:
        points: shape (n, 3),球面上的坐标(满足 ||x|| = R_max)
    """
    dir_vec = rng.standard_normal((n, 3))
    dir_vec /= np.linalg.norm(dir_vec, axis=1, keepdims=True)
    return dir_vec * R_max


def sobol_volume(n: int, R_max: float, seed: Optional[int] = None) -> np.ndarray:
    """Sobol 低差异序列生成的球内体积积分点(κ 求解用)。

    Args:
        n    : 点数
        R_max: 球半径
        seed : Sobol 扰动种子。None=每次随机(旧行为);给定值=可复现。

    Returns:
        points: shape (n, 3),球内点(QMC,比纯随机更均匀)
    """
    sampler = Sobol(d=3, scramble=True, seed=seed)
    u = sampler.random(n)                   # (n,3) [0,1)³ 低差异点
    r = R_max * np.cbrt(u[:, 0])            # 半径:均匀球内分布
    # 面积保持变换:cos(θ) 均匀分布于 [-1,1],确保方向在球面上均匀
    th = np.arccos(1.0 - 2.0 * u[:, 1])
    ph = u[:, 2] * 2.0 * np.pi
    dir_vec = np.stack([np.sin(th) * np.cos(ph),
                        np.sin(th) * np.sin(ph),
                        np.cos(th)], axis=1)
    return dir_vec * r[:, None]


def sobol_sphere_surface(n: int, R_max: float, seed: Optional[int] = None) -> np.ndarray:
    """Sobol 序列生成的球面边界积分点(κ 求解用)。

    Args:
        n    : 点数
        R_max: 球半径
        seed : Sobol 扰动种子。None=每次随机(旧行为);给定值=可复现。

    Returns:
        points: shape (n, 3),球面点
    """
    sampler = Sobol(d=2, scramble=True, seed=seed)
    u = sampler.random(n)                   # (n,2) [0,1)²
    # 面积保持变换:cos(θ) 均匀分布于 [-1,1],确保方向在球面上均匀
    th = np.arccos(1.0 - 2.0 * u[:, 0])
    ph = u[:, 1] * 2.0 * np.pi
    dir_vec = np.stack([np.sin(th) * np.cos(ph),
                        np.sin(th) * np.sin(ph),
                        np.cos(th)], axis=1)
    return dir_vec * R_max


class DataBundle:
    """一次采样得到的所有训练数据(论文 §II.B.2 的"训练前采样一次")。

    属性:
        x_int  : 内部配置点, shape (NΩ, 3), float32
        x_bnd  : 边界配置点, shape (N∂Ω, 3), float32
        ps_int : 内部点 ψ_sing, shape (NΩ,)
        kk_int : 内部点 K̄K̄,   shape (NΩ,)
        ug_int : 内部点引导解, shape (NΩ,)
        w_int  : 内部点窗函数, shape (NΩ,)
        ps_bnd : 边界点 ψ_sing, shape (N∂Ω,)
        kk_bnd : 边界点 K̄K̄,   shape (N∂Ω,)
        ug_bnd : 边界点引导解, shape (N∂Ω,)
        w_bnd  : 边界点窗函数, shape (N∂Ω,)
        kappa  : 全局尺度因子 κ(两阶段训练阶段一的产物)
        u_min, u_max: 引导解在采样点上的最小/最大值(窗函数归一化用)
    """

    def __init__(self, cfg: TrainConfig, bb: BBHConfig, seed: Optional[int] = None):
        seed = seed if seed is not None else cfg.seed
        rng = np.random.default_rng(seed)
        self.R_max = cfg.R_max

        # 先提取物理参数(采样时需奇点位置)
        masses, xs, Ps, Ss = bb.as_arrays()

        # ---- 1. 训练配置点(论文:NΩ=20000 内部 + N∂Ω=8000 边界) ----
        if cfg.adaptive_sampling:
            self.x_int = sample_ball_adaptive(
                cfg.N_Omega, cfg.R_max, xs,
                rng,
                inner_radius=cfg.adaptive_inner_radius,
                inner_fraction=cfg.adaptive_inner_fraction,
            ).astype(np.float32)
        else:
            self.x_int = sample_ball(cfg.N_Omega, cfg.R_max, rng).astype(np.float32)
        self.x_bnd = sample_sphere_surface(cfg.N_boundary, cfg.R_max, rng).astype(np.float32)

        # ---- 2. 解析量(用 torch 计算后转回 numpy 缓存) ----
        import physics
        m_t = torch.from_numpy(masses).double()
        xs_t = torch.from_numpy(xs).double()
        P_t = torch.from_numpy(Ps).double()
        S_t = torch.from_numpy(Ss).double()

        def _compute(x_np: np.ndarray):
            x_t = torch.from_numpy(x_np).double()
            ps = physics.psi_sing(x_t, m_t, xs_t).numpy()
            kk = physics.bowen_york_KK(x_t, m_t, xs_t, P_t, S_t).numpy()
            ug = physics.guide_u(x_t, m_t, xs_t, P_t, S_t).numpy()
            return (ps.astype(np.float32), kk.astype(np.float32), ug.astype(np.float32))

        self.ps_int, self.kk_int, self.ug_int = _compute(self.x_int)
        self.ps_bnd, self.kk_bnd, self.ug_bnd = _compute(self.x_bnd)

        # ---- 3. 窗函数归一化常数(u_min/u_max)与 κ ----
        all_ug = np.concatenate([self.ug_int, self.ug_bnd])
        self.u_min = float(all_ug.min())
        self.u_max = float(all_ug.max())
        self.w_int = (self.ug_int - self.u_min) / (self.u_max - self.u_min + 1e-8)
        self.w_bnd = (self.ug_bnd - self.u_min) / (self.u_max - self.u_min + 1e-8)

        # ---- 4. κ(阶段一:散度定理 + QMC 积分) ----
        # sobol_volume/sobol_sphere_surface 返回笛卡尔坐标点(非原始 Sobol 序列)
        x_qmc_vol = sobol_volume(cfg.n_qmc_vol, cfg.R_max)
        x_qmc_surf = sobol_sphere_surface(cfg.n_qmc_surf, cfg.R_max)
        self.kappa = physics.solve_kappa(cfg, masses, xs, Ps, Ss,
                                         x_qmc_vol, x_qmc_surf, cfg.R_max)

        # 保存物理参数(评估/可视化时要用)
        self.masses = masses.astype(np.float32)
        self.xs = xs.astype(np.float32)
        self.Ps = Ps.astype(np.float32)
        self.Ss = Ss.astype(np.float32)

    def to_torch(self, device):
        """把 numpy 缓存转成 torch 张量并搬到指定设备。

        Args:
            device: torch 设备字符串(如 'cuda', 'cpu')

        Returns:
            dict: 训练所需的全部张量
                - x_int, x_bnd: (NΩ,3)/(N∂Ω,3) 配置点
                - ps_int, kk_int, ug_int, w_int: (NΩ,) 内部点解析量
                - ps_bnd, kk_bnd, ug_bnd, w_bnd: (N∂Ω,) 边界点解析量
                - kappa: float 全局尺度
        """
        t = torch
        def _t(a): return t.tensor(a, dtype=t.float32, device=device)
        return {
            "x_int": _t(self.x_int),
            "x_bnd": _t(self.x_bnd),
            "ps_int": _t(self.ps_int), "kk_int": _t(self.kk_int),
            "ug_int": _t(self.ug_int), "w_int": _t(self.w_int),
            "ps_bnd": _t(self.ps_bnd), "kk_bnd": _t(self.kk_bnd),
            "ug_bnd": _t(self.ug_bnd), "w_bnd": _t(self.w_bnd),
            "kappa": float(self.kappa),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_npz(cls, path: str) -> "DataBundle":
        """从 data.npz 缓存恢复 DataBundle(跳过采样与 κ 求解)。

        与 save_runs 保存的 data.npz 完全对应。用于"已有模型 → 跳过训练
        直接评估"的流程:数据是训练时的快照,保证评估与训练用同一采样点。

        Args:
            path: data.npz 路径(main.py 保存的缓存)

        Returns:
            DataBundle: 与训练时一致的采样与解析量
        """
        d = np.load(path)
        obj = cls.__new__(cls)                     # 不走 __init__(跳过采样)
        obj.R_max = float(d["R_max"])
        obj.x_int = d["x_int"]
        obj.x_bnd = d["x_bnd"]
        obj.ps_int = d["ps_int"]
        obj.kk_int = d["kk_int"]
        obj.ug_int = d["ug_int"]
        obj.w_int = d["w_int"]
        obj.ps_bnd = d["ps_bnd"]
        obj.kk_bnd = d["kk_bnd"]
        obj.ug_bnd = d["ug_bnd"]
        obj.w_bnd = d["w_bnd"]
        obj.u_min = float(d["u_min"])
        obj.u_max = float(d["u_max"])
        obj.kappa = float(d["kappa"])
        obj.masses = d["masses"]
        obj.xs = d["xs"]
        obj.Ps = d["Ps"]
        obj.Ss = d["Ss"]
        return obj


def load_reference(path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """加载 TwoPunctures 参考解(npz 格式: x_ref (M,3), u_ref (M,))。

    参考解用于计算论文 Eq.18 的 L2RE。如果文件不存在,返回 None,
    评估脚本将退化为"残差自检"(无外部参考)。

    Args:
        path: npz 文件路径,含键 'x_ref' 与 'u_ref'

    Returns:
        (x_ref, u_ref) 或 None
    """
    if path is None:
        return None
    try:
        d = np.load(path)
        return d["x_ref"], d["u_ref"]
    except Exception:
        return None

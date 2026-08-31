"""
tools/spectral_reference.py —— 任意 BBH 参数的谱方法参考解求解器
================================================================

输入 8 维物理参数 (m+, m-, x+, x-, P+y, P-y, S+z, S-z)，输出与
reference_u.npz 同格式的参考解文件 (x_ref, u_ref)，供 eval/viz 直接使用。

方法（球谐谱方法，Ansorg 单域谱的球谐替代实现）:
    未知量 u(共形因子修正项) 满足哈密顿约束
        Δu + (1/8)(ψ_s + u)^{-7} K̄K̄ = 0,   u(∞) = 0,  ψ_s = 1 + Σ m/(2r)
    - 以两奇点中点为心的球坐标; 径向紧化 r = R0(1+s)/(1-s), s∈[-1,1],
      Chebyshev 配点(s=1 对应 r=∞, s=-1 对应 r=0);
    - 角向 cosθ Gauss-Legendre × φ 均匀网格, 球谐展开 l ≤ L
      (Y_lm 自实现, 含 Condon-Shortley 相位, 与 scipy 一致);
    - 每个 l 的径向算子是稠密 (N_r+1)² 矩阵, 原点正则性(u_lm~r^l)与
      无穷远衰减(u_lm→0)作边界条件, LU 分解一次反复使用;
    - 非线性用 "源项延拓 + 自适应阻尼 Picard 迭代":
      轻质量+高自旋奇点附近源项 ~10^4, 直接迭代会在饱和与非饱和间振荡,
      源项系数 λ 从小到大 ramp 保证收敛。
    - 精度由 (N_r, L, N_θ, N_φ) 控制, 默认分辨率对轻自旋奇点足够;
      --selftest 验证算子精度; base 配置可用 --compare 与 reference_u.npz 对比。

用法:
    python paper/tools/spectral_reference.py --params "0.5,0.5,3,-3,0.2,-0.2,0,0" \
        --out paper/tools/refs/ref_base.npz
    python paper/tools/spectral_reference.py --params "..." \
        --compare paper/tools/reference_u.npz     # base 验证
    python paper/tools/spectral_reference.py --selftest

输出 npz 键:
    x_ref (M,3) float32, u_ref (M,) float32 —— 与 make_reference.py 一致
    raw (8,) 求解参数, xc (3,) 球坐标中心, meta (str) JSON 元信息
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.special import gammaln

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "A3_multi_param"))
from logutil import setup_logging  # noqa: E402

log = logging.getLogger("paper.tools.spectral_reference")

import physics  # noqa: E402
try:
    from multi_param_model import build_bbh_from_params  # noqa: E402
except ImportError:  # 仓库化分支无 A3 代码时内联等价实现(约定一致)
    def build_bbh_from_params(raw):
        m_p, m_m, x_p, x_m, P_py, P_my, S_pz, S_mz = raw
        return (np.array([m_p, m_m]),
                np.array([[x_p, 0.0, 0.0], [x_m, 0.0, 0.0]]),
                np.array([[0.0, P_py, 0.0], [0.0, P_my, 0.0]]),
                np.array([[0.0, 0.0, S_pz], [0.0, 0.0, S_mz]]))


# ── 球谐(Y_lm 自实现, 含 Condon-Shortley 相位) ────────────────

def sph_harm_torch(lmax: int, mu: torch.Tensor, phi: torch.Tensor,
                   device, l_list, m_list, dtype=torch.float64) -> torch.Tensor:
    """在 (mu=cosθ, phi) 点集上求 Y_lm。返回 (P, K) 复张量(实部 dtype 同 dtype)。

    递推(m≥0, 含 Condon-Shortley 相位):
        P_m^m = (-1)^m (2m-1)!! (1-μ²)^{m/2}
        P_{m+1}^m = μ(2m+1) P_m^m
        P_l^m = ((2l-1)μ P_{l-1}^m - (l-1+m) P_{l-2}^m) / (l-m)
    归一: N_lm = sqrt((2l+1)/(4π) (l-m)!/(l+m)!)
    负 m 由共轭关系补全: Y_{l,-m} = (-1)^m conj(Y_{l,m})
    """
    P = mu.shape[0]
    K = len(l_list)
    # 递推全程 float64: (2m-1)!! 项在 float32 下 m≥20 即溢出(63!!≈1e51)→NaN;
    # 最终 Y 值 O(1), 在出口处按 dtype 降精度是安全的
    mu = mu.to(device=device, dtype=torch.float64)
    phi = phi.to(device=device, dtype=torch.float64)
    one_m_mu2 = torch.clamp(1.0 - mu * mu, min=0.0)
    sq = torch.sqrt(one_m_mu2)
    Plm = torch.zeros(P, lmax + 1, lmax + 1, dtype=torch.float64, device=device)
    Plm[:, 0, 0] = 1.0
    if lmax >= 1:
        Plm[:, 1, 0] = mu                       # m=0 的首项(m 循环从 1 开始, 不覆盖它)
    dfact = 1.0
    for m in range(1, lmax + 1):
        dfact *= (2 * m - 1)                       # (2m-1)!!
        Plm[:, m, m] = ((-1.0) ** m) * dfact * sq ** m
        if m + 1 <= lmax:
            Plm[:, m + 1, m] = mu * (2 * m + 1.0) * Plm[:, m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            Plm[:, l, m] = ((2 * l - 1.0) * mu * Plm[:, l - 1, m]
                            - (l - 1.0 + m) * Plm[:, l - 2, m]) / (l - m)
    l_a = torch.tensor(l_list, dtype=torch.float64, device=device)
    m_a = torch.tensor(m_list, dtype=torch.float64, device=device)
    logn = (0.5 * (torch.log(2.0 * l_a + 1.0) - torch.log(torch.tensor(4 * np.pi))
                   + torch.lgamma(l_a - m_a + 1.0) - torch.lgamma(l_a + m_a + 1.0)))
    norm = torch.exp(logn)                          # (K,)
    eimp = torch.exp(1j * torch.outer(phi, m_a))
    idx = (l_a.long() * (lmax + 1) + m_a.long())
    Y = Plm.reshape(P, -1)[:, idx] * norm[None, :] * eimp
    if dtype == torch.float32:
        Y = Y.to(torch.complex64)
    # 负 m: Y_{l,-m} = (-1)^m conj(Y_{l,m})
    col = {(l, m): k for k, (l, m) in enumerate(zip(l_list, m_list))}
    for k, (l, m) in enumerate(zip(l_list, m_list)):
        if m < 0:
            Y[:, k] = ((-1.0) ** m) * Y[:, col[(l, -m)]].conj()
    return Y


# ── 径向紧化 Chebyshev ────────────────────────────────────────

def cheb_nodes_D(N: int):
    """Chebyshev-Gauss-Lobatto 节点 s_j = cos(πj/N) 与一阶导矩阵 (Trefethen)。"""
    j = np.arange(N + 1)
    s = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** j
    ds = s[:, None] - s[None, :]                # ds[i,j] = s_i - s_j
    np.fill_diagonal(ds, 1.0)
    D = np.outer(c, 1.0 / c) / ds
    D -= np.diag(np.sum(D, axis=1))
    return s, D


def build_radial(N_r: int, R0: float):
    """返回 s, r(含 r[0]=∞), 一阶/二阶径向导算子 d/dr。

    节点 0 处 r=∞: D1/D2 该行数学上为 0(算子只以 BC 行使用它), 显式清零。"""
    s, D = cheb_nodes_D(N_r)
    t = 1.0 - s
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = 2.0 * R0 / t ** 2                    # dr/ds
        rpp = 4.0 * R0 / t ** 3
        r = R0 * (1.0 + s) / t                    # r[0]=inf (s=1)
        D1 = D / rp[:, None]
        D2 = (D @ D) / rp[:, None] ** 2 - (rpp / rp ** 3)[:, None] * D
    D1[0, :] = 0.0
    D2[0, :] = 0.0
    return s, r, D1, D2


def radial_matrix(l: int, r: np.ndarray, D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """l 模态径向算子 u'' + 2u'/r - l(l+1)/r² u, 含边界条件行。

    行 0 (r=∞): u=0;  行 N (r=0): l=0 → du/dr(0)=0 (正则性), l≥1 → u(0)=0。
    """
    ri = r.copy()
    ri[0] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        A = D2 + np.diag(2.0 / ri) @ D1 - np.diag(l * (l + 1.0) / ri ** 2)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    if l == 0:
        A[-1, :] = D1[-1, :]
    else:
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
    return A


# ── 求解器主体 ────────────────────────────────────────────────

class SpectralPunctureSolver:
    def __init__(self, raw: np.ndarray, N_r=512, L=48, N_th=72, N_ph=144,
                 R0=15.0, device="cuda"):
        self.raw = np.asarray(raw, dtype=np.float64)
        self.masses, self.xs, self.Ps, self.Ss = build_bbh_from_params(self.raw)
        self.xc = np.array([0.5 * (self.xs[0, 0] + self.xs[1, 0]), 0.0, 0.0])
        self.N_r, self.L, self.N_th, self.N_ph, self.R0 = N_r, L, N_th, N_ph, R0
        self.device = torch.device(device if torch.cuda.is_available()
                                   or device == "cpu" else "cpu")

        # 径向 (复数分解: 右端 S_lm 为复数)
        self.s, self.r, self.D1, self.D2 = build_radial(N_r, R0)
        self.lus = [lu_factor(radial_matrix(l, self.r, self.D1, self.D2)
                              .astype(np.complex128)) for l in range(L + 1)]

        # 角向: cosθ Gauss-Legendre × φ 均匀
        mu, w = np.polynomial.legendre.leggauss(N_th)
        self.theta = np.arccos(mu)
        self.phi = 2.0 * np.pi * np.arange(N_ph) / N_ph
        TH, PH = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.l_list, self.m_list = [], []
        for l in range(L + 1):
            for m in range(-l, l + 1):
                self.l_list.append(l)
                self.m_list.append(m)
        self.K = len(self.l_list)
        # 注意 sph_harm_torch 接收 μ=cosθ, 不是极角本身
        mu_t = torch.tensor(np.cos(TH).reshape(-1), dtype=torch.float64)
        ph_t = torch.tensor(PH.reshape(-1), dtype=torch.float64)
        Y = sph_harm_torch(L, mu_t, ph_t, self.device, self.l_list, self.m_list)
        self.Y = Y.reshape(N_th * N_ph, self.K)            # (n_ang, K)
        self.Yc = self.Y.conj()
        # 角向求积权重: 每个 θ 的 GL 权重 × Δφ, 沿展平后的 (θ,φ) 索引重复
        qw = np.repeat(w * (2.0 * np.pi / N_ph), N_ph).astype(np.float64)
        self.qw = torch.tensor(qw, dtype=torch.float64, device=self.device)

        # 源项静态部分 (ψ_s 与 K̄K̄ 不依赖 u, 预计算一次)
        n_ang = N_th * N_ph
        rr = self.r[1:]                                     # 内部径向节点 j=1..N_r (j=0 是 r=∞)
        xyz = np.empty((N_r, N_th, N_ph, 3))
        st, ct = np.sin(TH), np.cos(TH)
        xyz[..., 0] = rr[:, None, None] * (st * np.cos(PH))[None, :, :]
        xyz[..., 1] = rr[:, None, None] * (st * np.sin(PH))[None, :, :]
        xyz[..., 2] = rr[:, None, None] * ct[None, :, :]
        pts = xyz.reshape(N_r, n_ang, 3) + self.xc[None, None, :]   # 绝对坐标
        self.grid_pts = pts
        with torch.no_grad():
            xt = torch.tensor(pts.reshape(-1, 3), dtype=torch.float64, device=self.device)
            m_t = torch.tensor(self.masses, dtype=torch.float64, device=self.device)
            xs_t = torch.tensor(self.xs, dtype=torch.float64, device=self.device)
            P_t = torch.tensor(self.Ps, dtype=torch.float64, device=self.device)
            S_t = torch.tensor(self.Ss, dtype=torch.float64, device=self.device)
            self.psi_s = physics.psi_sing(xt, m_t, xs_t).reshape(N_r, n_ang)
            self.kk = physics.bowen_york_KK(xt, m_t, xs_t, P_t, S_t).reshape(N_r, n_ang)
        self.u_grid = torch.zeros(N_r, n_ang, dtype=torch.float64, device=self.device)

    # ---- 非线性源与一次 Picard 步 ----
    def _source(self, lam: float) -> torch.Tensor:
        psi = (self.psi_s + self.u_grid).clamp(min=0.05)
        return -(1.0 / 8.0) * lam * self.kk / psi ** 7

    def _synthesize(self, U: torch.Tensor) -> torch.Tensor:
        """(K, n_rad) 复模态 → (n_rad, n_ang) 实网格值。"""
        return torch.einsum("kj,ak->aj", U, self.Y).real.T

    def _solve_linear(self, S: torch.Tensor) -> torch.Tensor:
        """Δu = S 的谱解: 角谱分析 → 逐 l 径向回代 → 合成。返回内部网格 u。"""
        S_lm = torch.einsum("ja,ak->jk", (S * self.qw).to(torch.complex128), self.Yc)  # (N_r, K)
        rhs = S_lm.T.cpu().numpy()                                     # (K, N_r)
        full = np.zeros((self.K, self.N_r + 1), dtype=complex)
        full[:, 1:-1] = rhs[:, :-1]     # 方程行 j=1..N_r-1; 行0/行N 保持齐次 BC
        U = np.empty_like(full)
        l_arr = np.array(self.l_list)
        for l in range(self.L + 1):
            sel = np.where(l_arr == l)[0]
            x = lu_solve(self.lus[l], full[sel].T, check_finite=False)
            # 迭代精化×2: 压制高 N_r 下 κ~N⁴ 的 LU 舍入放大
            for _ in range(2):
                res = full[sel].T - radial_matrix(l, self.r, self.D1, self.D2) @ x
                x = x + lu_solve(self.lus[l], res, check_finite=False)
            U[sel] = x.T
        U_t = torch.from_numpy(U).to(self.device)
        u_new = self._synthesize(U_t[:, 1:])
        self._U_full = U_t
        return u_new

    def solve(self, lam_schedule=(0.02, 0.06, 0.15, 0.35, 0.65, 1.0),
              maxit_stage=40, maxit_final=300, anderson_depth=8, verbose=True):
        """源项延拓 + Anderson 加速不动点迭代。返回 (迭代数, 残差统计)。"""
        total_it = 0
        t0 = time.time()
        for si, lam in enumerate(lam_schedule):
            last = si == len(lam_schedule) - 1
            umax = max(self.u_grid.abs().max().item(), 1e-12)
            tol = (1e-10 if last else 3e-3) * max(1.0, umax)
            x_hist, f_hist = [], []
            it, fnorm = 0, np.inf
            cap = maxit_final if last else maxit_stage
            while it < cap:
                S = self._source(lam)
                u_new = self._solve_linear(S)
                f = u_new - self.u_grid                  # 不动点残差(未阻尼)
                fnorm = f.abs().max().item()
                x_hist.append(self.u_grid.clone())
                f_hist.append(f.clone())
                if len(x_hist) > anderson_depth + 1:
                    x_hist.pop(0)
                    f_hist.pop(0)
                m = len(x_hist) - 1
                if m >= 1:
                    Xh = torch.stack([x_hist[i + 1] - x_hist[i] for i in range(m)])
                    Fh = torch.stack([f_hist[i + 1] - f_hist[i] for i in range(m)])
                    Ff = Fh.reshape(m, -1)
                    A = Ff @ Ff.T
                    A = A + torch.eye(m, dtype=A.dtype, device=A.device) * \
                        (1e-12 * float(A.diagonal().abs().max()) + 1e-300)
                    theta = torch.linalg.solve(A, -(Ff @ f.reshape(-1)))
                    step = f + torch.einsum("i,i...", theta, Xh + Fh)
                else:
                    step = f
                # 安全阀: Anderson 步异常放大时回退普通步
                if step.abs().max().item() > 50.0 * max(fnorm, 1e-10):
                    step = f
                    x_hist, f_hist = [self.u_grid.clone()], [f.clone()]
                self.u_grid = self.u_grid + step
                it += 1
                total_it += 1
                if fnorm <= tol:
                    break
            if verbose:
                log.info(f"  λ={lam:<5g} stage {si + 1}/{len(lam_schedule)}: it={it:>3d} "
                         f"|f|={fnorm:.3e} max|u|={self.u_grid.abs().max().item():.6e} "
                         f"({time.time() - t0:.1f}s)")
        # 残差: 对最终 u_grid 直接谱求 Δu, 与源比较(算子行 j=1..N_r-1 ↔ 网格行 0..N_r-2)
        u_lm = torch.einsum("ja,ak->jk",
                            (self.u_grid * self.qw).to(torch.complex128), self.Yc)
        u_full = np.zeros((self.K, self.N_r + 1), dtype=complex)
        u_full[:, 1:] = u_lm.T.cpu().numpy()
        lap = np.zeros((self.K, self.N_r + 1), dtype=complex)
        l_arr = np.array(self.l_list)
        for l in range(self.L + 1):
            sel = np.where(l_arr == l)[0]
            lap[sel] = (radial_matrix(l, self.r, self.D1, self.D2) @ u_full[sel].T).T
        lap_int = torch.from_numpy(lap[:, 1:-1]).to(self.device)
        R_grid = self._synthesize(lap_int) + self._source(1.0)[:-1]
        R_lm = torch.einsum("ja,ak->jk", (R_grid * self.qw).to(torch.complex128), self.Yc)
        stats = {"res_max": float(R_grid.abs().max()),
                 "res_rms": float(R_grid.pow(2).mean().sqrt()),
                 "res_modal_max": float(R_lm.abs().max())}
        return total_it, stats

    # ---- 任意 Cartesian 点求值 ----
    @torch.no_grad()
    def evaluate(self, pts_abs: np.ndarray, chunk=32768,
                 dtype=torch.float64) -> np.ndarray:
        """绝对坐标点集上的 u。径向 barycentric 插值 × 角向逐点 Y_lm (GPU)。

        dtype=float32 时求值链路用单精度(输出仍 float64, 误差 ~1e-7 相对,
        远低于谱解自身 1e-4 的截断误差), 大网格输出提速 ~50×。
        """
        n = pts_abs.shape[0]
        out = np.empty(n, dtype=np.float64)
        s_nodes = self.s
        wb = ((-1.0) ** np.arange(len(s_nodes))).astype(np.float64)
        wb[0] *= 0.5
        wb[-1] *= 0.5
        wb_t = torch.tensor(wb, dtype=dtype, device=self.device)
        s_t = torch.tensor(s_nodes, dtype=dtype, device=self.device)
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        U = self._U_full.to(cdtype)
        for i0 in range(0, n, chunk):
            p = torch.tensor(pts_abs[i0:i0 + chunk], dtype=dtype, device=self.device)
            d = p - torch.tensor(self.xc, dtype=dtype, device=self.device)
            r = torch.linalg.norm(d, dim=1).clamp(min=1e-30)   # float32 安全下限(1e-300 会下溢)
            sp = (r - self.R0) / (r + self.R0)
            den = sp[:, None] - s_t[None, :]
            den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
            coef = wb_t[None, :] / den
            B = coef / coef.sum(dim=1, keepdim=True)        # (P, N_r+1)
            Ur = B.to(cdtype) @ U.T                         # (P, K) complex
            mu = (d[:, 2] / r).clamp(-1.0, 1.0)
            phi = torch.atan2(d[:, 1], d[:, 0])
            Yp = sph_harm_torch(self.L, mu, phi, self.device, self.l_list,
                                self.m_list, dtype=dtype)
            out[i0:i0 + chunk] = torch.einsum("pk,pk->p", Ur, Yp).real.double().cpu().numpy()
        return out

    # ---- 谱系数导出 / 免重解加载 ----
    def export_coefficients(self) -> np.ndarray:
        """径向谱系数 U (K, N_r+1) complex128: 每个 (l,m) 模态在 Chebyshev 节点
        上的值(含边界行)。npz 存一份仅 ~7-20 MB, 配合 from_coefficients 可对
        任意点集(针尖加密轴、稠密平面)免重解求值。"""
        return self._U_full.to(torch.complex128).cpu().numpy()

    @classmethod
    def from_coefficients(cls, path, device="cuda", verify=True):
        """从含 radial_coeffs 的 npz 重建"只求值"对象(不建算子、不需重解)。

        npz 需含: radial_coeffs (K, N_r+1) complex, raw (8,), xc (3,),
        meta(JSON, 含 N_r/L/R0); 可选 x_ref/u_ref 用于自校验。
        支持 evaluate()(含 float32 加速档), 不支持 solve()。"""
        rd = np.load(path)
        meta = json.loads(str(rd["meta"]))
        obj = cls.__new__(cls)
        obj.raw = np.asarray(rd["raw"], dtype=np.float64)
        obj.masses, obj.xs, obj.Ps, obj.Ss = build_bbh_from_params(obj.raw)
        obj.xc = np.asarray(rd["xc"], dtype=np.float64)
        obj.N_r = int(meta["N_r"])
        obj.L = int(meta["L"])
        obj.R0 = float(meta.get("R0", 15.0))
        obj.device = torch.device(device if torch.cuda.is_available()
                                  or device == "cpu" else "cpu")
        obj.s, _ = cheb_nodes_D(obj.N_r)
        obj.l_list, obj.m_list = [], []
        for l in range(obj.L + 1):
            for m in range(-l, l + 1):
                obj.l_list.append(l)
                obj.m_list.append(m)
        obj.K = len(obj.l_list)
        U = np.asarray(rd["radial_coeffs"], dtype=np.complex128)
        if U.shape != (obj.K, obj.N_r + 1):
            raise ValueError(f"radial_coeffs 形状 {U.shape} 与 "
                             f"(K={obj.K}, N_r+1={obj.N_r + 1}) 不符")
        obj._U_full = torch.from_numpy(U).to(obj.device)
        if verify and "x_ref" in rd:
            n = rd["x_ref"].shape[0]
            idx = np.random.default_rng(0).choice(n, size=min(1000, n), replace=False)
            ue = obj.evaluate(rd["x_ref"][idx].astype(np.float64),
                              dtype=torch.float64)
            ur = rd["u_ref"][idx].astype(np.float64)
            rel = float(np.sqrt(np.sum((ue - ur) ** 2) / max(np.sum(ur ** 2), 1e-30)))
            log.info(f"[from_coefficients] {os.path.basename(path)}: "
                     f"网格自校验 L2RE={rel:.2e} (L={obj.L}, N_r={obj.N_r})")
        return obj


# ── 算子自检 ──────────────────────────────────────────────────

def selftest(L=16, N_r=128, N_th=28, N_ph=48, device="cuda"):
    """用有理解析解验证 Δ 算子与球谐实现(紧化映射下谱收敛):
        u = 1/(1+r²)          (l=0) → Δu = 2(r²-3)/(1+r²)³
        u = z/(1+r²)²         (l=1) → Δu = z(4r²-20)/(1+r²)⁴
        u = (x²-y²)/(1+r²)²   (l=2) → Δu = (x²-y²)(4r²-20)/(1+r²)⁴
    """
    base = dict(N_r=N_r, L=L, N_th=N_th, N_ph=N_ph, R0=15.0, device=device)
    raw = np.array([0.5, 0.5, 3.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    errs = {}
    for name, f, lap in [
            ("l=0 1/(1+r²)", lambda X, Y, Z, R: 1.0 / (1.0 + R * R),
             lambda X, Y, Z, R: 2.0 * (R * R - 3.0) / (1.0 + R * R) ** 3),
            ("l=1 z/(1+r²)²", lambda X, Y, Z, R: Z / (1.0 + R * R) ** 2,
             lambda X, Y, Z, R: Z * (4.0 * R * R - 20.0) / (1.0 + R * R) ** 4),
            ("l=2 (x²-y²)/(1+r²)²",
             lambda X, Y, Z, R: (X * X - Y * Y) / (1.0 + R * R) ** 2,
             lambda X, Y, Z, R: -(X * X - Y * Y) * (4.0 * R * R + 28.0) / (1.0 + R * R) ** 4)]:
        # 把解析函数投影到网格 → 分析 → 谱算子 → 与解析 Δ 比较
        solver = SpectralPunctureSolver.__new__(SpectralPunctureSolver)
        SpectralPunctureSolver.__init__(solver, raw, **base)
        pts = solver.grid_pts
        X = torch.tensor(pts[..., 0] - solver.xc[0], dtype=torch.float64, device=solver.device)
        Yc = torch.tensor(pts[..., 1], dtype=torch.float64, device=solver.device)
        Z = torch.tensor(pts[..., 2], dtype=torch.float64, device=solver.device)
        R = torch.linalg.norm(torch.stack([X, Yc, Z], dim=-1), dim=-1)
        u = f(X, Yc, Z, R)
        lap_exact = lap(X, Yc, Z, R)
        # 分析
        S_lm = torch.einsum("ja,ak->jk", (u * solver.qw).to(torch.complex128), solver.Yc)
        rhs = S_lm.T.cpu().numpy()
        full = np.zeros((solver.K, N_r + 1), dtype=complex)
        full[:, 1:] = rhs
        l_arr = np.array(solver.l_list)
        lap_num = np.zeros((solver.K, N_r + 1), dtype=complex)
        for l in range(L + 1):
            sel = np.where(l_arr == l)[0]
            lap_num[sel] = (radial_matrix(l, solver.r, solver.D1, solver.D2)
                            @ full[sel].T).T
        lap_num_t = torch.from_numpy(lap_num[:, 1:-1]).to(solver.device)
        lap_grid = solver._synthesize(lap_num_t)
        # 算子方程行只覆盖 j=1..N_r-1 ↔ 网格前 N_r-1 个点(末点是 r=0 BC 行, 不比较)
        rel = ((lap_grid - lap_exact[:-1]).abs().max()
               / lap_exact[:-1].abs().max()).item()
        errs[name] = rel
    return errs


# ── 主流程 ────────────────────────────────────────────────────

def main():
    setup_logging("tools", "spectral_reference")
    p = argparse.ArgumentParser(description="任意 BBH 参数的谱方法参考解求解器")
    p.add_argument("--params", default=None,
                   help="8 维参数逗号串: m+,m-,x+,x-,P+y,P-y,S+z,S-z")
    p.add_argument("--out", default=None, help="输出 npz 路径")
    p.add_argument("--label", default="", help="仅写入元信息/命名建议")
    p.add_argument("--selftest", action="store_true", help="只跑算子精度自检")
    p.add_argument("--compare", default=None, help="与已有参考解 npz 对比(不写文件)")
    p.add_argument("--n-r", type=int, default=512)
    p.add_argument("--lmax", type=int, default=48)
    p.add_argument("--n-theta", type=int, default=72)
    p.add_argument("--n-phi", type=int, default=144)
    p.add_argument("--R0", type=float, default=15.0)
    p.add_argument("--grid-n", type=int, default=197, help="输出均匀网格每方向点数(奇数)")
    p.add_argument("--R", type=float, default=30.0, help="输出网格半边长")
    p.add_argument("--device", default="cuda")
    p.add_argument("--maxit-final", type=int, default=200)
    args = p.parse_args()

    if args.selftest:
        log.info("算子自检 (L=16, N_r=128)...")
        errs = selftest(device=args.device)
        for k, v in errs.items():
            log.info(f"  {k:>20s}: 相对误差 {v:.3e}")
        worst = max(errs.values())
        # float64 二阶导算子舍入地板 ~ eps·N_r⁴ ≈ 3e-8 (N_r=128), 阈值取 1e-6
        log.info(f"  {'结论':>20s}: {'通过' if worst < 1e-6 else '未通过'} (阈值 1e-6)")
        return

    if args.params is None:
        sys.exit("需要 --params 或 --selftest")
    raw = np.array([float(v) for v in args.params.split(",")], dtype=np.float64)
    assert raw.shape == (8,), "--params 需要 8 个数"
    masses, xs, Ps, Ss = build_bbh_from_params(raw)
    log.info(f"参数: {raw.tolist()}")
    log.info(f"  m=({masses[0]:.4f},{masses[1]:.4f}) x=({xs[0,0]:.4f},{xs[1,0]:.4f}) "
             f"P+y={Ps[0,1]:.4f} P-y={Ps[1,1]:.4f} S+z={Ss[0,2]:.4f} S-z={Ss[1,2]:.4f}")

    log.info(f"构建求解器 (N_r={args.n_r}, L={args.lmax}, N_θ={args.n_theta}, "
             f"N_φ={args.n_phi}, R0={args.R0})...")
    t0 = time.time()
    solver = SpectralPunctureSolver(raw, N_r=args.n_r, L=args.lmax,
                                    N_th=args.n_theta, N_ph=args.n_phi,
                                    R0=args.R0, device=args.device)
    log.info(f"  网格 {args.n_r}×{args.n_theta}×{args.n_phi} = "
             f"{args.n_r * args.n_theta * args.n_phi:,} 点, "
             f"模态 {solver.K}, 设备 {solver.device} ({time.time() - t0:.1f}s)")

    log.info("求解 (源项延拓 + 阻尼 Picard)...")
    it, stats = solver.solve(maxit_final=args.maxit_final)
    log.info(f"  迭代 {it} 次, 谱残差 max={stats['res_max']:.3e} rms={stats['res_rms']:.3e}")

    if args.compare:
        rd = np.load(args.compare)
        xr, ur = rd["x_ref"], rd["u_ref"].astype(np.float64)
        stride = max(1, xr.shape[0] // 1_500_000)
        xr_s, ur_s = xr[::stride].astype(np.float64), ur[::stride]
        log.info(f"对比 {args.compare}: 采样 {xr_s.shape[0]} 点...")
        t0 = time.time()
        um = solver.evaluate(xr_s)
        l2 = float(np.sqrt(np.sum((um - ur_s) ** 2) / np.sum(ur_s ** 2)))
        rr = np.linalg.norm(xr_s, axis=1)
        log.info(f"  L2RE(本解 vs 该参考) = {l2:.3e}  (评估 {time.time() - t0:.1f}s)")
        for lo, hi in [(0, 0.5), (0.5, 2), (2, 10), (10, 100)]:
            msk = (rr >= lo) & (rr < hi)
            if msk.sum() > 100:
                l2b = float(np.sqrt(np.sum((um[msk] - ur_s[msk]) ** 2)
                                    / np.sum(ur_s[msk] ** 2)))
                log.info(f"    r[{lo},{hi}): L2RE = {l2b:.3e}  ({msk.sum()} pts)")
        return

    # ---- 输出均匀 Cartesian 网格 ----
    n = args.grid_n
    assert n % 2 == 1, "--grid-n 需为奇数(z=0 面落在网格上)"
    axis = np.linspace(-args.R, args.R, n)
    X, Yc, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack([X, Yc, Z], axis=-1).reshape(-1, 3)
    log.info(f"输出网格 {n}³ = {pts.shape[0]:,} 点 (±{args.R})...")
    t0 = time.time()
    u = solver.evaluate(pts, dtype=torch.float32)
    log.info(f"  求值完成 ({time.time() - t0:.1f}s), u ∈ [{u.min():.6e}, {u.max():.6e}]")

    if args.out is None:
        args.out = os.path.join(PAPER, "tools", "refs", "ref_custom.npz")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"params": raw.tolist(), "label": args.label,
            "N_r": args.n_r, "L": args.lmax, "N_theta": args.n_theta,
            "N_phi": args.n_phi, "R0": args.R0, "grid_n": n, "R": args.R,
            "iterations": it, "res_max": stats["res_max"], "res_rms": stats["res_rms"],
            "method": "spherical-harmonic spectral, continuation+Picard"}
    np.savez(args.out,
             x_ref=pts.astype(np.float32),
             u_ref=u.astype(np.float32),
             radial_coeffs=solver.export_coefficients(),
             raw=raw, xc=solver.xc.astype(np.float64),
             meta=json.dumps(meta))
    log.info(f"完成: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

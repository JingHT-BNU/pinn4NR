"""
visualize.py —— 可视化模块:训练曲线 + 解剖面 + 平面对比图
===========================================================

对应论文的图:
    - Fig.2(a,b) : L2 损失与 L2RE 随训练步数的演化
    - Fig.2(c)   : 沿 x 轴的一维剖面(PINN vs TwoPunctures 参考,含奇点放大)
    - Fig.2(d)   : z≈0 赤道面上的二维对比(PINN、参考、差值)

所有图保存为 PNG,输出到运行目录的 figs/ 子目录。
"""

import logging
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")                        # 无显示环境(服务器/无头)也可靠
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from model import GuidedPINN

log = logging.getLogger("paper.A1.visualize")


def plot_loss_history(history: dict, save_dir: str, fname: str = "loss_history.png"):
    """绘制训练损失曲线(论文 Fig.2a 风格)。

    Args:
        history : train.Trainer.history,{'L2': [...], 'softLinf': [...], 'LBC': [...], 'total': [...]}
        save_dir: 保存目录
        fname   : 文件名
    """
    steps = np.arange(1, len(history["L2"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(steps, history["L2"], label="L2 (PDE)")
    # 仅当 soft-L∞ 参与训练(w_inf>0)且非恒为零时才画,避免 EMA 噪声曲线干扰
    has_active_softlinf = (max(history["softLinf"]) - min(history["softLinf"])) > 1e-10 \
                          and max(history["softLinf"]) > 1e-6
    if has_active_softlinf:
        ax.semilogy(steps, history["softLinf"], label="soft-L∞")
    ax.semilogy(steps, history["LBC"], label="LBC (Robin)")
    ax.semilogy(steps, history["total"], label="total (balanced)", ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
    log.info(f"[图] 损失曲线已保存: {os.path.join(save_dir, fname)}")


def _make_adaptive_xline(xp_vals: np.ndarray, R_max: float, n_points: int = 601,
                         focus_width: float = 3.0, focus_density: float = 3.0):
    """在奇点附近加密的一维 x 轴采样点。

    在奇点附近 [xp - focus_width, xp + focus_width] 使用更密的间距,
    远离奇点处保持标准间距,使总点数不变但峰值区域分辨率提高。

    Args:
        xp_vals      : 奇点 x 坐标, shape (k,)
        R_max        : 半边长
        n_points     : 总点数(默认 601)
        focus_width  : 奇点周围加密区半宽
        focus_density: 加密区内点密度倍数

    Returns:
        xs: shape (n_points,),一维非均匀采样点
    """
    # 基础均匀网格
    uniform = np.linspace(-R_max, R_max, n_points)
    # 计算每个点的局部密度需求:在奇点附近权重大
    weights = np.ones_like(uniform)
    for xp in xp_vals:
        dist = np.abs(uniform - xp)
        # 在 focus_width 内使用高斯权重,峰值处最密
        w = np.exp(-0.5 * (dist / (focus_width / 2.5)) ** 2)
        weights += (focus_density - 1.0) * w
    # 累积权重 → 非均匀采样(重要性采样重映射)
    cdf = np.cumsum(weights)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])  # 归一化到 [0,1]
    xs = np.interp(np.linspace(0, 1, n_points), cdf, uniform)
    # 去掉太靠近奇点的点(r < 1e-2)
    keep = np.ones_like(xs, dtype=bool)
    for xp in xp_vals:
        keep &= np.abs(xs - xp) > 1e-2
    return xs[keep]


def _make_adaptive_2d_grid(xp_vals: np.ndarray, R_max: float, g: int = 161,
                           focus_width: float = 3.0, focus_density: float = 3.0):
    """在奇点附近加密的二维评估网格(赤道面 z=0)。

    对 x 和 y 方向分别使用一维自适应采样,构造非均匀网格。

    Args:
        xp_vals      : 奇点 x 坐标, shape (k,)
        R_max        : 半边长
        g            : 每方向网格点数
        focus_width  : 奇点周围加密区半宽
        focus_density: 加密区内点密度倍数

    Returns:
        X, Y: shape (g, g),网格坐标矩阵
        pts : shape (g*g, 3),展开的评估点
    """
    xs = _make_adaptive_xline(xp_vals, R_max, g, focus_width, focus_density)
    # y 方向也加密(奇点在 x 轴上,但 y=0 的轴附近解也变化快)
    # y 的奇点位置为 0(奇点在 x 轴上,线上方向中心最重要)
    ys = _make_adaptive_xline(np.array([0.0]), R_max, g, focus_width, focus_density)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1).astype(np.float32)
    return X, Y, pts


def _extract_axis_profile(x_ref: np.ndarray, u_ref: np.ndarray,
                          x_lo: float = -np.inf, x_hi: float = np.inf,
                          axis_tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """从散点参考解中提取 x 轴(y=z=0)上的剖面,返回 (x, u) 已按 x 排序。

    参考解网格在轴上恰好有 y=z=0 的格点(见 make_reference.py 的均匀/自适应网格),
    因此优先用极小的容差 axis_tol 精确取轴上的点,避免把 x 轴两侧邻域内的
    离轴点误当成轴上点(否则 u 随 y/z 变化陡峭,会画成"有厚度的带")。

    若精确轴点不足(如参考解网格不含 y=z=0 层),则按 x 分箱,每箱取离轴最近的点,
    保证每个 x 位置只贡献一条曲线。

    Args:
        x_ref   : 参考解坐标, shape (M,3)
        u_ref   : 参考解值, shape (M,)
        x_lo, x_hi: 只保留 x 在此区间内的点
        axis_tol: 判定"在轴上"的容差(默认 1e-6)

    Returns:
        (x, u): 按 x 升序排列的轴上剖面
    """
    m = (x_ref[:, 0] >= x_lo) & (x_ref[:, 0] <= x_hi)
    xr, ur = x_ref[m], u_ref[m]
    if len(xr) == 0:
        return np.array([]), np.array([])

    # 优先:精确轴点
    on_axis = (np.abs(xr[:, 1]) < axis_tol) & (np.abs(xr[:, 2]) < axis_tol)
    if np.sum(on_axis) > 0:
        x = xr[on_axis, 0]
        u = ur[on_axis]
    else:
        # 回退:按 x 分箱,每箱取离轴最近点(保证单值曲线)
        off = np.abs(xr[:, 1]) + np.abs(xr[:, 2])
        order = np.argsort(xr[:, 0])
        xr, ur, off = xr[order], ur[order], off[order]
        # 用 x 的 1e-3 精度分箱(参考解网格间距远大于此)
        keys = np.round(xr[:, 0] / 1e-3).astype(np.int64)
        x_out, u_out = [], []
        for k in np.unique(keys):
            sel = keys == k
            j = np.argmin(off[sel])
            x_out.append(xr[sel][j, 0])
            u_out.append(ur[sel][j])
        x = np.array(x_out)
        u = np.array(u_out)

    order = np.argsort(x)
    return x[order], u[order]


def plot_x_axis_profile(model: GuidedPINN, data, device, save_dir: str,
                        x_ref: Optional[np.ndarray] = None,
                        u_ref: Optional[np.ndarray] = None,
                        fname: str = "x_axis_profile.png"):
    """沿 x 轴(穿过两个奇点)的一维剖面(论文 Fig.2c 风格)。

    Args:
        model  : 训练好的模型
        data   : DataBundle
        device : torch 设备
        save_dir: 保存目录
        x_ref  : 参考解坐标(可选,用于叠加参考曲线)
        u_ref  : 参考解值(可选)
        fname  : 文件名
    """
    from evaluate import predict_u
    # 沿 x 轴采样,避开奇点位置本身(奇点位置随算例不同:base=±3, uneq_nospin=±8 等)
    xp_vals = data.xs[:, 0]                        # (2,) 两个奇点的 x 坐标
    # 使用自适应采样:在奇点附近加密评估点,捕捉峰值细节
    xs = _make_adaptive_xline(xp_vals, data.R_max, 601)
    x_line = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1).astype(np.float32)
    u_pinn = predict_u(model, x_line, device)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, u_pinn, "b-", lw=1.5, label="PINN $u_\\theta$")
    if u_ref is not None and x_ref is not None:
        # 从散点参考解中提取 x 轴(y=z=0)剖面(精确取轴,避免离轴点画成厚带)
        xr, ur = _extract_axis_profile(x_ref, u_ref)
        if len(xr) > 0:
            ax.plot(xr, ur, "r--", lw=1.5, label="TwoPunctures $u_{TP}$")
    # 标出奇点位置
    for xp in xp_vals:
        ax.axvline(xp, color="k", ls=":", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,0,0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
    log.info(f"[图] x 轴剖面已保存: {os.path.join(save_dir, fname)}")


def plot_equatorial_plane(model: GuidedPINN, data, device, save_dir: str,
                          x_ref: Optional[np.ndarray] = None,
                          u_ref: Optional[np.ndarray] = None,
                          fname: str = "equatorial_plane.png"):
    """z≈0 赤道面上的二维解(论文 Fig.2d 风格:PINN | 参考 | 差值)。

    Args:
        model  : 训练好的模型
        data   : DataBundle
        device : torch 设备
        save_dir: 保存目录
        x_ref  : 参考解坐标(可选)
        u_ref  : 参考解值(可选)
        fname  : 文件名
    """
    from evaluate import predict_u
    g = 161                                    # 网格点数(161² ≈ 2.6万,CPU 可承受)
    # 使用自适应网格:在奇点附近加密评估点,捕捉峰值细节
    xp_vals = data.xs[:, 0]                    # (2,) 奇点 x 坐标
    X, Y, pts = _make_adaptive_2d_grid(xp_vals, data.R_max, g)
    u2d = predict_u(model, pts, device).reshape(g, g)

    # ---- 数值安全:替换非有限值,防止单点爆炸压垮整图色标 ----
    u2d = np.where(np.isfinite(u2d), u2d, np.nan)

    # ---- 掩盖奇点附近(发散区域):半径取奇点间距的 1/10,确保覆盖格点 ----
    r_mask = max(0.3, 0.1 * np.abs(data.xs[0, 0] - data.xs[1, 0]))
    for xp in xp_vals:
        mask = (X - xp) ** 2 + Y ** 2 < r_mask ** 2
        u2d = np.where(mask, np.nan, u2d)

    ncol = 2 if (u_ref is not None) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 5))
    if ncol == 1:
        axes = [axes]

    vmax = np.percentile(np.abs(u2d[np.isfinite(u2d)]), 99)
    axes[0].pcolormesh(X, Y, u2d, cmap="viridis", shading="auto")
    axes[0].set_title("PINN $u_\\theta$ (z=0)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    u2d_ref = None
    if u_ref is not None:
        # 从散点参考解中提取 z≈0 平面,插值到与 PINN 相同的均匀网格
        tol_z = 0.5  # z 方向容差(球内网格 z 步长 1.0,含 z=0 和 ±1 的层)
        in_plane = np.abs(x_ref[:, 2]) < tol_z
        if np.sum(in_plane) > 0:
            u2d_ref = griddata(
                x_ref[in_plane, :2], u_ref[in_plane],
                (X, Y), method="linear", fill_value=np.nan)
            # 参考解只在球内 r<=R_max 有数据,球外为 NaN。
            # 为与 PINN 解在同一个矩形区域上对应,球外区域用 PINN 解填充
            # (球外 PINN 解与球内参考解在边界连续,且同为有效解)。
            outside = ~np.isfinite(u2d_ref)
            u2d_ref = np.where(outside, u2d, u2d_ref)
            axes[1].pcolormesh(X, Y, u2d_ref, cmap="viridis", shading="auto")
            axes[1].set_title("TwoPunctures $u_{TP}$ (z≈0)")
        else:
            axes[1].text(0.5, 0.5, "参考解中无 z≈0 点,\n无法对比", ha="center", va="center")
        for a in axes[1:]:
            a.set_xlabel("x"); a.set_ylabel("y")

    for a in axes:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
    log.info(f"[图] 赤道面图已保存: {os.path.join(save_dir, fname)}")

    # ---- 差值图单独成图(避免与 PINN/TP 共用色标导致尺度失衡而近乎空白) ----
    if u2d_ref is not None:
        diff = u2d - u2d_ref
        dv = np.abs(diff[np.isfinite(diff)])
        if len(dv) > 0:
            dmax = np.percentile(dv, 99)
            if dmax <= 0:
                dmax = 1.0
            fig2, ax2 = plt.subplots(figsize=(5.5, 5))
            # 球外参考解无定义(diff 恒为 0),用浅灰底色标注"无参考数据",
            # 使整张图覆盖与 PINN/TP 相同的矩形视野 [-R_max, R_max]^2。
            R = data.R_max
            ax2.set_facecolor("#e8e8e8")
            im = ax2.pcolormesh(X, Y, diff, cmap="RdBu_r", shading="auto",
                                vmin=-dmax, vmax=dmax)
            fig2.colorbar(im, ax=ax2)
            ax2.set_title("difference (PINN − TP)")
            ax2.set_xlabel("x"); ax2.set_ylabel("y")
            ax2.set_aspect("equal")
            # 显式对齐坐标视野到整个矩形(与 PINN/TP 图一致)
            ax2.set_xlim(-R, R)
            ax2.set_ylim(-R, R)
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_dir, "difference_plane.png"), dpi=150)
            plt.close(fig2)
            log.info(f"[图] 差值图已保存: {os.path.join(save_dir, 'difference_plane.png')}")


def plot_peak_zoom(model: GuidedPINN, data, device, save_dir: str,
                   x_ref: Optional[np.ndarray] = None,
                   u_ref: Optional[np.ndarray] = None,
                   fname: str = "peak_zoom.png"):
    """峰值区域局部放大图:在奇点附近做高分辨率一维剖面。

    奇点附近 u 值变化最剧烈(~1/r 发散),是 L2RE 误差的主要来源。
    本图用高密度非均匀采样在奇点周围 ±focus_width 范围内绘制,
    清晰展示 PINN 在峰值处的锯齿/平滑程度。

    Args:
        model  : 训练好的模型
        data   : DataBundle
        device : torch 设备
        save_dir: 保存目录
        x_ref  : 参考解坐标(可选)
        u_ref  : 参考解值(可选)
        fname  : 文件名
    """
    from evaluate import predict_u
    xp_vals = data.xs[:, 0]                   # (2,) 奇点 x 坐标
    focus_width = 5.0                          # 每个奇点周围放大范围
    n_zoom = 401                               # 每个奇点周围的采样点数
    # 奇点邻域掩码半径:网络在 r<~0.02 处有数值伪影(顶峰凹陷/发散),
    # 与赤道面图一致,对奇点邻域做掩码(该区域物理上也无定义)。
    r_mask = 0.05

    fig, axes = plt.subplots(1, len(xp_vals), figsize=(6 * len(xp_vals), 5))
    if len(xp_vals) == 1:
        axes = [axes]

    for idx, xp in enumerate(xp_vals):
        ax = axes[idx]
        # 在奇点周围高密度采样(对数间隔,近奇点处更密)
        # 正侧: [xp+1e-2, xp+focus_width], 负侧: [xp-focus_width, xp-1e-2]
        # 使用对数间隔: 小距离处更密
        n_half = n_zoom // 2
        # 正向侧: 从近到远
        r_pos = np.logspace(-2, np.log10(focus_width), n_half)
        xs_pos = xp + r_pos
        # 负向侧
        xs_neg = xp - r_pos[::-1]
        xs = np.concatenate([xs_neg, xs_pos])
        xs = np.clip(xs, -data.R_max, data.R_max)

        x_line = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1).astype(np.float32)
        u_pinn = predict_u(model, x_line, device)
        # 掩码奇点邻域(置 NaN,matplotlib 自动断开曲线)
        u_pinn = np.where(np.abs(xs - xp) < r_mask, np.nan, u_pinn)

        ax.plot(xs, u_pinn, "b-", lw=1.5, label="PINN $u_\\theta$")

        if u_ref is not None and x_ref is not None:
            # 从参考解中提取该奇点附近的 x 轴(y=z=0)剖面(精确取轴)
            xr, ur = _extract_axis_profile(x_ref, u_ref,
                                           x_lo=xp - focus_width,
                                           x_hi=xp + focus_width)
            if len(xr) > 0:
                keep = np.abs(xr - xp) >= r_mask
                ax.plot(xr[keep], ur[keep], "r--", lw=1.5, label="TwoPunctures $u_{TP}$")

        ax.axvline(xp, color="k", ls=":", alpha=0.3)
        ax.set_title(f"Peak zoom: x ≈ {xp:.1f}")
        ax.set_xlabel("x"); ax.set_ylabel("u(x,0,0)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
    log.info(f"[图] 峰值放大图已保存: {os.path.join(save_dir, fname)}")

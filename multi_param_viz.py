"""
multi_param_viz.py —— 多参数参数化 PINN 可视化
================================================

生成 4 张图:
    1. loss_history    — 训练损失曲线 (L2, LBC, L_ref)
    2. param_sensitivity — 参数敏感性: u(0) vs 各参数
    3. axis_profiles   — 多个配置的 x 轴剖面对比
    4. pde_residual_map — PDE 残差在参数空间的分布
"""

import argparse, json, logging, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
# 中文标签需要 CJK 字体(Windows 自带微软雅黑; Linux 回退 Noto/文泉驿)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball
from logutil import setup_logging
import physics
from multi_param_model import (
    MultiParamGuidedPINN, normalize_params, build_bbh_from_params,
    PARAM_NAMES, PARAM_LO, PARAM_HI, BASE_RAW,
)
from multi_param_train import predict, resolve_input_path
from multi_param_eval import load_model, pde_selfcheck

log = logging.getLogger("paper.A3.multi_param_viz")


def plot_loss_history(exp_dir, save_path):
    """训练损失曲线。"""
    hist = json.load(open(os.path.join(exp_dir, "history.json")))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps = np.arange(1, len(hist["L2"]) + 1)

    # 上图: PDE + Robin
    ax = axes[0]
    ax.semilogy(steps, hist["L2"], alpha=0.3, label="L2 (raw)", color="C0")
    ax.semilogy(steps, hist["LBC"], alpha=0.3, label="LBC (raw)", color="C1")
    # EMA 平滑
    def ema(vals, alpha=0.99):
        out = [vals[0]]
        for v in vals[1:]:
            out.append(alpha * out[-1] + (1 - alpha) * v)
        return np.array(out)
    ax.semilogy(steps, ema(hist["L2"]), label="L2 (EMA)", color="C0", lw=2)
    ax.semilogy(steps, ema(hist["LBC"]), label="LBC (EMA)", color="C1", lw=2)
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.set_title("PDE & Robin Loss")
    ax.grid(True, alpha=0.3)

    # 下图: 参考损失
    ax = axes[1]
    lref = np.array(hist["L_ref"])
    mask = lref > 0
    if mask.any():
        ax.semilogy(steps[mask], lref[mask], alpha=0.3, label="L_ref (raw)", color="C2")
        ax.semilogy(steps[mask], ema(lref[mask]), label="L_ref (EMA)", color="C2", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("L_ref")
    ax.legend(loc="upper right")
    ax.set_title("Reference Loss (base case only)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved: {save_path}")


def plot_param_sensitivity(model, ckpt, info, device, save_path):
    """参数敏感性: 沿每个维度变化, 观察 u(0,0,0) 的变化。"""
    model.eval()
    x_origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # base 配置的 κ(近似: 真实 κ 随参数变化, 定性观察足够)
    base_cfg = min(info["train"], key=lambda c: np.linalg.norm(
        np.array(c["raw"]) - BASE_RAW))
    base_kappa = base_cfg["kappa"]

    # 选择要展示的维度
    dims_to_show = [0, 1, 2, 4, 6]  # m+, m-, x+, P+y, S+z
    dim_labels = ["m_+", "m_-", "x_+", "P_{+y}", "S_{+z}"]

    fig, axes = plt.subplots(1, len(dims_to_show), figsize=(4 * len(dims_to_show), 3.5))
    if len(dims_to_show) == 1:
        axes = [axes]

    n_scan = 21
    for ax, dim_idx, dim_label in zip(axes, dims_to_show, dim_labels):
        vals = []
        ts = np.linspace(-0.9, 0.9, n_scan)
        for t in ts:
            raw = BASE_RAW.copy()
            raw[dim_idx] = PARAM_LO[dim_idx] + (t + 1) / 2 * (PARAM_HI[dim_idx] - PARAM_LO[dim_idx])
            masses, xs, Ps, Ss = build_bbh_from_params(raw)
            try:
                u_val = predict(model, x_origin, raw, base_kappa, device)
                vals.append(u_val[0])
            except Exception:
                vals.append(float("nan"))
        vals = np.array(vals)
        ax.plot(ts, vals, "o-", markersize=3, color="C0")
        ax.set_xlabel(f"{dim_label} (norm)")
        ax.set_ylabel("u(0)")
        ax.set_title(f"u(0) vs {dim_label}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved: {save_path}")


def _load_xaxis_reference(ref_x, ref_u):
    """从谱方法参考解中提取 x 轴剖面 (|y|,|z| 在一个网格层内)。

    返回 (x_sorted, u_sorted)；网格层厚由点数与包围盒自动估计。
    """
    n_eff = max(round(ref_x.shape[0] ** (1 / 3)), 2)
    span = float(np.ptp(ref_x[:, 0]))
    tol = 0.51 * span / (n_eff - 1)
    for _ in range(3):
        m = (np.abs(ref_x[:, 1]) < tol) & (np.abs(ref_x[:, 2]) < tol)
        if m.sum() >= 30:
            break
        tol *= 2.0
    pts, u = ref_x[m], ref_u[m]
    order = np.argsort(pts[:, 0])
    return pts[order, 0], u[order]


def _load_refs_dir(refs_dir):
    """加载 paper/tools/refs/ 下全部谱方法参考解 (spectral_reference.py 生成)。

    返回 [{"raw", "x_ref", "u_ref", "path"}]; 无目录或为空则返回 []。"""
    refs = []
    if refs_dir is None or not os.path.isdir(refs_dir):
        return refs
    for fn in sorted(os.listdir(refs_dir)):
        if not fn.endswith(".npz"):
            continue
        rd = np.load(os.path.join(refs_dir, fn))
        if "raw" not in rd or "x_ref" not in rd:
            continue
        refs.append({"raw": np.array(rd["raw"], dtype=np.float64),
                     "x_ref": rd["x_ref"], "u_ref": rd["u_ref"],
                     "path": os.path.join(refs_dir, fn)})
    log.info(f"参考解目录 {refs_dir}: {len(refs)} 个文件")
    return refs


_EV_CACHE = {}


def _ref_evaluator(ref_entry, device):
    """从含 radial_coeffs 的参考解 npz 构建免重解求值器(带缓存)。

    旧格式(无 radial_coeffs)返回 None, 调用方回退到网格散点/切片。"""
    if ref_entry is None or not ref_entry.get("path"):
        return None
    key = (ref_entry["path"], str(device))
    if key not in _EV_CACHE:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        from spectral_reference import SpectralPunctureSolver
        try:
            _EV_CACHE[key] = SpectralPunctureSolver.from_coefficients(
                ref_entry["path"], device=str(device))
        except Exception as e:
            log.warning(f"参考解求值器构建失败({ref_entry['path']}): {e}, "
                        f"回退网格散点")
            _EV_CACHE[key] = None
    return _EV_CACHE[key]


def _dense_axis(xs_punc, r_min=0.015):
    """x 轴稠密采样: 均匀 3001 点 + 每奇点 ±1.2 内加密 2001 点。
    针尖分辨率 Δx≈0.0012。模型/引导/参考三条曲线统一用这套采样——
    引导与模型若用粗采样(Δx≈0.019), 尖峰(本征宽度 ~0.05)两侧会被画成
    假"断点"(v5 复盘教训: 引导函数本身经 dx=1e-4 扫描验证处处光滑)。"""
    parts = [np.linspace(-28.0, 28.0, 3001)]
    for xp in xs_punc:
        seg = xp + np.linspace(-1.2, 1.2, 2001)
        parts.append(seg[np.abs(seg - xp) > r_min])
    return np.unique(np.round(np.concatenate(parts), 9))


def _match_ref(cfg_raw, refs, legacy_ref=None):
    """按 8 维参数匹配参考解; 找不到返回 None。"""
    for r in refs or []:
        if np.allclose(np.array(cfg_raw, dtype=np.float64), r["raw"],
                       rtol=0.0, atol=1e-9):
            return r
    return legacy_ref


def _ref_plane_slice(ref):
    """从均匀立方网格参考解中取 z=0 切片。返回 (axis, u_plane[n,n]) 或 None。"""
    if ref is None or "raw" not in ref:
        return None
    x_ref, u_ref = ref["x_ref"], ref["u_ref"].astype(np.float64)
    M = x_ref.shape[0]
    n = int(round(M ** (1 / 3)))
    if n ** 3 != M or n < 9:
        return None
    xs = np.unique(x_ref[:, 0])
    ys = np.unique(x_ref[:, 1])
    zs = np.unique(x_ref[:, 2])
    if len(xs) != n or len(ys) != n or len(zs) != n:
        return None
    k = int(np.argmin(np.abs(zs)))
    if abs(zs[k]) > 1e-6:
        return None
    u3 = u_ref.reshape(n, n, n)          # x_ref 由 meshgrid('ij') 展平
    return xs, u3[:, :, k]


def plot_axis_profiles(model, ckpt, info, device, save_path,
                       ref=None, refs=None, R_max=30.0):
    """x 轴剖面 (y=z=0): 每个配置一个子图, 叠加 κ·u_g 基线与谱方法参考解。

    有参考解的配置(base 及 tools/refs/ 内已生成的)叠加谱方法参考解散点;
    其余以 ansatz 零修正基线 κ·u_g 作对照。
    仅在精确命中奇点处(r<0.015)断线; 掩膜点置 NaN, 不用直线跨越缺口。"""
    model.eval()

    configs = info["train"]
    # base + 按 κ 分位数取 5 个代表性配置
    base_cfg = min(configs, key=lambda c: np.linalg.norm(
        np.array(c["raw"]) - BASE_RAW))
    order = np.argsort([c["kappa"] for c in configs])
    picks = [configs[order[q]] for q in
             np.linspace(0, len(configs) - 1, 5).astype(int)]
    selected = []
    for c in [base_cfg] + picks:
        if all(c["label"] != s["label"] for s in selected):
            selected.append(c)
    selected = selected[:6]

    # 剖面采样点: 三条曲线统一用 _dense_axis 稠密采样(针尖 Δx≈0.0012)。
    # guide_u 只在精确命中奇点(r→0)时因 84ℓ⁵/R 与 84ln(ℓ)/R² 的浮点相消
    # 产生伪影(实测 r=0.008 处已干净), 故仅剔除 r<0.015 的点;
    # 被剔除处置 NaN 断线, 绝不用直线跨越缺口(否则出现"削峰"假象)。

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8), sharex=True)
    axes = axes.ravel()

    for i, (ax, cfg) in enumerate(zip(axes, selected)):
        masses, xs, Ps, Ss = build_bbh_from_params(np.array(cfg["raw"]))
        xs_d = _dense_axis(xs[:, 0])
        pts = np.zeros((len(xs_d), 3), dtype=np.float64)
        pts[:, 0] = xs_d
        keep = np.ones(len(xs_d), dtype=bool)
        for xp in xs[:, 0]:
            keep &= np.abs(xs_d - xp) > 0.015
        xt = torch.from_numpy(pts)
        mat = torch.from_numpy(masses.astype(np.float64))
        xst = torch.from_numpy(xs.astype(np.float64))
        Pt = torch.from_numpy(Ps.astype(np.float64))
        St = torch.from_numpy(Ss.astype(np.float64))
        ug = cfg["kappa"] * physics.guide_u(xt, mat, xst, Pt, St).numpy()
        up = predict(model, pts, cfg["raw"], cfg["kappa"], device).astype(float)
        up[~keep] = np.nan
        ug[~keep] = np.nan

        ax.plot(xs_d, up, "-", lw=0.8, color="C0", label="模型 u_θ")
        ax.plot(xs_d, ug, "--", lw=0.55, color="C2", label="引导基线 κ·u_g")
        ref_k = _match_ref(cfg["raw"], refs, legacy_ref=ref)
        ev = _ref_evaluator(ref_k, device) if ref_k is not None else None
        if ev is not None:
            # 谱系数稠密求值 → 光滑曲线, 与模型/引导同采样
            u_r = ev.evaluate(pts, chunk=16384, dtype=torch.float64)
            u_r = np.where(keep, u_r, np.nan)
            ax.plot(xs_d, u_r, "-", lw=0.9, color="C3", alpha=0.9,
                    label="谱方法参考解")
        elif ref_k is not None:
            ref_curve = _load_xaxis_reference(ref_k["x_ref"], ref_k["u_ref"])
            ax.plot(ref_curve[0], ref_curve[1], ".", ms=1.2, color="C3",
                    alpha=0.55, label="谱方法参考解(网格)")
        ax.set_title(f"{cfg['label']}  (κ={cfg['kappa']:.3f})", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-R_max + 2.0, R_max - 2.0)

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    fig.suptitle("x 轴剖面 (y=z=0): 模型 vs 引导基线 κ·u_g vs 谱方法参考解",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    # 1000 dpi: 6 子图细节密集, 曲线细线宽以便区分
    fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved: {save_path}")


def plot_equatorial_difference(model, ckpt, info, device, save_path,
                               ref=None, refs=None, R_max=30.0):
    """赤道面 (z=0) difference 图: u_θ − u_ref 的带符号差值图。

    每个有参考解的配置一个子图; 奇点邻域 r<0.05 置 NaN;
    色标对称 (±|diff| 的 99.5 分位), 发散色图 RdBu_r。"""
    model.eval()

    configs = info["train"]
    base_cfg = min(configs, key=lambda c: np.linalg.norm(
        np.array(c["raw"]) - BASE_RAW))
    order = np.argsort([c["kappa"] for c in configs])
    picks = [configs[order[q]] for q in
             np.linspace(0, len(configs) - 1, 5).astype(int)]
    selected = []
    for c in [base_cfg] + picks:
        if all(c["label"] != s["label"] for s in selected):
            selected.append(c)
    selected = selected[:6]

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.ravel()

    for i, (ax, cfg) in enumerate(zip(axes, selected)):
        ref_k = _match_ref(cfg["raw"], refs, legacy_ref=ref)
        ev = _ref_evaluator(ref_k, device) if ref_k is not None else None
        if ev is not None:
            # 谱系数稠密求值 (501², Δ≈0.12): 针尖结构不再受 197³ 参考网格限制
            n_plane = 501
            axis = np.linspace(-R_max, R_max, n_plane)
            Xp, Yp = np.meshgrid(axis, axis, indexing="ij")
            pts_plane = np.stack([Xp, Yp, np.zeros_like(Xp)], axis=-1).reshape(-1, 3)
            u_ref_plane = ev.evaluate(pts_plane, chunk=16384,
                                      dtype=torch.float32).reshape(
                                          n_plane, n_plane).astype(float)
        else:
            plane = _ref_plane_slice(ref_k)
            if plane is None:
                ax.text(0.5, 0.5, "无参考解\n(可用 spectral_reference.py 生成)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{cfg['label']}  (κ={cfg['kappa']:.3f})", fontsize=10)
                ax.axis("off")
                continue
            axis, u_ref_plane = plane
            n = len(axis)
            Xp, Yp = np.meshgrid(axis, axis, indexing="ij")
            pts_plane = np.stack([Xp, Yp, np.zeros_like(Xp)], axis=-1).reshape(-1, 3)
        n = len(axis)
        pred = predict(model, pts_plane, cfg["raw"], cfg["kappa"],
                       device).astype(float).reshape(n, n)
        diff = pred - u_ref_plane
        # 奇点邻域掩膜
        for xp in build_bbh_from_params(np.array(cfg["raw"]))[1][:, 0]:
            diff[(Xp - xp) ** 2 + Yp ** 2 < 0.05 ** 2] = np.nan
        vmax = float(np.nanpercentile(np.abs(diff), 99.5))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1e-12
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.85")
        im = ax.imshow(diff.T, origin="lower", extent=[axis[0], axis[-1], axis[0], axis[-1]],
                       cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        for xp in build_bbh_from_params(np.array(cfg["raw"]))[1][:, 0]:
            ax.plot(xp, 0, "k^", ms=4)
        ax.set_title(f"{cfg['label']}  (κ={cfg['kappa']:.3f})", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("赤道面 (z=0): u_θ − 谱方法参考解 差值图 (色标对称, ±99.5 分位)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved: {save_path}")


def plot_pde_residual_scan(model, ckpt, info, device, save_path, R_max=30.0):
    """PDE 残差在参数空间的分布: 对每个训练配置计算平均 |R|。"""
    model.eval()

    train_pde = []
    for cfg in info["train"]:
        r = pde_selfcheck(model, cfg["raw"], cfg["kappa"], device, n_pts=3000, R_max=R_max)
        train_pde.append(abs(r).mean())

    val_pde = []
    for cfg in info["val"]:
        r = pde_selfcheck(model, cfg["raw"], cfg["kappa"], device, n_pts=3000, R_max=R_max)
        val_pde.append(abs(r).mean())

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(range(len(train_pde)), train_pde, "o", markersize=3, label="train", alpha=0.6)
    ax.semilogy(range(len(train_pde), len(train_pde) + len(val_pde)),
                val_pde, "s", markersize=3, label="val", alpha=0.6, color="C1")
    ax.axhline(np.mean(train_pde), color="C0", ls="--", alpha=0.5, label=f"train mean={np.mean(train_pde):.2e}")
    ax.axhline(np.mean(val_pde), color="C1", ls="--", alpha=0.5, label=f"val mean={np.mean(val_pde):.2e}")
    ax.set_xlabel("Config index")
    ax.set_ylabel("Mean |PDE residual|")
    ax.set_title("PDE residual across parameter space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved: {save_path}")


def main():
    setup_logging("A3", "multi_param_viz")
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="runs/multi_param_a1")
    p.add_argument("--reference", default=None,
                   help="谱方法参考解 npz(base 配置), 用于剖面对比")
    p.add_argument("--refs-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "refs"),
        help="谱方法参考解目录(spectral_reference.py 生成, 按参数自动匹配)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # 输入路径解析: 优先按 CWD 相对路径, 不存在则锚定到 paper/(与运行时 CWD 无关)
    args.exp = resolve_input_path(args.exp)
    args.reference = resolve_input_path(args.reference)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu")
    log.info(f"设备: {device}")

    model, ckpt, info = load_model(args.exp, device)
    R_max = ckpt.get("kappa_cache_meta", {}).get("R_max", 30.0)
    fig_dir = os.path.join(args.exp, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    ref = None
    if args.reference:
        rd = np.load(args.reference)
        ref = {"x_ref": rd["x_ref"], "u_ref": rd["u_ref"]}
        log.info(f"参考解: {ref['x_ref'].shape[0]} 点 (用于剖面叠加)")
    refs = _load_refs_dir(resolve_input_path(args.refs_dir))

    log.info("生成可视化...")
    plot_loss_history(args.exp, os.path.join(fig_dir, "mp_loss_history.png"))
    plot_param_sensitivity(model, ckpt, info, device,
                           os.path.join(fig_dir, "mp_param_sensitivity.png"))
    plot_axis_profiles(model, ckpt, info, device,
                       os.path.join(fig_dir, "mp_axis_profiles.png"),
                       ref=ref, refs=refs, R_max=R_max)
    plot_equatorial_difference(model, ckpt, info, device,
                               os.path.join(fig_dir, "mp_equatorial_difference.png"),
                               ref=ref, refs=refs, R_max=R_max)
    plot_pde_residual_scan(model, ckpt, info, device,
                           os.path.join(fig_dir, "mp_pde_residual.png"), R_max=R_max)
    log.info("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

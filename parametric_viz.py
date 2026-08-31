"""
parametric_viz.py —— 参数化 PINN 结果可视化
============================================

图:
    1. param_axis_profiles.png        — 逐参数 x 轴剖面(10 配置按 q 从小到大排列,
                                        每参数一个子图): 模型 u_θ / 引导基线 κ·u_g /
                                        谱方法参考解(L=48, --refs-dir 按 (m1,m2) 匹配;
                                        无匹配时 q10 回退 TwoPunctures)三条曲线同图
    2. param_equatorial_difference.png — 赤道面 (z=0) 残差热图 u_θ − 谱方法参考解,
                                        6 个训练配置(2×3), 每面板 ±99.5 分位对称色标
    3. param_loss_history.png         — 训练损失曲线(raw 半透明 + EMA α=0.99 粗线,
                                        与 A3 v5 同格式): L2/LBC 与 L_ref 两栏
    4. param_u0_vs_m2.png             — 奇点峰高随 m2 变化: m2 奇点峰(x=-3)与
                                        m1 奇点峰(x=+3)两条曲线(峰高=邻域轴向最大值,
                                        奇点精确命中处 guide_u 有相消伪影不可用)
"""

import argparse, json, logging, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# 微软雅黑: SimHei 缺 U+2212(对数轴指数负号)与 ² 等字形, 雅黑均有
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))
from logutil import setup_logging
import physics
from spectral_reference import SpectralPunctureSolver
from parametric_train import predict, TRAIN_PARAMS, VAL_PARAMS
from parametric_eval import load_model

log = logging.getLogger("paper.A2.parametric_viz")

KAPPA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa_cache.json")
XP = (3.0, -3.0)          # 奇点位置(所有配置固定)
AXIS_RMIN = 0.05          # 轴剖面奇点邻域掩膜半径(guide_u 相消伪影仅在精确命中时出现)


def _dense_axis(r_min=AXIS_RMIN):
    """x 轴稠密采样: 均匀 3001 点 + 每奇点 ±1.2 内加密 2001 点(针尖 Δx≈0.0012)。
    模型/引导/参考三条曲线统一用这套采样, 与 A3 v5 同形式。"""
    parts = [np.linspace(-28.0, 28.0, 3001)]
    for xp in XP:
        seg = xp + np.linspace(-1.2, 1.2, 2001)
        parts.append(seg[np.abs(seg - xp) > r_min])
    return np.unique(np.round(np.concatenate(parts), 9))


def _guide_dense(kappa, m2, xs_d):
    """引导基线 κ·u_g 沿稠密轴求值(float64), 奇点邻域置 NaN。"""
    pts = np.zeros((len(xs_d), 3))
    pts[:, 0] = xs_d
    xt = torch.from_numpy(pts).double()
    ma = torch.tensor([0.5, m2], dtype=torch.float64)
    xst = torch.tensor([[XP[0], 0.0, 0.0], [XP[1], 0.0, 0.0]], dtype=torch.float64)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    St = torch.zeros((2, 3), dtype=torch.float64)
    ug = kappa * physics.guide_u(xt, ma, xst, Pt, St).numpy()
    keep = np.ones(len(xs_d), dtype=bool)
    for xp in XP:
        keep &= np.abs(xs_d - xp) > AXIS_RMIN
    ug[~keep] = np.nan
    return ug


def _load_ref_axis_and_slice(reference_path):
    """从 TwoPunctures 参考解 npz 提取: (x 轴剖面, z=0 切片散点 (x, y, u))。"""
    rd = np.load(reference_path)
    xr, ur = rd["x_ref"], rd["u_ref"].astype(np.float64)
    on_axis = (np.abs(xr[:, 1]) < 1e-6) & (np.abs(xr[:, 2]) < 1e-6)
    xa, ua = xr[on_axis, 0], ur[on_axis]
    o = np.argsort(xa)
    xa, ua = xa[o], ua[o]
    rmask = np.ones(len(xa), dtype=bool)
    for xp in XP:
        rmask &= np.abs(xa - xp) > AXIS_RMIN
    m = np.abs(xr[:, 2]) < 1e-6
    return (xa[rmask], ua[rmask]), (xr[m, 0], xr[m, 1], ur[m])


def main():
    setup_logging("A2", "parametric_viz")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "parametric_a1", "model.pt"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "parametric_a1", "figs"))
    parser.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "reference_u.npz"))
    parser.add_argument("--refs-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "refs_a2"),
                        help="A2 谱参考解目录(ref_a2_<label>.npz), 按 (m1,m2) 匹配逐参数绘制")
    parser.add_argument("--history", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "parametric_a1", "history.json"))
    parser.add_argument("--title", default="parametric_a1")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    kc = json.load(open(KAPPA_CACHE))
    kc = {k: v["kappa"] for k, v in kc.items()}

    ref_axis = None
    if args.reference and os.path.exists(args.reference):
        ref_axis, _ = _load_ref_axis_and_slice(args.reference)
        log.info(f"TwoPunctures 参考解轴上 {len(ref_axis[0])} 点(仅 q10 兜底用)")

    # ---- A2 谱参考解索引: (m1, m2) -> npz 路径, 惰性建求值器 ----
    a2_refs = {}
    if args.refs_dir and os.path.isdir(args.refs_dir):
        for fn in sorted(os.listdir(args.refs_dir)):
            if fn.endswith(".npz"):
                try:
                    raw = np.load(os.path.join(args.refs_dir, fn))["raw"]
                    a2_refs[(float(raw[0]), float(raw[1]))] = os.path.join(args.refs_dir, fn)
                except Exception:
                    pass
    log.info(f"谱参考解目录 {args.refs_dir}: {len(a2_refs)} 个配置")
    _a2_eval_cache = {}

    def _a2_ref_eval(m1, m2):
        for (a, b), p in a2_refs.items():
            if abs(a - m1) < 1e-9 and abs(b - m2) < 1e-9:
                if p not in _a2_eval_cache:
                    _a2_eval_cache[p] = SpectralPunctureSolver.from_coefficients(
                        p, device=str(device), verify=False)
                return _a2_eval_cache[p]
        return None

    # ---- 1. 逐参数 x 轴剖面(每参数一个子图: 模型/引导/参考同图) ----
    xs_d = _dense_axis()
    pts = np.zeros((len(xs_d), 3))
    pts[:, 0] = xs_d
    configs = sorted([(m1, m2, lb, False) for m1, m2, lb in TRAIN_PARAMS] +
                     [(m1, m2, lb, True) for m1, m2, lb in VAL_PARAMS],
                     key=lambda c: c[1])   # 按 q(=m2/m1, m1 固定)从小到大排列
    fig, axes = plt.subplots(2, 5, figsize=(25, 9), sharex=True)
    for i, (ax, (m1, m2, lb, is_val)) in enumerate(zip(axes.ravel(), configs)):
        u_model = predict(model, pts, m1, m2, kc[lb], device).astype(float)
        keep = np.ones(len(xs_d), dtype=bool)
        for xp in XP:
            keep &= np.abs(xs_d - xp) > AXIS_RMIN
        u_model[~keep] = np.nan
        u_guide = _guide_dense(kc[lb], m2, xs_d)
        ax.plot(xs_d, u_model, "-", lw=0.9, color="C0", label="模型 u_θ")
        ax.plot(xs_d, u_guide, "--", lw=0.6, color="C2", label="引导基线 κ·u_g")
        ev = _a2_ref_eval(m1, m2)
        if ev is not None:
            u_r = ev.evaluate(pts, chunk=16384, dtype=torch.float64)
            u_r = np.where(keep, u_r, np.nan)
            ax.plot(xs_d, u_r, "-", lw=0.9, color="C3", alpha=0.9,
                    label="谱方法参考解")
        elif lb == "q10" and ref_axis is not None:
            ax.plot(ref_axis[0], ref_axis[1], "-", lw=0.9, color="C3", alpha=0.9,
                    label="TwoPunctures 参考")
        ax.set_title(f"{lb} (m2={m2:g}{' | 零样本' if is_val else ''})", fontsize=10)
        ax.set_xlabel("x")
        if i % 5 == 0:
            ax.set_ylabel("u")
        if i in (0, 5):
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-28, 28)
    fig.suptitle(f"{args.title} x 轴剖面 (y=z=0): 模型 vs 引导基线 vs 谱方法参考解(L=48)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out, "param_axis_profiles.png"), dpi=800)
    plt.close(fig)
    log.info("[图] param_axis_profiles.png (10 参数面板, 模型/引导/参考同图)")

    # ---- 2. 赤道面 (z=0) 残差热图: u_θ − 谱方法参考解, 6 个训练配置 ----
    n = 501
    g = np.linspace(-30.0, 30.0, n)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    targets = np.stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)], axis=1)
    near_flat = np.zeros(GX.size, dtype=bool)
    for xp in XP:
        near_flat |= np.hypot(targets[:, 0] - xp, targets[:, 1]) < 0.15
    fig, axes2 = plt.subplots(2, 3, figsize=(16.5, 10.2), sharex=True, sharey=True)
    for ax2, (m1, m2, lb) in zip(axes2.ravel(), TRAIN_PARAMS):
        ev = _a2_ref_eval(m1, m2)
        if ev is None:
            ax2.text(0.5, 0.5, f"{lb} 无谱参考解", ha="center", va="center",
                     transform=ax2.transAxes)
            continue
        uref = ev.evaluate(targets, chunk=131072, dtype=torch.float32).astype(float)
        umod = predict(model, targets.astype(np.float32), m1, m2, kc[lb], device).astype(float)
        D = umod - uref
        D[near_flat] = np.nan
        D = D.reshape(n, n)
        v = np.nanpercentile(np.abs(D), 99.5)
        im = ax2.imshow(D.T, origin="lower", extent=[-30, 30, -30, 30],
                        cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax2.plot([XP[0], XP[1]], [0.0, 0.0], "k^", ms=5)
        ax2.set_title(f"{lb} (m2={m2:g})", fontsize=11)
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        fig.colorbar(im, ax=ax2, shrink=0.85)
        log.info(f"  赤道残差 {lb}: ±99.5 分位 = ±{v:.2e}")
    fig.suptitle(f"{args.title} 赤道面 (z=0): u_θ - 谱方法参考解 残差 (训练配置, 色标对称 ±99.5 分位)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out, "param_equatorial_difference.png"), dpi=300)
    plt.close(fig)
    log.info("[图] param_equatorial_difference.png (6 训练配置, u_θ-谱参考残差)")

    # ---- 3. 训练损失 (raw 半透明 + EMA 粗线, 与 A3 v5 同格式) ----
    if os.path.exists(args.history):
        hist = json.load(open(args.history))

        def _ema(vals, alpha=0.99):
            out = [vals[0]]
            for v in vals[1:]:
                out.append(alpha * out[-1] + (1 - alpha) * v)
            return np.array(out)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        steps = np.arange(1, len(hist["L2"]) + 1)
        ax = axes[0]
        ax.semilogy(steps, hist["L2"], alpha=0.3, label="L2 (raw)", color="C0")
        ax.semilogy(steps, hist["LBC"], alpha=0.3, label="LBC (raw)", color="C1")
        ax.semilogy(steps, _ema(hist["L2"]), label="L2 (EMA)", color="C0", lw=2)
        ax.semilogy(steps, _ema(hist["LBC"]), label="LBC (EMA)", color="C1", lw=2)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.set_title("PDE & Robin Loss")
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        lref = np.array(hist["L_ref"])
        mask = lref > 0
        if mask.any():
            ax.semilogy(steps[mask], lref[mask], alpha=0.3, label="L_ref (raw)", color="C2")
            ax.semilogy(steps[mask], _ema(lref[mask]), label="L_ref (EMA)", color="C2", lw=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("L_ref")
        ax.set_title("Reference Loss (base case only)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "param_loss_history.png"), dpi=150)
        plt.close(fig)
        log.info("[图] param_loss_history.png (raw + EMA)")

    # ---- 4. 奇点峰高随 m2 变化 ----
    # 注(build_bbh 确认): m1 位于 x=+3, m2 位于 x=-3。奇点精确命中处 guide_u 有
    # 相消伪影(探针实测 u~1e29), 故峰高定义为奇点邻域 |x-xp|∈[0.05,0.35] 的轴向
    # 最大值(与剖面图掩膜口径一致)。
    m2s = np.linspace(0.25, 1.0, 16)
    kappas = np.interp(m2s, [0.25, 0.35, 0.5, 0.65, 0.8, 1.0],
                       [kc["q05"], kc["q07"], kc["q10"], kc["q13"], kc["q16"], kc["q20"]])
    offs = np.concatenate([np.linspace(0.05, 0.35, 31), -np.linspace(0.05, 0.35, 31)])

    def _peak(xp, m2, k):
        pts = np.stack([xp + offs, np.zeros_like(offs), np.zeros_like(offs)], axis=1)
        return float(np.max(predict(model, pts.astype(np.float32), 0.5, m2, k, device)))

    pk_m2 = [_peak(XP[1], m2, k) for m2, k in zip(m2s, kappas)]   # x=-3: m2 奇点峰
    pk_m1 = [_peak(XP[0], m2, k) for m2, k in zip(m2s, kappas)]   # x=+3: m1 奇点峰
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(m2s, pk_m2, "o-", lw=1.6, color="C0", label="m2 奇点峰 (x=-3)")
    ax.plot(m2s, pk_m1, "s--", lw=1.4, color="C1", label="m1 奇点峰 (x=+3, 即 u(3,0,0) 邻域)")
    ax.set_title(f"{args.title} 奇点峰高随 m2 变化\n(峰高 = |x-xp|∈[0.05,0.35] 轴向最大值, 避开奇点精确命中伪影)")
    ax.set_xlabel("m2 (m1=0.5)"); ax.set_ylabel("u 峰值")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "param_u0_vs_m2.png"), dpi=150)
    plt.close(fig)
    log.info("[图] param_u0_vs_m2.png (m1/m2 奇点峰高 vs m2)")

    log.info("可视化完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

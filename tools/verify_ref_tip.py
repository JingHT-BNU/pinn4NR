"""
tools/verify_ref_tip.py —— 复核谱方法参考解的针尖(奇点近场)精度
================================================================

背景: 2026-08-29 发现 train_0059/0164 模型针尖峰值比谱方法参考解高 2.6~3.5×。
在归因于模型之前, 必须排除"参考解本身在针尖欠分辨"。两项独立检查:

  1) 分辨率自收敛: 批量默认 (N_r=384, L=32) vs 高分辨率 (N_r=512, L=48),
     同一稠密轴采样(奇点邻域加密, r≥0.015)逐点对比。若两分辨率的针尖峰值
     与分区 L2 一致, 则 L=32 参考解在针尖已收敛。
  2) 独立 PDE 残差: 在奇点邻域立方体内对谱解任意点求值, 用 2 阶中心差分算
     Δu, 与 (1/8)ψ^{-7}K̄K̄ 比较 —— 完全不经过求解器的 Chebyshev/球谐离散。
     判读: FD 步长减半残差应 ~4× 下降(说明残差是 FD 误差, 解干净);
     若残差不随 h 下降、且随谱分辨率提高而显著减小, 则低分辨率解未收敛。

用法:
    python paper/tools/verify_ref_tip.py --labels train_0059,train_0164

输出:
    日志(paper/logs/tools/) + paper/tools/refs/verify_tip/<label>_axis.npz
    (x/u32/u48 稠密轴) + paper/tools/refs/verify_tip/summary.json
"""

import argparse, json, logging, os, sys, time
import numpy as np
import torch

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.join(PAPER, "tools"))
sys.path.insert(0, os.path.join(PAPER, "A3_multi_param"))
from logutil import setup_logging  # noqa: E402
import physics  # noqa: E402
from multi_param_model import build_bbh_from_params  # noqa: E402
from spectral_reference import SpectralPunctureSolver  # noqa: E402

log = logging.getLogger("paper.tools.verify_ref_tip")

SOL32 = dict(N_r=384, L=32, N_th=56, N_ph=96)     # 批量默认分辨率
SOL48 = dict(N_r=512, L=48, N_th=72, N_ph=144)    # 高分辨率


def dense_axis(xs_punc, r_min=0.015):
    """x 轴稠密采样: 均匀 3001 点 + 每奇点 ±1.2 内 2001 点(剔除 r<r_min)。"""
    parts = [np.linspace(-28.0, 28.0, 3001)]
    for xp in xs_punc:
        seg = xp + np.linspace(-1.2, 1.2, 2001)
        parts.append(seg[np.abs(seg - xp) > r_min])
    return np.unique(np.round(np.concatenate(parts), 9))


def eval_axis(solver, xs_line):
    pts = np.zeros((len(xs_line), 3))
    pts[:, 0] = xs_line
    return solver.evaluate(pts, chunk=8192, dtype=torch.float64)


def zone_compare(xs_line, u32, u48, xs_punc):
    """两分辨率的分区相对 L2 差 (相对 L32 的 RMS)。"""
    r_p = np.min(np.abs(xs_line[:, None] - np.asarray(xs_punc)[None, :]), axis=1)
    rows = []
    for name, lo, hi in [("tip  r_p[0.015,0.5)", 0.015, 0.5),
                         ("mid  r_p[0.5,3)", 0.5, 3.0),
                         ("bulk r_p>=3", 3.0, 1e9)]:
        m = (r_p >= lo) & (r_p < hi)
        rel = float(np.sqrt(np.sum((u48[m] - u32[m]) ** 2) / np.sum(u32[m] ** 2)))
        rows.append((name, rel, int(m.sum())))
    return rows


def tip_peaks(xs_line, u, xs_punc):
    """每奇点 |x-xp|<0.5 内的 max|u| (针尖峰值)。

    注意掩膜必须按奇点各自限定 (|x-xp|<0.5), 不能用 min 距离 r_p<0.5 ——
    后者会让"x- 区"包含整个 x+ 邻域, 两个区给出相同的 max (2026-08-29 教训)。"""
    out = []
    for xp in xs_punc:
        d = np.abs(xs_line - xp)
        m = (d < 0.5) & (d > 0.015)
        out.append((float(xp), float(np.abs(u[m]).max()) if m.any() else float("nan")))
    return out


def fd_residual(solver, xp, masses, xs, Ps, Ss, h=0.01, half=0.35, r_guard=0.03):
    """奇点邻域立方体上 Δu(2阶FD) + (1/8)ψ^{-7}K̄K̄ 的归一化残差。

    盒子沿 x 以 xp 为中心, y/z 以 0 为中心(奇点在 x 轴上);
    ψ = ψ_sing + u(谱解), K̄K̄ 用与训练/求解器一致的 physics 实现。
    残差以立方体内源项峰值归一。"""
    n = int(round(2 * half / h)) + 1
    gx = xp + np.linspace(-half, half, n)
    gt = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(gx, gt, gt, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    rp = np.linalg.norm(pts - np.array([xp, 0.0, 0.0]), axis=1)
    u = solver.evaluate(pts, chunk=8192, dtype=torch.float64).reshape(n, n, n)
    lap = np.full((n, n, n), np.nan)
    lap[1:-1, 1:-1, 1:-1] = (
        u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:] - 6.0 * u[1:-1, 1:-1, 1:-1]) / h ** 2
    with torch.no_grad():
        xt = torch.from_numpy(pts).double()
        mt = torch.tensor(masses, dtype=torch.float64)
        xst = torch.tensor(xs, dtype=torch.float64)
        Pt = torch.tensor(Ps, dtype=torch.float64)
        St = torch.tensor(Ss, dtype=torch.float64)
        psi = physics.psi_sing(xt, mt, xst) + torch.from_numpy(u.reshape(-1)).double()
        kk = physics.bowen_york_KK(xt, mt, xst, Pt, St)
        src = (kk / (8.0 * psi ** 7)).numpy().reshape(n, n, n)
    valid = np.isfinite(lap) & (rp.reshape(n, n, n) > r_guard)
    R = np.abs((lap + src)[valid])
    scale = float(np.abs(src[valid]).max())
    return {"median": float(np.median(R) / scale),
            "p95": float(np.percentile(R, 95) / scale),
            "max": float(R.max() / scale),
            "n": int(valid.sum()), "src_scale": scale}


def verify_label(label, raw, device, out_dir, fd_only=False):
    masses, xs, Ps, Ss = build_bbh_from_params(raw)
    xs_punc = [float(v) for v in xs[:, 0]]
    results = {"raw": raw.tolist(), "xs_punc": xs_punc}
    sols = {}
    for tag, kw in [("L32", SOL32), ("L48", SOL48)]:
        cpath = os.path.join(out_dir, f"{label}_{tag}_coeffs.npz")
        if fd_only:
            log.info(f"[{label}] === {tag}: 从谱系数加载 {os.path.basename(cpath)} ===")
            sol = SpectralPunctureSolver.from_coefficients(cpath, device=device,
                                                           verify=False)
            results[tag] = {"iters": -1}
        else:
            t0 = time.time()
            log.info(f"[{label}] === 构建+求解 {tag}: N_r={kw['N_r']}, L={kw['L']} ===")
            sol = SpectralPunctureSolver(raw, device=device, **kw)
            it, stats = sol.solve(verbose=True)
            log.info(f"[{label}] {tag}: 迭代 {it}, res_max={stats['res_max']:.2e}, "
                     f"rms={stats['res_rms']:.2e}, modal_max={stats['res_modal_max']:.2e} "
                     f"({time.time() - t0:.0f}s)")
            results[tag] = {"iters": int(it),
                            **{k: float(v) for k, v in stats.items()}}
            os.makedirs(out_dir, exist_ok=True)
            np.savez(cpath, radial_coeffs=sol.export_coefficients(),
                     raw=raw, xc=sol.xc.astype(np.float64),
                     meta=json.dumps({"N_r": kw["N_r"], "L": kw["L"],
                                      "R0": 15.0, "label": label, "tag": tag}))
        sols[tag] = sol

    xs_line = dense_axis(xs_punc)
    t0 = time.time()
    u32 = eval_axis(sols["L32"], xs_line)
    u48 = eval_axis(sols["L48"], xs_line)
    log.info(f"[{label}] 轴稠密求值 {len(xs_line)} 点 ({time.time() - t0:.0f}s)")

    for name, rel, npts in zone_compare(xs_line, u32, u48, xs_punc):
        log.info(f"[{label}] L48 vs L32 {name}: relL2={rel:.3e} ({npts} pts)")
    results["zones"] = [{"zone": n, "relL2": r, "npts": k}
                        for n, r, k in zone_compare(xs_line, u32, u48, xs_punc)]

    for tag, u in [("L32", u32), ("L48", u48)]:
        peaks = tip_peaks(xs_line, u, xs_punc)
        results[tag]["tip_peaks"] = peaks
        log.info(f"[{label}] {tag} 针尖峰值: "
                 + "  ".join(f"x={xp:+.3f}: {pk:.6f}" for xp, pk in peaks))

    dom = int(np.argmax([pk for _, pk in results["L32"]["tip_peaks"]]))
    xp_dom = xs_punc[dom]
    log.info(f"[{label}] 主导奇点 x={xp_dom:+.4f}")
    for tag in ("L32", "L48"):
        for h, guard in [(0.02, 0.05), (0.01, 0.03)]:
            r = fd_residual(sols[tag], xp_dom, masses, xs, Ps, Ss, h=h, r_guard=guard)
            results[tag][f"fd_h{h}"] = r
            log.info(f"[{label}] {tag} FD(h={h}) |Δu+src|/src_peak: "
                     f"median={r['median']:.2e} p95={r['p95']:.2e} "
                     f"max={r['max']:.2e} ({r['n']} pts, src_peak={r['src_scale']:.3e})")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{label}_axis.npz"),
             x=xs_line, u32=u32, u48=u48, raw=raw)
    return results


def main():
    setup_logging("tools", "verify_ref_tip")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="train_0059,train_0164")
    ap.add_argument("--config", default=os.path.join(
        PAPER, "runs", "multi_param_a2", "configs.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fd-only", action="store_true",
                   help="跳过求解, 从 verify_tip/<label>_<tag>_coeffs.npz 加载谱系数"
                        "重做轴对比与 FD 残差(先跑过完整模式才有系数)")
    ap.add_argument("--out-dir", default=os.path.join(
        PAPER, "tools", "refs", "verify_tip"))
    args = ap.parse_args()

    info = json.load(open(args.config))
    raws = {c["label"]: np.array(c["raw"], dtype=np.float64) for c in info["train"]}
    summary = {}
    for label in args.labels.split(","):
        label = label.strip()
        if not label:
            continue
        if label not in raws:
            log.warning(f"未知配置 {label}, 跳过")
            continue
        summary[label] = verify_label(label, raws[label], args.device,
                                      args.out_dir, fd_only=args.fd_only)
        # 每个配置完成即落盘, 中断也有部分结果
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
    log.info(f"全部完成: {args.out_dir}/summary.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

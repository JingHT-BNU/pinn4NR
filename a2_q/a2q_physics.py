# --- path bootstrap (added during repo restructure: shared code lives in core/ and tools/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while True:
    if _os.path.isdir(_os.path.join(_ROOT, "core")):
        break
    _parent = _os.path.dirname(_ROOT)
    if _parent == _ROOT:
        break
    _ROOT = _parent
for _sub in ("core", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
"""
a2q_physics.py —— A2 参数化模型的**物理型指标**评估
====================================================

与参考解无关的独立物理检验，用于回答"模型学到的是物理，还是对参考解的过拟合"：

  1. **Hamilton 约束残差场**
         R = Δu_θ + (1/8)·ψ^{-7}·K̄_ij K̄^ij  ,  ψ = ψ_sing + u_θ
     模型训练只用了参考解监督（纯 L_ref），**从未见过 PDE 残差**。
     因此 R 是否足够小，是对"模型是否学出物理一致解"的独立检验。
     归一化基准取源项 S = (1/8)ψ^{-7}K̄K̄ 的 RMS：
         相对残差 = ||R||_rms / ||S||_rms
     即"本该被 Δu 抵消掉的源项，还剩多少没抵消"。

  2. **ADM 质量与远场多极展开**
     在半径 r 的球面上把 ψ-1 展开为
         ψ - 1 = A/r + d·n̂/r² + n̂ᵀQn̂/r³ + …
     单极系数给出 ADM 质量  M_ADM = 2A（由 M_ADM = -(1/2π)∮∂_rψ dS 推得）。
     同时报告偶极 d 与四极 Q。
     物理论证：Σm = m1+m2 只是裸质量之和，u 的 1/r 尾巴还会贡献
     (引力场+动能) 能量，故应有 M_ADM ≥ Σm；模型是否与谱参考解一致是关键。

用法：
    python a2_q/a2q_physics.py --run data/runs/a2/a2q_v6a
    python a2_q/a2q_physics.py --configs q10,q15,q50 --n-res 6000
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

import physics
from logutil import setup_logging

log = logging.getLogger("paper.A2.a2q_physics")

XS = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])   # masses[0]=m1 at +3, masses[1]=m2 at -3
PS = np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
SS = np.zeros((2, 3))

DEFAULT_CONFIGS = "q05,q10,q20,q15,q25,q50,q74,q86,q100"


# ----------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------

def _nearest_d(pts):
    """到两个奇点的最小距离 d = min(r1, r2)。"""
    r1 = np.linalg.norm(pts - XS[0], axis=1)
    r2 = np.linalg.norm(pts - XS[1], axis=1)
    return np.minimum(r1, r2), r1, r2


def sample_stratified(n_total, rcut=0.05, r_far=28.0, seed=0):
    """按物理区域分层采样（与训练/评估的关注区域一致）。

    区域划分（d = 到最近奇点的距离，r0 = 到原点距离）：
      near  : d < 0.5           峰区（针尖附近，残差最难压）
      mid   : 0.5 <= d < 2      中场
      outer : d >= 2 且 r0 < 10 外场
      far   : 10 <= r0 < r_far  远场
    """
    rng = np.random.default_rng(seed)
    quota = {"near": n_total // 4, "mid": n_total // 4,
             "outer": n_total // 4, "far": n_total - 3 * (n_total // 4)}
    out = {k: [] for k in quota}
    got = {k: 0 for k in quota}
    # 峰区在 [-30,30]³ 里体积极小，直接在奇点周围的小立方体内采样以提高命中率
    boxes = [np.array([3.0, 0, 0]), np.array([-3.0, 0, 0])]
    while any(got[k] < quota[k] for k in quota):
        need = {k: quota[k] - got[k] for k in quota}
        # 远场需要大盒子；峰区用小盒子
        if need["far"] > 0 or need["outer"] > 0:
            pts = rng.uniform(-r_far, r_far, size=(200000, 3))
        else:
            c = boxes[rng.integers(0, 2)]
            pts = c + rng.uniform(-2.5, 2.5, size=(200000, 3))
        d, r1, r2 = _nearest_d(pts)
        r0 = np.linalg.norm(pts, axis=1)
        keep = (r1 > rcut) & (r2 > rcut)
        for k, m in (("near", d < 0.5), ("mid", (d >= 0.5) & (d < 2.0)),
                     ("outer", (d >= 2.0) & (r0 < 10.0)),
                     ("far", (r0 >= 10.0) & (r0 < r_far))):
            sel = keep & m
            n = min(int(sel.sum()), need[k])
            if n > 0:
                idx = np.where(sel)[0][:n]
                out[k].append(pts[idx])
                got[k] += n
    res = {k: np.concatenate(v)[:quota[k]] for k, v in out.items() if v}
    return res


def sample_region(name, n, rcut=0.05, r_far=28.0, seed=0):
    """按名字采样单个物理区域（用于有限差分残差对比）。"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        if name in ("far", "outer"):
            p = rng.uniform(-r_far, r_far, size=(60000, 3))
        else:
            c = XS[0] if rng.random() < 0.5 else XS[1]
            p = c + rng.uniform(-2.5, 2.5, size=(60000, 3))
        r1 = np.linalg.norm(p - XS[0], axis=1)
        r2 = np.linalg.norm(p - XS[1], axis=1)
        d = np.minimum(r1, r2)
        r0 = np.linalg.norm(p, axis=1)
        keep = (r1 > rcut) & (r2 > rcut)
        if name == "near":
            sel = keep & (d < 0.5)
        elif name == "mid":
            sel = keep & (d >= 0.5) & (d < 2.0)
        elif name == "outer":
            sel = keep & (d >= 2.0) & (r0 < 10.0)
        else:
            sel = keep & (r0 >= 10.0) & (r0 < r_far)
        out.append(p[sel])
    return np.concatenate(out)[:n]


def fibonacci_sphere(n, seed=0):
    """Fibonacci 球面均匀方向。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    ga = np.pi * (1.0 + 5.0 ** 0.5)
    th = ga * i
    off = (2.0 * np.pi * 0.618033988749895 * (seed % 97)) / 97.0
    return np.stack([np.sin(phi) * np.cos(th + off),
                     np.sin(phi) * np.sin(th + off),
                     np.cos(phi)], axis=1)


# ----------------------------------------------------------------------
# 多极展开
# ----------------------------------------------------------------------

def multipole_basis(n_hat):
    """返回 (N,9) 基函数 [1, n_x,n_y,n_z, Q1..Q5]（不含 1/r^l 因子）。"""
    x, y, z = n_hat[:, 0], n_hat[:, 1], n_hat[:, 2]
    quad = np.stack([
        0.5 * (3 * z * z - 1.0),          # l=2,m=0
        3 * x * z,                        #  ~ xz
        3 * y * z,                        #  ~ yz
        3 * x * y,                        #  ~ xy
        1.5 * (x * x - y * y),            #  ~ x²-y²
    ], axis=1)
    return np.concatenate([np.ones((len(n_hat), 1)), n_hat, quad], axis=1)


def fit_multipoles(pts, vals):
    """在球壳上做线性最小二乘：
        ψ-1 = A/r + d·n̂/r² + q·Q(n̂)/r³
       返回 dict(A, dipole(3), quad(5))。"""
    r = np.linalg.norm(pts, axis=1)
    n_hat = pts / r[:, None]
    B = multipole_basis(n_hat)
    A_mat = np.stack([B[:, 0] / r,
                      B[:, 1] / r ** 2, B[:, 2] / r ** 2, B[:, 3] / r ** 2,
                      B[:, 4] / r ** 3, B[:, 5] / r ** 3, B[:, 6] / r ** 3,
                      B[:, 7] / r ** 3, B[:, 8] / r ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(A_mat, vals, rcond=None)
    resid = vals - A_mat @ coef
    return {"A": float(coef[0]),
            "dipole": coef[1:4].tolist(),
            "quad": coef[4:9].tolist(),
            "fit_rms": float(np.sqrt(np.mean(resid ** 2))) / float(np.mean(1.0 / r))}


# ----------------------------------------------------------------------
# Hamilton 残差
# ----------------------------------------------------------------------

def hamilton_residual(model, pts, cinfo, device, chunk=2048):
    """返回 (R, S, d, r0)，R 为 Hamilton 约束残差，S 为源项。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    m1 = float(cinfo.get("m1", 0.5))
    m2 = float(cinfo["m2"])
    masses = torch.tensor([m1, m2], dtype=torch.float64, device=device)
    xs = torch.tensor(XS, dtype=torch.float64, device=device)
    Ps = torch.tensor(PS, dtype=torch.float64, device=device)
    Ss = torch.tensor(SS, dtype=torch.float64, device=device)

    Rs, Ss_, Ds, R0s = [], [], [], []
    d_all, r1_all, r2_all = _nearest_d(pts)
    r0_all = np.linalg.norm(pts, axis=1)
    for i in range(0, len(pts), chunk):
        xb = torch.tensor(pts[i:i + chunk], dtype=torch.float64,
                          device=device, requires_grad=True)
        p = torch.tensor([[np.log10(float(cinfo["q"])), m2 / 5.0]],
                         dtype=torch.float64, device=device)
        u = model(xb, masses, xs, Ps, Ss, p,
                  float(cinfo["kappa"]), float(cinfo["wmin"]),
                  float(cinfo["wmax"]), float(cinfo["sq"]))
        psi_s = physics.psi_sing(xb, masses, xs)
        kk = physics.bowen_york_KK(xb, masses, xs, Ps, Ss)
        psi = torch.clamp(psi_s + u, min=1e-4)
        S = (1.0 / 8.0) * kk / psi ** 7                      # 源项
        R = physics.pde_residual(u, xb, psi_s, kk)           # Δu + S
        Rs.append(R.detach().cpu().numpy().astype(np.float64))
        Ss_.append(S.detach().cpu().numpy().astype(np.float64))
        Ds.append(d_all[i:i + chunk])
        R0s.append(r0_all[i:i + chunk])
    return (np.concatenate(Rs), np.concatenate(Ss_),
            np.concatenate(Ds), np.concatenate(R0s))


def fd_laplacian(u_fn, pts, h):
    """中心差分拉普拉斯 Δu = Σ_i [u(x+he_i) - 2u(x) + u(x-he_i)]/h²。

    用于对谱参考解求残差（谱求解器不支持 autograd）。
    同一点集、同一 h 也对模型求一次，用来量化差分噪声，保证对比公平。
    """
    lap = np.zeros(len(pts))
    u0 = u_fn(pts)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        lap += (u_fn(pts + e) - 2.0 * u0 + u_fn(pts - e)) / h ** 2
    return lap, u0


def source_term(pts, m1, m2, u):
    """S = (1/8)·ψ^{-7}·K̄K̄  (numpy, double)。"""
    r1 = np.maximum(np.linalg.norm(pts - XS[0], axis=1), 1e-12)
    r2 = np.maximum(np.linalg.norm(pts - XS[1], axis=1), 1e-12)
    psi_s = 1.0 + m1 / (2.0 * r1) + m2 / (2.0 * r2)
    # K̄_ij K̄^ij 用 torch 版本算，避免重复实现 Bowen-York
    xt = torch.tensor(pts, dtype=torch.float64)
    masses = torch.tensor([m1, m2], dtype=torch.float64)
    xs = torch.tensor(XS, dtype=torch.float64)
    Ps = torch.tensor(PS, dtype=torch.float64)
    Ss = torch.zeros((2, 3), dtype=torch.float64)
    with torch.no_grad():
        kk = physics.bowen_york_KK(xt, masses, xs, Ps, Ss).numpy()
    psi = np.maximum(psi_s + u, 1e-4)
    return (1.0 / 8.0) * kk / psi ** 7


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(_ROOT, "data", "runs", "a2",
                                                  "a2q_v6a"))
    ap.add_argument("--configs", default=DEFAULT_CONFIGS,
                    help="逗号分隔的标签; --configs all 表示全部")
    ap.add_argument("--n-res", type=int, default=12000, help="残差采样点数")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--radii", default="10,14,18,22,26",
                    help="多极展开的球壳半径")
    ap.add_argument("--n-dir", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-ref", action="store_true", help="跳过谱参考解对照")
    ap.add_argument("--fd-ref", action="store_true",
                    help="用同一差分 stencil 计算模型与谱参考解的残差(公平对比)")
    ap.add_argument("--n-fd", type=int, default=200,
                    help="每区域有限差分采样点数")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run
    log_f = setup_logging("A2", "a2q_physics")
    log.info("run = %s", run_dir)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "auto" else "cpu")
    log.info("device = %s", device)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import a2q_model as A2
    model, ck = A2.load_run(run_dir, device)
    model.double()
    model.eval()
    meta = ck["meta"]
    train_labels = set(ck.get("train_labels", meta.keys()))
    heldout = set(ck.get("heldout_labels", []))

    labels = sorted(meta.keys(), key=lambda lb: meta[lb]["q"])
    if args.configs and args.configs != "all":
        want = set(args.configs.split(","))
        labels = [lb for lb in labels if lb in want]
    log.info("评估 %d 个配置: %s", len(labels), ",".join(labels))

    # 谱参考解求解器（用于 ADM 质量对照）
    solvers = {}
    if not args.no_ref:
        sys.path.insert(0, os.path.join(_ROOT, "tools"))
        import spectral_reference as sr
        for lb in labels:
            src = None
            for d in ("a2v2", "a2"):
                for pat in (f"ref_{lb}.npz", f"ref_a2_{lb}.npz"):
                    p = os.path.join(_ROOT, "data", "refs", d, pat)
                    if os.path.exists(p):
                        src = p
                        break
                if src:
                    break
            if src is None:
                log.warning("%s: 无谱参考解,跳过对照", lb)
                continue
            try:
                solvers[lb] = sr.SpectralPunctureSolver.from_coefficients(
                    src, device=str(device), verify=False)
            except Exception as e:            # noqa: BLE001
                log.warning("%s: 谱求解器加载失败 %s", lb, e)

    radii = [float(x) for x in args.radii.split(",")]
    dirs = fibonacci_sphere(args.n_dir)

    results = {}
    for lb in labels:
        m = meta[lb]
        cinfo = {k: m[k] for k in ("q", "m1", "m2", "kappa", "sq",
                                   "wmin", "wmax")}
        m1 = float(cinfo.get("m1", 0.5))
        m2 = float(m["m2"])
        t0 = time.time()
        res = {"q": float(m["q"]),
               "group": ("heldout" if lb in heldout else
                         "train" if lb in train_labels else "zero_shot")}

        # ---------- 1. Hamilton 残差 ----------
        regions = sample_stratified(args.n_res, seed=abs(hash(lb)) % (2 ** 31))
        allpts = np.concatenate(list(regions.values()), 0)
        R, S, d, r0 = hamilton_residual(model, allpts, cinfo, device,
                                        chunk=args.chunk)
        s_rms = float(np.sqrt(np.mean(S ** 2)))
        res["resid"] = {
            "n_pts": int(len(R)),
            "S_rms": s_rms,
            "R_rms": float(np.sqrt(np.mean(R ** 2))),
            "R_rel_rms": float(np.sqrt(np.mean(R ** 2)) / max(s_rms, 1e-300)),
            "R_rel_median": float(np.median(np.abs(R)) / max(s_rms, 1e-300)),
            "R_rel_p95": float(np.percentile(np.abs(R), 95) / max(s_rms, 1e-300)),
            "R_max_rel": float(np.max(np.abs(R)) / max(s_rms, 1e-300)),
        }
        # 分区（用与采样一致的判据）
        for name, mask in (("near", d < 0.5),
                           ("mid", (d >= 0.5) & (d < 2.0)),
                           ("outer", (d >= 2.0) & (r0 < 10.0)),
                           ("far", r0 >= 10.0)):
            if mask.sum() > 10:
                sr_ = float(np.sqrt(np.mean(S[mask] ** 2)))
                res["resid"][name] = {
                    "n": int(mask.sum()),
                    "R_rel_rms": float(np.sqrt(np.mean(R[mask] ** 2))
                                       / max(sr_, 1e-300))}
        res["_R_hist"] = np.log10(np.abs(R) / max(s_rms, 1e-300))

        # ---------- 2. ADM 质量与多极 ----------
        def psi_of(u_fn, pts):
            xs_t = torch.tensor(XS, dtype=torch.float64)
            ps_s = 1.0 + 0.5 * (m1 / np.maximum(np.linalg.norm(
                pts - XS[0], axis=1), 1e-12)) + 0.5 * (m2 / np.maximum(
                np.linalg.norm(pts - XS[1], axis=1), 1e-12))
            return ps_s + u_fn(pts)

        def u_model(pts):
            out = []
            with torch.no_grad():
                masses = torch.tensor([m1, m2], dtype=torch.float64,
                                      device=device)
                xs_t = torch.tensor(XS, dtype=torch.float64, device=device)
                Ps_t = torch.tensor(PS, dtype=torch.float64, device=device)
                Ss_t = torch.tensor(SS, dtype=torch.float64, device=device)
                p = torch.tensor([[np.log10(float(m["q"])), m2 / 5.0]],
                                 dtype=torch.float64, device=device)
                for i in range(0, len(pts), 65536):
                    xb = torch.tensor(pts[i:i + 65536], dtype=torch.float64,
                                      device=device)
                    out.append(model(xb, masses, xs_t, Ps_t, Ss_t, p,
                                     float(m["kappa"]), float(m["wmin"]),
                                     float(m["wmax"]),
                                     float(m["sq"])).cpu().numpy())
            return np.concatenate(out)

        shells = {}
        for r in radii:
            pts = dirs * r
            pm = psi_of(u_model, pts)
            fm = fit_multipoles(pts, pm - 1.0)
            entry = {"M_ADM": 2.0 * fm["A"],
                     "dipole": fm["dipole"],
                     "dipole_abs": float(np.linalg.norm(fm["dipole"])),
                     "quad_abs": float(np.linalg.norm(fm["quad"])),
                     "fit_rms_rel": fm["fit_rms"]}
            if lb in solvers:
                pr = psi_of(lambda z: solvers[lb].evaluate(
                    z, chunk=32768, dtype=torch.float64).astype(np.float64),
                    pts)
                fr = fit_multipoles(pts, pr - 1.0)
                entry["M_ADM_ref"] = 2.0 * fr["A"]
                entry["dipole_ref"] = fr["dipole"]
                entry["dipole_abs_ref"] = float(np.linalg.norm(fr["dipole"]))
            shells[f"r{r:g}"] = entry
        res["shells"] = shells
        r_last = f"r{radii[-1]:g}"
        res["M_ADM"] = shells[r_last]["M_ADM"]
        res["M_ADM_ref"] = shells[r_last].get("M_ADM_ref")
        res["sum_m"] = float(m1 + m2)
        # 解析偶极（仅由 ψ_sing 的远场展开给出，不含 u 的贡献）
        res["dipole_analytic"] = (0.5 * (m1 * XS[0] + m2 * XS[1])).tolist()
        res["dipole_abs_analytic"] = float(np.linalg.norm(
            res["dipole_analytic"]))
        res["dipole"] = shells[r_last]["dipole"]
        res["dipole_abs"] = shells[r_last]["dipole_abs"]
        if "dipole_abs_ref" in shells[r_last]:
            res["dipole_abs_ref"] = shells[r_last]["dipole_abs_ref"]

        # ---------- 3. 有限差分残差：模型 vs 谱参考解（同一 stencil 公平对比）----------
        if args.fd_ref and lb in solvers:
            fd = {}
            for reg, h in (("near", 0.02), ("mid", 0.05),
                           ("outer", 0.05), ("far", 0.15)):
                pts = sample_region(reg, args.n_fd, seed=abs(hash(lb + reg))
                                    % (2 ** 31))
                lap_m, u_m = fd_laplacian(u_model, pts, h)
                S_m = source_term(pts, m1, m2, u_m)
                lap_r, u_r = fd_laplacian(
                    lambda z: solvers[lb].evaluate(
                        z, chunk=32768, dtype=torch.float64).astype(np.float64),
                    pts, h)
                S_r = source_term(pts, m1, m2, u_r)
                s = float(np.sqrt(np.mean(S_m ** 2)))
                fd[reg] = {
                    "h": h, "n": int(len(pts)), "S_rms": s,
                    "model": float(np.sqrt(np.mean((lap_m + S_m) ** 2))
                                   / max(s, 1e-300)),
                    "ref": float(np.sqrt(np.mean((lap_r + S_r) ** 2))
                                 / max(s, 1e-300)),
                    "lap_model_rms": float(np.sqrt(np.mean(lap_m ** 2))),
                    "lap_ref_rms": float(np.sqrt(np.mean(lap_r ** 2))),
                }
            res["fd"] = fd
        res["sec"] = round(time.time() - t0, 1)
        results[lb] = res
        log.info("%-6s q=%-7.3g  R_rel=%.3e  M_ADM=%.6f (Σm=%.4f)%s  %.0fs",
                 lb, res["q"], res["resid"]["R_rel_rms"], res["M_ADM"],
                 res["sum_m"],
                 "" if res["M_ADM_ref"] is None else
                 "  ref=%.6f" % res["M_ADM_ref"], res["sec"])

    # ---------------- 输出 ----------------
    out = args.out or os.path.join(run_dir, "physics.json")
    serial = {k: {kk: vv for kk, vv in v.items() if kk != "_R_hist"}
              for k, v in results.items()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run": os.path.relpath(run_dir, _ROOT),
                   "radii": radii, "n_res": args.n_res,
                   "configs": serial}, f, indent=2, ensure_ascii=False)
    log.info("结果 -> %s", out)

    _report(results, radii)
    _figures(results, radii, args.fig_dir or os.path.join(run_dir, "figs"))
    print(f"\n日志文件: {log_f}")


def _report(results, radii):
    print("\n" + "=" * 100)
    print("A. Hamilton 约束残差（相对源项 RMS；模型训练时从未见过 PDE 残差）")
    print("=" * 100)
    print(f"{'cfg':<7}{'q':>8}{'group':>10}{'R_rel_rms':>13}{'near':>11}"
          f"{'mid':>11}{'outer':>11}{'far':>11}")
    for lb, r in results.items():
        rd = r["resid"]
        g = {"heldout": "零样本", "train": "训练", "zero_shot": "零样本"}.get(
            r["group"], r["group"])
        cells = []
        for k in ("near", "mid", "outer", "far"):
            cells.append(f"{rd[k]['R_rel_rms']:.2e}" if k in rd else "-")
        print(f"{lb:<7}{r['q']:>8.3g}{g:>10}{rd['R_rel_rms']:>13.3e}"
              f"{cells[0]:>11}{cells[1]:>11}{cells[2]:>11}{cells[3]:>11}")

    if any("fd" in r for r in results.values()):
        print("\n" + "=" * 100)
        print("A2. 残差对照：模型 vs 谱参考解（同一有限差分 stencil，公平对比）")
        print("=" * 100)
        print(f"{'cfg':<7}{'region':>9}{'h':>7}{'model':>12}{'ref':>12}"
              f"{'|Δu|_mod':>12}{'|Δu|_ref':>12}{'|S|':>12}")
        for lb, r in results.items():
            for reg in ("near", "mid", "outer", "far"):
                if "fd" not in r or reg not in r["fd"]:
                    continue
                f = r["fd"][reg]
                print(f"{lb:<7}{reg:>9}{f['h']:>7.3f}{f['model']:>12.3e}"
                      f"{f['ref']:>12.3e}{f['lap_model_rms']:>12.3e}"
                      f"{f['lap_ref_rms']:>12.3e}{f['S_rms']:>12.3e}")

    print("\n" + "=" * 100)
    print("B. ADM 质量与远场多极（最外壳 r=%.0f）" % radii[-1])
    print("=" * 100)
    print(f"{'cfg':<7}{'q':>8}{'M_ADM':>13}{'M_ADM_ref':>13}{'diff':>11}"
          f"{'Σm':>10}{'M-Σm':>11}{'|d|':>10}{'|d|_ref':>10}")
    for lb, r in results.items():
        ref = r.get("M_ADM_ref")
        d_ref = r.get("dipole_abs_ref")
        diff = f"{r['M_ADM'] - ref:+.2e}" if ref is not None else "-"
        print(f"{lb:<7}{r['q']:>8.3g}{r['M_ADM']:>13.6f}"
              f"{(f'{ref:.6f}' if ref is not None else '-'):>13}"
              f"{diff:>11}"
              f"{r['sum_m']:>10.4f}{r['M_ADM'] - r['sum_m']:>+11.2e}"
              f"{r['dipole_abs']:>10.3e}"
              f"{(f'{d_ref:.3e}' if d_ref is not None else '-'):>10}")


def _figures(results, radii, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:      # noqa: BLE001
        log.warning("matplotlib 不可用,跳过绘图")
        return
    os.makedirs(fig_dir, exist_ok=True)
    labels = list(results.keys())
    qs = [results[l]["q"] for l in labels]

    # --- 图1: 残差分布 + 分区 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for lb in labels:
        h = results[lb]["_R_hist"]
        ax.hist(h, bins=60, histtype="step", lw=1.0,
                label=f"{lb} (q={results[lb]['q']:.3g})")
    ax.set_xlabel(r"$\log_{10}(|R|/S_{\rm rms})$")
    ax.set_ylabel("points")
    ax.set_title("Hamilton residual distribution")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for k, c in (("near", "tab:red"), ("mid", "tab:orange"),
                 ("outer", "tab:green"), ("far", "tab:blue")):
        xs_, ys_ = [], []
        for lb in labels:
            if k in results[lb]["resid"]:
                xs_.append(results[lb]["q"])
                ys_.append(results[lb]["resid"][k]["R_rel_rms"])
        if xs_:
            ax.plot(xs_, ys_, "o-", color=c, ms=4, label=k)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\|R\|/\|S\|$ (rms)")
    ax.set_title("Relative residual by region")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- 图2: M_ADM 收敛 + vs q ---
    ax = axes[2]
    r_last = f"r{radii[-1]:g}"
    for lb in labels[:4]:
        ys_ = [results[lb]["shells"][f"r{r:g}"]["M_ADM"] for r in radii]
        ax.plot(radii, ys_, "o-", ms=4, label=lb)
    for lb in labels[:4]:
        if "M_ADM_ref" in results[lb]["shells"][r_last]:
            ys_ = [results[lb]["shells"][f"r{r:g}"].get("M_ADM_ref")
                   for r in radii]
            ax.plot(radii, ys_, "s--", ms=4, alpha=0.7, label=lb + " ref")
    ax.set_xlabel("shell radius r")
    ax.set_ylabel("$M_{ADM}$")
    ax.set_title("ADM mass convergence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p1 = os.path.join(fig_dir, "physics_residual.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # --- 图3: M_ADM - Σm 与偶极 vs q ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    mm = [results[l]["M_ADM"] for l in labels]
    rr = [results[l].get("M_ADM_ref") for l in labels]
    sm = [results[l]["sum_m"] for l in labels]
    ax.plot(qs, sm, "k--", lw=1, label=r"$\Sigma m = m_1+m_2$ (bare)")
    ax.plot(qs, mm, "o-", ms=5, label="model")
    if any(x is not None for x in rr):
        ax.plot(qs, rr, "s--", ms=4, label="spectral reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("$M_{ADM}$")
    ax.set_title("ADM mass vs mass ratio")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(qs, [results[l]["dipole_abs"] for l in labels], "o-", ms=5,
            label="model")
    if any(results[l].get("dipole_abs_ref") is not None for l in labels):
        ax.plot(qs, [results[l].get("dipole_abs_ref", np.nan)
                     for l in labels], "s--", ms=4, label="reference")
    ax.plot(qs, [results[l]["dipole_abs_analytic"] for l in labels],
            "k:", lw=1, label=r"analytic ($\psi_{sing}$ only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel(r"$|\mathbf{d}|$")
    ax.set_title("Mass dipole vs q (continuity check)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(fig_dir, "physics_adm.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    log.info("图 -> %s , %s", p1, p2)


if __name__ == "__main__":
    main()

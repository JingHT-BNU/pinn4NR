"""tp_to_traindata.py —— TP 子午面参考解 → opv3 训练数据(cfg + refsub)。

轴对称利用:S=0 无动量时 u(x,y,z)=u(x,ρ,0),ρ=√(y²+z²)。TP 子午面半平面
(x,ρ) 网格上的 u 可通过双线性插值给出**任意 3D 点**的参考值 —— 即用一条
子午线重构全 3D 参考场(refsub),点数与原谱 refsub 相同(8192 球内+2048 球面)。

对每个 data/tp_opv3/ref_tq*.npz:
  1. 构造 (x,ρ) 规则网格插值器(块重叠去重取均值);
  2. κ 求解:与 a2q_kappa2 同款(种子 777,2M 体积点+40 万球面点,QMC);
     另算 κ*_tp(对子午面点最小二乘,与 a2q_kappa_fit 同法),写入 json;
  3. 采样 3D 点 → 插值参考 u → refsub_tq*.npz(与 a2q_refsub 同格式);
  4. cfg_tq*.npz:配点 + u_g/∇u_g/Δu_g 预计算(prep+prep2 同款快路径字段);
     kappa 字段写 κ*_tp(与原 15 配置的 κ* 口径一致)。
幂等:已有 cfg_tq*.npz 的配置跳过。
"""
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainConfig
from data import sample_ball, sample_sphere_surface, sobol_volume, \
    sobol_sphere_surface
from logutil import setup_logging
import physics

log = logging.getLogger("paper.A2.tp_to_traindata")

HERE = os.path.dirname(os.path.abspath(__file__))
TP_DIR = os.path.join(HERE, "data", "tp_opv3")
DATA_DIR = os.path.join(HERE, "data", "datasets", "a2q_data")
R_MAX = 30.0
M1 = 0.5
CFG = TrainConfig(n_qmc_vol=2000000, n_qmc_surf=400000, R_max=R_MAX)


def meridian_interp(npz):
    """TP 子午面点 → (x,ρ) 散点线性插值(块步长不同,不能正则网格化;
    块重叠去重取均值)。返回 f(x, ρ) callable 与元信息。"""
    from scipy.interpolate import griddata
    x = npz["x_ref"].astype(np.float64)
    u = npz["u_ref"].astype(np.float64)
    rho = np.hypot(x[:, 1], x[:, 2])
    key = np.stack([np.round(x[:, 0], 6), np.round(rho, 6)], axis=1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    umean = np.bincount(inv, weights=u) / cnt
    # 按半径排序剔除重复半径上的退化三角形:griddata 直接吃散点即可
    def itp(qr):
        # CloughTocher 三次插值(C1,误差 O(h⁴)):外层子午面步长 0.25 下
        # 线性插值误差 ~1e-2(相对),不满足训练参考精度;三次插值可到 ~1e-4。
        vals = griddata(uk, umean, qr, method="cubic")
        bad = ~np.isfinite(vals)
        if bad.any():
            # 凸包外 cubic 返回 NaN → 先 nearest 补,再对凸包内 NaN 用 linear
            vals2 = griddata(uk, umean, qr[bad], method="nearest")
            lin = griddata(uk, umean, qr[bad], method="linear")
            ok2 = np.isfinite(lin)
            vals2[ok2] = lin[ok2]
            vals[bad] = vals2
        return vals
    return itp, dict(n_pts=len(x), n_unique=len(uk))


def kappa_tp(itp, info, q):
    """κ*_tp:在子午面网格点上对 κ·u_g 最小二乘(u_ref = TP,u_g = LZ2008)。"""
    m2 = M1 * q
    gx, gr = info["x"], info["rho"]
    GX, GR = np.meshgrid(gx, gr, indexing="ij")
    pts = np.stack([GX.ravel(), GR.ravel(), np.zeros(GX.size)], axis=1)
    # 排除奇点邻域(r<0.3)与网格缺失
    r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
    keep = np.minimum(r1, r2) > 0.3
    pts_k = pts[keep]
    vals = itp(np.stack([pts_k[:, 0], np.hypot(pts_k[:, 1], pts_k[:, 2])],
                        axis=1)).astype(np.float64)
    t = torch.from_numpy(pts[keep]).double()
    ma = torch.tensor([M1, m2], dtype=torch.float64)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    St = torch.zeros((2, 3), dtype=torch.float64)
    with torch.no_grad():
        ug = physics.guide_u(t, ma, xst, Pt, St).numpy()
    return float((ug * vals).sum() / (ug ** 2).sum())


def solve_kappa_qmc(q, seed=777):
    """与 a2q_kappa2 同款高密度 QMC κ(供参考/对照)。"""
    m2 = M1 * q
    ma = torch.tensor([M1, m2], dtype=torch.float64)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    St = torch.zeros((2, 3), dtype=torch.float64)
    x_vol = sobol_volume(CFG.n_qmc_vol, R_MAX, seed=seed)
    x_srf = sobol_sphere_surface(CFG.n_qmc_surf, R_MAX, seed=seed)
    return float(physics.solve_kappa(CFG, ma, xst, Pt, St, x_vol, x_srf, R_MAX))


def main():
    setup_logging("A2", "tp_to_traindata")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(DATA_DIR, exist_ok=True)
    srcs = sorted(f for f in os.listdir(TP_DIR)
                  if f.startswith("ref_tq") and f.endswith(".npz"))
    log.info(f"TP 参考解 {len(srcs)} 个 → 训练数据")
    ks_out = {}
    ks_path = os.path.join(DATA_DIR, "kappa_tp.json")
    if os.path.exists(ks_path):
        ks_out = json.load(open(ks_path))
    t0 = time.time()
    for i, fn in enumerate(srcs, 1):
        lb = fn[4:-4]  # ref_tq1p5.npz → tq1p5
        cfg_p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        ref_p = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        npz = np.load(os.path.join(TP_DIR, fn))
        q = float(npz["q"])
        if os.path.exists(cfg_p) and os.path.exists(ref_p):
            log.info(f"[skip] {lb}")
            continue
        itp, info = meridian_interp(npz)
        # ---- κ ----
        if lb in ks_out:
            k = float(ks_out[lb]["kappa_star_tp"])
        else:
            k = kappa_tp(itp, info, q)
            ks_out[lb] = dict(q=q, kappa_star_tp=k)
            json.dump(ks_out, open(ks_path, "w"), indent=1)
        # ---- refsub:3D 采样点 → 子午面插值 ----
        rng = np.random.default_rng(777)
        xb = sample_ball(8192, R_MAX, rng).astype(np.float64)
        xs_ = sample_sphere_surface(2048, R_MAX, rng).astype(np.float64)
        pts = np.concatenate([xb, xs_], axis=0)
        rho = np.hypot(pts[:, 1], pts[:, 2])
        u = itp(np.stack([pts[:, 0], rho], axis=1)).astype(np.float64)
        np.savez(ref_p, x=pts.astype(np.float32), u=u)
        # ---- cfg(快路径字段) ----
        m2 = M1 * q
        ma = torch.tensor([M1, m2], dtype=torch.float64)
        xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
        Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                          dtype=torch.float64)
        St = torch.zeros((2, 3), dtype=torch.float64)
        N_INT, N_BND = 12000, 4000
        rng2 = np.random.default_rng(abs(hash(lb)) % (2 ** 31))
        xi = sample_ball(N_INT, R_MAX, rng2).astype(np.float32)
        xbn = sample_sphere_surface(N_BND, R_MAX, rng2).astype(np.float32)

        def _c(x):
            xt = torch.from_numpy(x).double()
            xt.requires_grad_(True)
            ug = physics.guide_u(xt, ma, xst, Pt, St)
            g1 = torch.autograd.grad(ug.sum(), xt, create_graph=True)[0]
            lap = torch.zeros_like(ug)
            for c in range(3):
                g2 = torch.autograd.grad(g1[:, c].sum(), xt,
                                         create_graph=False,
                                         retain_graph=True)[0]
                lap = lap + g2[:, c]
            return (physics.psi_sing(xt, ma, xst).detach().numpy().astype(np.float32),
                    physics.bowen_york_KK(xt, ma, xst, Pt, St).detach().numpy().astype(np.float32),
                    ug.detach().numpy().astype(np.float32),
                    g1.detach().numpy().astype(np.float32),
                    lap.detach().numpy().astype(np.float32))

        pi, ki, ui, gi, li = _c(xi)
        pb, kb, ub, gb, lb_ = _c(xbn)
        sq = float(np.sqrt(np.mean(ui.astype(np.float64) ** 2)) + 1e-30)
        au = np.concatenate([ui, ub]).astype(np.float64)
        np.savez(cfg_p, x_int=xi, x_bnd=xbn, ps_int=pi, kk_int=ki, ug_int=ui,
                 grad_ug=gi, lap_ug=li, ps_bnd=pb, kk_bnd=kb, ug_bnd=ub,
                 grad_ug_b=gb,
                 masses=ma.numpy().astype(np.float32),
                 xs=xst.numpy().astype(np.float32),
                 Ps=Pt.numpy().astype(np.float32),
                 Ss=St.numpy().astype(np.float32),
                 sq=sq, wmin=float(au.min()), wmax=float(au.max()),
                 kappa=float(k), q=float(q), m1=M1, m2=float(m2), heldout=0)
        log.info(f"[{i}/{len(srcs)}] {lb}: q={q:g} κ*_tp={k:.4f} "
                 f"sq={sq:.3e} ({time.time()-t0:.0f}s)")
    log.info(f"完成,用时 {(time.time()-t0)/60:.1f} min → {DATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

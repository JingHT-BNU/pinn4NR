"""post_refs_opv3.py —— 谱参考解后处理:refsub + cfg + κ*_spec 标定。

对 data/refs/a2/ref_tq*.npz(自研谱解,from_coefficients 可重建):
  1. κ*_spec:81³ 网格 rcut=0.3 上对 κ·u_g 最小二乘(与 a2q_kappa_fit 同法,
     与原 15 配置的 kappa_star 口径一致),写 data/datasets/a2q_data/kappa_spec.json;
  2. refsub_tq*.npz:8192 球内 + 2048 球面,float64(seed 777,与 a2q_refsub 同);
  3. cfg_tq*.npz:快路径字段(κ 用 κ*_spec)。
幂等:已有文件跳过;可反复运行增量补齐。GPU ~1 min/配置。
"""
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, r"D:\AIs\PINN\paper\tools")
from logutil import setup_logging  # noqa: E402
from data import sample_ball, sample_sphere_surface  # noqa: E402
import physics  # noqa: E402
from spectral_reference import SpectralPunctureSolver  # noqa: E402

log = logging.getLogger("paper.A2.post_refs_opv3")

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "data", "refs", "a2")
DATA_DIR = os.path.join(HERE, "data", "datasets", "a2q_data")
R_MAX = 30.0
M1 = 0.5
RCUT = 0.3


def main():
    setup_logging("A2", "post_refs_opv3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)
    srcs = sorted(f for f in os.listdir(REFS)
                  if f.startswith("ref_tq") and f.endswith(".npz"))
    log.info(f"谱参考解 {len(srcs)} 个 → 后处理")
    ks_path = os.path.join(DATA_DIR, "kappa_spec.json")
    ks = json.load(open(ks_path)) if os.path.exists(ks_path) else {}
    t0 = time.time()
    for i, fn in enumerate(srcs, 1):
        lb = fn[4:-4]
        ref_p = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        cfg_p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        if os.path.exists(ref_p) and os.path.exists(cfg_p) and lb in ks:
            log.info(f"[skip] {lb}")
            continue
        src = os.path.join(REFS, fn)
        z = np.load(src)
        raw = z["raw"]
        q = float(raw[1] / raw[0])
        m2 = float(raw[1])
        ev = SpectralPunctureSolver.from_coefficients(src, device=str(device),
                                                      verify=False)
        # ---- κ*_spec(81³ rcut=0.3) ----
        if lb not in ks:
            g = np.linspace(-30, 30, 81)
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
            r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
            keep = np.minimum(r1, r2) >= RCUT
            u_ref = ev.evaluate(pts, chunk=131072,
                                dtype=torch.float32).astype(np.float64)
            ma = torch.tensor([M1, m2], dtype=torch.float64, device=device)
            xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                               device=device)
            Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                              dtype=torch.float64, device=device)
            St = torch.zeros((2, 3), dtype=torch.float64, device=device)
            ug = np.empty(len(pts))
            with torch.no_grad():
                for c0 in range(0, len(pts), 262144):
                    xt = torch.from_numpy(pts[c0:c0 + 262144]).double().to(device)
                    ug[c0:c0 + 262144] = physics.guide_u(
                        xt, ma, xst, Pt, St).cpu().numpy()
            k = float((ug[keep] * u_ref[keep]).sum() / (ug[keep] ** 2).sum())
            ks[lb] = dict(q=q, kappa_star_spec=k)
            json.dump(ks, open(ks_path, "w"), indent=1)
            del u_ref
            torch.cuda.empty_cache()
        else:
            k = float(ks[lb]["kappa_star_spec"])
        # ---- refsub ----
        if not os.path.exists(ref_p):
            rng = np.random.default_rng(777)
            xb = sample_ball(8192, R_MAX, rng).astype(np.float64)
            xs_ = sample_sphere_surface(2048, R_MAX, rng).astype(np.float64)
            pts = np.concatenate([xb, xs_], axis=0)
            u = np.asarray(ev.evaluate(pts, chunk=131072, dtype=torch.float64))
            np.savez(ref_p, x=pts.astype(np.float32), u=u.astype(np.float64))
        del ev
        torch.cuda.empty_cache()
        # ---- cfg ----
        if not os.path.exists(cfg_p):
            _make_cfg(lb, q, k, device)
        log.info(f"[{i}/{len(srcs)}] {lb}: q={q:g} κ*_spec={k:.4f} "
                 f"({time.time()-t0:.0f}s)")
    log.info(f"后处理完成,用时 {(time.time()-t0)/60:.1f} min")


def _make_cfg(lb, q, k, device):
    m2 = M1 * q
    ma = torch.tensor([M1, m2], dtype=torch.float64, device=device)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                       device=device)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                      device=device)
    St = torch.zeros((2, 3), dtype=torch.float64, device=device)
    N_INT, N_BND = 12000, 4000
    rng = np.random.default_rng(abs(hash(lb)) % (2 ** 31))
    xi = sample_ball(N_INT, R_MAX, rng).astype(np.float32)
    xbn = sample_sphere_surface(N_BND, R_MAX, rng).astype(np.float32)

    def _c(x):
        xt = torch.from_numpy(x).double().to(device)
        xt.requires_grad_(True)
        ug = physics.guide_u(xt, ma, xst, Pt, St)
        g1 = torch.autograd.grad(ug.sum(), xt, create_graph=True)[0]
        lap = torch.zeros_like(ug)
        for c in range(3):
            g2 = torch.autograd.grad(g1[:, c].sum(), xt, create_graph=False,
                                     retain_graph=True)[0]
            lap = lap + g2[:, c]
        return (physics.psi_sing(xt, ma, xst).detach().cpu().numpy().astype(np.float32),
                physics.bowen_york_KK(xt, ma, xst, Pt, St).detach().cpu().numpy().astype(np.float32),
                ug.detach().cpu().numpy().astype(np.float32),
                g1.detach().cpu().numpy().astype(np.float32),
                lap.detach().cpu().numpy().astype(np.float32))

    pi, ki, ui, gi, li = _c(xi)
    pb, kb, ub, gb, lbv = _c(xbn)
    sq = float(np.sqrt(np.mean(ui.astype(np.float64) ** 2)) + 1e-30)
    au = np.concatenate([ui, ub]).astype(np.float64)
    np.savez(os.path.join(DATA_DIR, f"cfg_{lb}.npz"),
             x_int=xi, x_bnd=xbn, ps_int=pi, kk_int=ki, ug_int=ui,
             grad_ug=gi, lap_ug=li, ps_bnd=pb, kk_bnd=kb, ug_bnd=ub,
             grad_ug_b=gb, masses=ma.cpu().numpy().astype(np.float32),
             xs=xst.cpu().numpy().astype(np.float32),
             Ps=Pt.cpu().numpy().astype(np.float32),
             Ss=St.cpu().numpy().astype(np.float32),
             sq=sq, wmin=float(au.min()), wmax=float(au.max()),
             kappa=float(k), q=float(q), m1=M1, m2=float(m2), heldout=0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

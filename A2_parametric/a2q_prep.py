"""a2q_prep.py —— A2 单参数 q∈[1,10] 攻关:训练数据准备。

产出(全部落在 a2q_data/):
  kappa_a2q.json              18 个配置(15 训练 + 3 留出)的 κ
  cfg_<label>.npz             每配置: 配点 + 引导场预计算 + 归一化常数
      x_int(N_int,3) x_bnd(N_bnd,3) ps_int/kk_int/ug_int ps_bnd/kk_bnd/ug_bnd
      sq(引导 RMS, 残差归一) wmin/wmax(窗口归一) kappa q m1 m2
  断点续跑:已有 npz 的配置自动跳过;κ 缓存已存在则复用。
"""
import json, logging, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BBHConfig, TrainConfig
from data import sample_ball, sample_sphere_surface, sobol_volume, sobol_sphere_surface
from logutil import setup_logging
import physics

log = logging.getLogger("paper.A2.a2q_prep")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "a2q_data")
os.makedirs(DATA_DIR, exist_ok=True)

# q = m2/m1, m1 = 0.5 固定 → m2 = 0.5 q
Q_TRAIN = [(1.0, "q10"), (1.2, "q12"), (1.4, "q14"), (1.7, "q17"), (2.0, "q20"),
           (2.4, "q24"), (2.8, "q28"), (3.3, "q33"), (3.9, "q39"), (4.6, "q46"),
           (5.4, "q54"), (6.3, "q63"), (7.4, "q74"), (8.6, "q86"), (10.0, "q100")]
Q_HELDOUT = [(1.5, "q15"), (2.5, "q25"), (5.0, "q50")]

N_INT, N_BND = 12000, 4000
R_MAX = 30.0


def build_bbh(m1, m2):
    return BBHConfig(m_plus=m1, m_minus=m2, x_plus=(3, 0, 0), x_minus=(-3, 0, 0),
                     P_plus=(0, 0.2, 0), P_minus=(0, -0.2, 0))


def main():
    setup_logging("A2", "a2q_prep")
    cfg = TrainConfig(n_qmc_vol=200000, n_qmc_surf=20000, R_max=R_MAX)
    kc_path = os.path.join(DATA_DIR, "kappa_a2q.json")
    kc = json.load(open(kc_path)) if os.path.exists(kc_path) else {}

    all_cfg = [(q, lb, False) for q, lb in Q_TRAIN] + [(q, lb, True) for q, lb in Q_HELDOUT]
    t0 = time.time()
    for q, lb, held in all_cfg:
        out = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        if os.path.exists(out):
            log.info(f"[skip] {lb} 已存在")
            continue
        m1, m2 = 0.5, 0.5 * q
        if lb not in kc:
            log.info(f"[κ] {lb} (q={q}, m2={m2}) 求解中...({time.time()-t0:.0f}s)")
            bb = build_bbh(m1, m2)
            ma, xs, Ps, Ss = bb.as_arrays()
            x_vol = sobol_volume(cfg.n_qmc_vol, R_MAX, seed=hash(lb) % (2**31))
            x_srf = sobol_sphere_surface(cfg.n_qmc_surf, R_MAX, seed=hash(lb) % (2**31))
            kc[lb] = float(physics.solve_kappa(cfg, ma, xs, Ps, Ss, x_vol, x_srf, R_MAX))
            json.dump(kc, open(kc_path, "w"), indent=1)
            log.info(f"[κ] {lb}: κ={kc[lb]:.6f}")
        k = kc[lb]
        bb = build_bbh(m1, m2)
        ma, xs, Ps, Ss = bb.as_arrays()
        rng = np.random.default_rng(abs(hash(lb)) % (2**31))
        xi = sample_ball(N_INT, R_MAX, rng).astype(np.float32)
        xb = sample_sphere_surface(N_BND, R_MAX, rng).astype(np.float32)
        mt = torch.from_numpy(ma).double()
        xst = torch.from_numpy(xs).double()
        Pt = torch.from_numpy(Ps).double()
        St = torch.from_numpy(Ss).double()

        def _c(x):
            xt = torch.from_numpy(x).double()
            return (physics.psi_sing(xt, mt, xst).numpy().astype(np.float32),
                    physics.bowen_york_KK(xt, mt, xst, Pt, St).numpy().astype(np.float32),
                    physics.guide_u(xt, mt, xst, Pt, St).numpy().astype(np.float32))

        pi, ki, ui = _c(xi)
        pb, kb, ub = _c(xb)
        au = np.concatenate([ui, ub])
        sq = float(np.sqrt(np.mean(ui.astype(np.float64) ** 2)) + 1e-30)
        np.savez(out, x_int=xi, x_bnd=xb, ps_int=pi, kk_int=ki, ug_int=ui,
                 ps_bnd=pb, kk_bnd=kb, ug_bnd=ub,
                 masses=ma.astype(np.float32), xs=xs.astype(np.float32),
                 Ps=Ps.astype(np.float32), Ss=Ss.astype(np.float32),
                 sq=sq, wmin=float(au.min()), wmax=float(au.max()),
                 kappa=float(k), q=float(q), m1=float(m1), m2=float(m2),
                 heldout=int(held))
        big = ui.max()
        log.info(f"[cfg] {lb}: q={q} m2={m2} κ={k:.4f} sq={sq:.3e} "
                 f"ug_max={big:.3e} ({time.time()-t0:.0f}s)")
    log.info(f"准备完成,总用时 {(time.time()-t0)/60:.1f} min → {DATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

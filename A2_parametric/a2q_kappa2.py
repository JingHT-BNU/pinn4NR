"""a2q_kappa2.py —— 用同一种子 + 高密度 Sobol 重解全部配置的 κ,替换 kappa 缓存。

背景:prep 用 seed=hash(label)(进程间随机)的 20 万点 Sobol 逐配置求 κ,
QMC 噪声导致 κ(q) 非单调跳变(±15%),与模型 L2RE 灾难区(q≈2)完全重合。
本脚本:统一 seed=777、2,000,000 体积点(共享采样 → 剩余偏差对 q 光滑),
对每配置在 GPU 上求 f(κ)=−κ·S_b−V(κ)/8 的根(brentq),更新:
  a2q_data/kappa_a2q.json          (旧值备份 kappa_old)
  a2q_data/cfg_<lb>.npz 的 kappa 字段
refsub/cfg 其余字段与 κ 无关,无需重算。
"""
import json, logging, os, sys, time

import numpy as np
import torch
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import physics
from data import sobol_volume, sobol_sphere_surface
from a2q_prep import Q_TRAIN, Q_HELDOUT, DATA_DIR, R_MAX

log = logging.getLogger("paper.A2.a2q_kappa2")

N_VOL = 2000000
N_SURF = 400000
SEED = 777


def main():
    setup_logging("A2", "a2q_kappa2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_vol = torch.from_numpy(sobol_volume(N_VOL, R_MAX, seed=SEED)).double().to(device)
    x_srf = torch.from_numpy(sobol_sphere_surface(N_SURF, R_MAX, seed=SEED)).double().to(device)
    x_srf_out = x_srf * 1.001
    vol_factor = (4.0 / 3.0) * np.pi * R_MAX ** 3

    kc_path = os.path.join(DATA_DIR, "kappa_a2q.json")
    kc_old = json.load(open(kc_path))
    kc_new, rows = {}, []
    t0 = time.time()
    for q, lb in [(q, lb) for q, lb in Q_TRAIN] + list(Q_HELDOUT):
        z = dict(np.load(os.path.join(DATA_DIR, f"cfg_{lb}.npz")))
        ma = torch.tensor(z["masses"], device=device, dtype=torch.float64)
        xs = torch.tensor(z["xs"], device=device, dtype=torch.float64)
        Ps = torch.tensor(z["Ps"], device=device, dtype=torch.float64)
        Ss = torch.tensor(z["Ss"], device=device, dtype=torch.float64)
        with torch.no_grad():
            ug_v = physics.guide_u(x_vol, ma, xs, Ps, Ss)
            kk_v = physics.bowen_york_KK(x_vol, ma, xs, Ps, Ss)
            ps_v = physics.psi_sing(x_vol, ma, xs)
            ug_b1 = physics.guide_u(x_srf, ma, xs, Ps, Ss)
            ug_b2 = physics.guide_u(x_srf_out, ma, xs, Ps, Ss)
        dudr = (ug_b2 - ug_b1) / (0.001 * R_MAX)
        S_b = float(4 * np.pi * R_MAX ** 2 * dudr.mean().item())
        src = kk_v * vol_factor          # V(κ) = mean(kk/ψ⁷)·vol_factor
        base = ps_v
        kappa_old = float(z["kappa"])    # 旧值以 cfg npz 为准(缓存 json 可能已被清)

        def f(k):
            with torch.no_grad():
                v = float((src / torch.clamp(base + k * ug_v, min=1e-3) ** 7).mean())
            return -k * S_b - v / 8.0

        # 扫描定符号变化区间
        ks = np.linspace(0.02, 1.6, 80)
        fs = [f(k) for k in ks]
        root = None
        for i in range(len(ks) - 1):
            if fs[i] < 0 <= fs[i + 1]:
                root = brentq(f, ks[i], ks[i + 1], xtol=1e-6)
                break
        if root is None:
            log.error(f"{lb}: 未找到根! fs 末值 {fs[-1]:.3e}")
            continue
        kc_new[lb] = {"q": float(q), "m1": 0.5, "m2": 0.5 * float(q),
                      "kappa": float(root), "kappa_old": kappa_old}
        # 更新 cfg npz
        z["kappa"] = np.float64(root)
        tmp = os.path.join(DATA_DIR, f"cfg_{lb}.npz.tmp.npz")
        np.savez(tmp, **z)
        os.replace(tmp, os.path.join(DATA_DIR, f"cfg_{lb}.npz"))
        rows.append((lb, float(q), kappa_old, float(root)))
        log.info(f"[κ2] {lb}: q={q:g} 旧={kappa_old:.4f} → 新={root:.4f} "
                 f"({time.time()-t0:.0f}s)")
    if not kc_new:
        log.error("全部配置未找到根,不写缓存!")
        raise SystemExit(1)
    json.dump(kc_new, open(kc_path, "w"), indent=1)
    log.info("=== κ 重解完成;旧→新对照 ===")
    for lb, q, old, new in rows:
        log.info(f"  {lb:<5} q={q:<5g} {old:.4f} → {new:.4f}  ({(new-old)/old*100:+.1f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

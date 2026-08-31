"""precompute_kappa.py —— 预计算所有参数的 κ 值并缓存"""
import logging, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from config import TrainConfig, BBHConfig
from logutil import setup_logging
import physics
from data import sobol_volume, sobol_sphere_surface
from parametric_train import TRAIN_PARAMS, VAL_PARAMS

log = logging.getLogger("paper.A2.precompute_kappa")

def build_bbh(m1, m2):
    return BBHConfig(m_plus=m1, m_minus=m2, x_plus=(3,0,0), x_minus=(-3,0,0),
                     P_plus=(0,0.2,0), P_minus=(0,-0.2,0))

def main():
    setup_logging("A2", "precompute_kappa")
    R_max = 30.0
    n_vol = 200000
    n_surf = 20000
    cfg = TrainConfig(n_qmc_vol=n_vol, n_qmc_surf=n_surf, R_max=R_max)

    all_params = TRAIN_PARAMS + VAL_PARAMS
    kappa_cache = {}

    log.info("预计算 κ 值...")
    for m1, m2, label in all_params:
        log.info(f"  {label}: m1={m1}, m2={m2}...")
        bb = build_bbh(m1, m2)
        masses, xs, Ps, Ss = bb.as_arrays()
        x_vol = sobol_volume(n_vol, R_max)
        x_surf = sobol_sphere_surface(n_surf, R_max)
        kappa = physics.solve_kappa(cfg, masses, xs, Ps, Ss, x_vol, x_surf, R_max)
        kappa_cache[label] = {"m1": m1, "m2": m2, "kappa": kappa}
        log.info(f"    κ={kappa:.6f}")

    # 保存缓存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa_cache.json")
    with open(out_path, "w") as f:
        json.dump(kappa_cache, f, indent=2)
    log.info(f"\nκ 缓存已保存: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
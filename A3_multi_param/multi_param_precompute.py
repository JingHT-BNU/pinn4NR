"""
multi_param_precompute.py —— 多参数参数化 PINN 的 κ 预计算(v2)
================================================================

v2: 修复 κ 估计的大方差问题:
    1. Sobol(scramble=True, seed=...) 固定种子 → 完全可复现
    2. n_vol=1,000,000, n_surf=50,000 → κ 收敛到 ~1% 精度
    3. 旧版 200k 点 + 随机扰动导致 base κ 在 0.55~0.67 间波动(真值 0.6385)

功能:
    1. 用 Latin Hypercube Sampling 在 8 维参数空间采样 N 个配置
    2. 对每个配置计算 κ (Sobol QMC)
    3. 计算 guide_u 范围 (用于窗函数 W)
    4. 保存为 JSON 缓存

用法:
    python multi_param_precompute.py --n-train 300 --n-val 100 --seed 42
"""

import argparse, json, logging, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sobol_volume, sobol_sphere_surface
from logutil import setup_logging
import physics
from multi_param_model import (
    PARAM_NAMES, PARAM_LO, PARAM_HI, PARAM_RANGE,
    normalize_params, build_bbh_from_params, BASE_RAW,
)

log = logging.getLogger("paper.A3.multi_param_precompute")

N_VOL = 2000000
N_SURF = 50000
SOBOL_SEED = 12345
N_KAPPA_SEEDS = 2                # κ 多 seed 平均, 抵消 scramble 噪声
R_MAX = 30.0


# ── LHS 配置采样 ─────────────────────────────────────────────

def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


def generate_configs(n_train: int, n_val: int, seed: int = 42):
    n_total = n_train + n_val
    lhs_unit = latin_hypercube_sample(n_total, len(PARAM_NAMES), seed=seed)
    all_raw = lhs_unit * PARAM_RANGE + PARAM_LO
    all_raw[0] = BASE_RAW

    base_norm = normalize_params(BASE_RAW)
    all_norm = normalize_params(all_raw)
    dists = np.linalg.norm(all_norm - base_norm, axis=1)
    sorted_idx = np.argsort(dists)

    train_raw = all_raw[sorted_idx[:n_train]].copy()
    val_raw = all_raw[sorted_idx[n_train:]].copy()

    if not np.allclose(train_raw[0], BASE_RAW):
        base_pos = np.where(np.all(np.isclose(train_raw, BASE_RAW), axis=1))[0]
        if len(base_pos) > 0:
            train_raw[[0, base_pos[0]]] = train_raw[[base_pos[0], 0]]
    return train_raw, val_raw


# ── κ 计算 ───────────────────────────────────────────────────

def compute_config_info(raw_params, n_vol, n_surf, base_seed, n_seeds=N_KAPPA_SEEDS):
    """κ = 多个独立 Sobol 序列估计的平均值(降低 scramble 噪声)。

    注: u_min/u_max 不在此计算 —— 训练脚本 MultiParamData 用实际训练采样点求范围。
    """
    masses, xs, Ps, Ss = build_bbh_from_params(raw_params)

    from config import TrainConfig
    cfg = TrainConfig()

    ks = []
    for s in range(n_seeds):
        x_vol = sobol_volume(n_vol, R_MAX, seed=base_seed + s * 7919)
        x_surf = sobol_sphere_surface(n_surf, R_MAX, seed=base_seed + s * 7919 + 1)
        ks.append(physics.solve_kappa(cfg, masses, xs, Ps, Ss, x_vol, x_surf, R_MAX))
    kappa = float(np.mean(ks))

    return {
        "kappa": kappa,
        "kappa_seeds": [float(k) for k in ks],
        "kappa_spread": float(max(ks) - min(ks)),
        "masses": masses.tolist(),
        "xs": xs.tolist(),
        "Ps": Ps.tolist(),
        "Ss": Ss.tolist(),
    }


# ── 主函数 ───────────────────────────────────────────────────

def main():
    setup_logging("A3", "multi_param_precompute")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-vol", type=int, default=N_VOL)
    parser.add_argument("--n-surf", type=int, default=N_SURF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multi_param_kappa_cache.json")

    log.info(f"生成 {args.n_train}+{args.n_val} 个配置 (LHS, 8维)...")
    train_raw, val_raw = generate_configs(args.n_train, args.n_val, seed=args.seed)
    log.info(f"  训练集: {train_raw.shape}, 验证集: {val_raw.shape}")
    log.info(f"  Base case 在 train[0]: {np.allclose(train_raw[0], BASE_RAW)}")
    log.info(f"  κ 积分: {args.n_vol} 体积 + {args.n_surf} 表面, {N_KAPPA_SEEDS} 个独立 seed 平均")
    log.info(f"  κ 噪声目标: base 配置应 ≈ 0.6385 (TwoPunctures 标称值)")

    cache = {"train": [], "val": [], "meta": {
        "n_train": args.n_train, "n_val": args.n_val,
        "n_vol": args.n_vol, "n_surf": args.n_surf, "n_kappa_seeds": N_KAPPA_SEEDS,
        "R_max": R_MAX, "seed": args.seed, "sobol_seed": SOBOL_SEED, "version": 2,
    }}

    t0 = time.time()
    total = len(train_raw) + len(val_raw)
    for split, raw_set in [("train", train_raw), ("val", val_raw)]:
        for i, raw in enumerate(raw_set):
            sobol_seed = SOBOL_SEED + i * 101
            info = compute_config_info(raw, args.n_vol, args.n_surf, sobol_seed)
            info["label"] = f"{split}_{i:04d}"
            info["raw_params"] = raw.tolist()
            info["norm_params"] = normalize_params(raw).tolist()
            cache[split].append(info)
            done = len(cache["train"]) + len(cache["val"])
            if done % 10 == 0 or done <= 3 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                log.info(f"  [{done}/{total}] {info['label']}: κ={info['kappa']:.4f} "
                         f"(seed spread={info['kappa_spread']:.4f}) "
                         f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"\nκ 缓存已保存: {out_path}")
    log.info(f"总用时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise
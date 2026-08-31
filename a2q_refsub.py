"""a2q_refsub.py —— 从谱参考解 npz 生成训练/评估用参考子采样。

对每个 label:若 refs_a2/ref_a2_<label>.npz 存在且 a2q_data/refsub_<label>.npz
缺失 → from_coefficients 重建求值器,采 8192 球内点 + 2048 球面点,float64 求值。
可反复运行(增量补齐),A2-2 变体需要全部 15 个训练配置的子采样。
"""
import json, logging, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball, sample_sphere_surface
from logutil import setup_logging
from a2q_prep import Q_TRAIN, Q_HELDOUT, DATA_DIR, R_MAX

log = logging.getLogger("paper.A2.a2q_refsub")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFS = os.path.join(ROOT, "paper", "tools", "refs_a2")
sys.path.insert(0, os.path.join(ROOT, "paper", "tools"))
from spectral_reference import SpectralPunctureSolver  # noqa: E402


def main():
    setup_logging("A2", "a2q_refsub")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = [lb for _, lb in Q_TRAIN] + [lb for _, lb in Q_HELDOUT]
    todo = []
    for lb in labels:
        src = os.path.join(REFS, f"ref_a2_{lb}.npz")
        dst = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        if os.path.exists(dst):
            log.info(f"[skip] {lb}")
        elif os.path.exists(src):
            todo.append((lb, src, dst))
        else:
            log.info(f"[wait] {lb} 参考解尚未生成")
    t0 = time.time()
    for i, (lb, src, dst) in enumerate(todo, 1):
        log.info(f"[{i}/{len(todo)}] {lb} 求值中...")
        ev = SpectralPunctureSolver.from_coefficients(src, device=device, verify=False)
        rng = np.random.default_rng(777)
        xb = sample_ball(8192, R_MAX, rng).astype(np.float64)
        xs_ = sample_sphere_surface(2048, R_MAX, rng).astype(np.float64)
        ub = ev.evaluate(xb, chunk=131072, dtype=torch.float64)
        us = ev.evaluate(xs_, chunk=131072, dtype=torch.float64)
        np.savez(dst, x=np.concatenate([xb, xs_], axis=0).astype(np.float32),
                 u=np.concatenate([ub, us], axis=0).astype(np.float64))
        log.info(f"    {lb}: |u|∈[{np.concatenate([ub,us]).min():.3e},"
                 f"{np.concatenate([ub,us]).max():.3e}] ({time.time()-t0:.0f}s)")
    log.info(f"refsub 完成 {len(todo)} 个,用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

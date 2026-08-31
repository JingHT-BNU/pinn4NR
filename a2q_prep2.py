"""a2q_prep2.py —— 为 cfg_*.npz / refsub_*.npz 追加引导场导数(训练加速用)。

对每个 cfg npz 计算 x_int 的 ∇u_g、Δu_g 与 x_bnd 的 ∇u_g(torch double autograd,
一次性成本),存回 npz(grad_ug, lap_ug, grad_ug_b)。对每个 refsub npz 计算参考
点上的 u_g(ug),供参考损失走"预计算部件"快路径。
幂等:已有字段的文件跳过。数学上与"对完整 u 自动微分"严格等价:
  Δu = κ[Δu_g(1+φ) + 2∇u_g·∇φ + u_g Δφ],  φ = w·ψ_θ 只依赖 MLP。
"""
import logging, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logutil import setup_logging
import physics
from a2q_prep import Q_TRAIN, Q_HELDOUT, DATA_DIR

log = logging.getLogger("paper.A2.a2q_prep2")


def guide_parts(x, ma, xs, Ps, Ss, want_lap=True):
    xt = torch.from_numpy(x).double()
    xt.requires_grad_(True)
    ug = physics.guide_u(xt, ma, xs, Ps, Ss)
    g1 = torch.autograd.grad(ug.sum(), xt, create_graph=True)[0]
    if want_lap:
        lap = torch.zeros_like(ug)
        for i in range(3):
            g2 = torch.autograd.grad(g1[:, i].sum(), xt,
                                     create_graph=False, retain_graph=True)[0]
            lap = lap + g2[:, i]
        return (ug.detach().numpy(), g1.detach().numpy(), lap.detach().numpy())
    return ug.detach().numpy(), g1.detach().numpy(), None


def main():
    setup_logging("A2", "a2q_prep2")
    t0 = time.time()
    for lb in [lb for _, lb in Q_TRAIN] + [lb for _, lb in Q_HELDOUT]:
        p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        z = dict(np.load(p))
        if "grad_ug" in z:
            log.info(f"[skip] cfg_{lb}")
        else:
            ma = torch.tensor(z["masses"]).double()
            xs = torch.tensor(z["xs"]).double()
            Ps = torch.tensor(z["Ps"]).double()
            Ss = torch.tensor(z["Ss"]).double()
            ug, gug, lug = guide_parts(z["x_int"], ma, xs, Ps, Ss, want_lap=True)
            _, gug_b, _ = guide_parts(z["x_bnd"], ma, xs, Ps, Ss, want_lap=False)
            z["grad_ug"] = gug.astype(np.float32)
            z["lap_ug"] = lug.astype(np.float32)
            z["grad_ug_b"] = gug_b.astype(np.float32)
            tmp = p + ".tmp.npz"
            np.savez(tmp, **z)
            os.replace(tmp, p)
            log.info(f"[cfg] {lb}: ∇/Δ u_g 完成 ({time.time()-t0:.0f}s)")
        # refsub 增强
        pr = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        if os.path.exists(pr):
            zr = dict(np.load(pr))
            if "ug" not in zr:
                ma = torch.tensor(z["masses"]).double()
                xs = torch.tensor(z["xs"]).double()
                Ps = torch.tensor(z["Ps"]).double()
                Ss = torch.tensor(z["Ss"]).double()
                ug_r, _, _ = guide_parts(zr["x"], ma, xs, Ps, Ss, want_lap=False)
                zr["ug"] = ug_r.astype(np.float32)
                tmp = pr + ".tmp.npz"
                np.savez(tmp, **zr)
                os.replace(tmp, pr)
                log.info(f"[refsub] {lb}: ug 完成 ({time.time()-t0:.0f}s)")
    log.info(f"prep2 完成,用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("运行失败")
        raise

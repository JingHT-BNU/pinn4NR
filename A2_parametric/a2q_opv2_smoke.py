"""a2q_opv2_smoke.py —— 算子 v2 冒烟 + patch 特征开销计时。

训练正确性由 a2q_train.py --variant opv2 的小步数冒烟覆盖;本脚本只测:
  1. patch_feats 在 GPU/CPU 上每步新增耗时(内点 8000 + 边界 4000,×24 偏移);
  2. OperatorV2Ansatz 一次 forward + backward 的数值健康(有限性)。
"""
import os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics
from a2q_model import OperatorV2Ansatz, param_vec, patch_offsets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")
    ma = torch.tensor([0.5, 1.0], dtype=torch.float64, device=dev)
    xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64, device=dev)
    Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                      device=dev)
    Ss = torch.zeros((2, 3), dtype=torch.float64, device=dev)
    off = torch.tensor(patch_offsets(), dtype=torch.float64, device=dev)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8000, 3))
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(0.3, 30, (8000, 1))
    xt = torch.tensor(x, dtype=torch.float64, device=dev)

    def patch_once(n):
        pts = (xt[:n].unsqueeze(1) + off).reshape(-1, 3)
        pug = physics.guide_u(pts, ma, xs, Ps, Ss).reshape(n, -1)
        return torch.log1p(pug.abs() / 0.3)

    for n, rep in ((12000, 5),):
        patch_once(n)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(rep):
            patch_once(n)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / rep
        print(f"patch eval: n={n}(x{off.shape[0]} pts={n * off.shape[0]}) "
              f"{dt * 1000:.0f} ms/次 → 每训练步(内8000+边4000)约 "
              f"{dt / 12000 * 12000 * (12000 / 12000) * 1000:.0f} ms 量级")

    m = OperatorV2Ansatz().to(dev).double()
    p = param_vec(2.0, 1.0, dev)
    xi = xt[:4000].requires_grad_(True)
    with torch.no_grad():
        pf = m.patch_feats(xi.detach(), ma, xs, Ps, Ss, 0.3)
    ug = physics.guide_u(xi, ma, xs, Ps, Ss)
    w = (ug - 0.0) / (1.0 + 1e-8)
    u, phi, psi = m.forward_from_parts(xi, p, 0.6, ug, w, sq=0.3, feats=pf)
    loss = (u ** 2).mean()
    loss.backward()
    gn = sum(pp.grad.abs().sum().item() for pp in m.parameters() if pp.grad is not None)
    print(f"forward/backward ok: u finite={torch.isfinite(u).all().item()} "
          f"|psi|max={psi.abs().max().item():.3e} grad_sum={gn:.3e}")
    # eval 路径:走真实 load_run → predict_a2q(冒烟训练产物,权重 float32)
    from a2q_model import load_run, predict_a2q
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", "smoke_opv2")
    model, ck = load_run(run_dir, dev)
    cinfo = {k: next(iter(ck["meta"].values()))[k]
             for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
    ue = predict_a2q(model, x.astype(np.float32), cinfo, dev)
    print(f"eval predict_a2q ok: shape={ue.shape} finite={np.isfinite(ue).all()}")


if __name__ == "__main__":
    main()

"""a2q_kappa_fit.py —— 用参考解远场幅值直接标定 κ*(q),验证"κ 标定地板"假说。

定义(官方口径:参考解 = 谱系数重建,81³ 网格 rcut=0.3):
  κ*_band = Σ_band(u_g·u_ref) / Σ_band(u_g²)   —— 使 κ·u_g 在该带最小二乘最优
  κ*_all  同式全网格。
输出每配置:QMC κ(当前)、κ*_far、κ*_all、相对偏差,以及
  "κ* 重标定后的引导解 L2RE"(far 与 ALL)—— 即无模型可达的地板。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))
import physics
from spectral_reference import SpectralPunctureSolver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(ROOT, "tools", "refs_a2")
RCUT = 0.3
LABELS = ["q10", "q12", "q14", "q15", "q17", "q20", "q24", "q25", "q28", "q33",
          "q39", "q46", "q50", "q54", "q63", "q74", "q86", "q100"]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = np.linspace(-30, 30, 81)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
    rmin = np.minimum(r1, r2)
    keep = rmin >= RCUT
    far = keep & (rmin >= 6)

    out = {}
    print(f"{'lb':<6}{'κ_qmc':>8}{'κ*_far':>9}{'κ*_all':>9}{'dev_far%':>9}"
          f"{'L2RE_g':>9}{'L2RE_g(κ*)':>11}{'L2RE_g_far':>11}{'far(κ*)':>9}")
    for lb in LABELS:
        src = os.path.join(REFS, f"ref_a2_{lb}.npz")
        ev = SpectralPunctureSolver.from_coefficients(src, device=str(dev),
                                                      verify=False)
        u_ref = ev.evaluate(pts, chunk=65536, dtype=torch.float32).astype(np.float64)
        del ev
        torch.cuda.empty_cache()
        z = np.load(os.path.join(HERE, "a2q_data", f"cfg_{lb}.npz"))
        kappa = float(z["kappa"])
        q, m2 = float(z["q"]), float(z["m2"])
        ma = torch.tensor([0.5, m2], dtype=torch.float64, device=dev)
        xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64, device=dev)
        Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                          device=dev)
        St = torch.zeros((2, 3), dtype=torch.float64, device=dev)
        ug = np.empty(len(pts))
        with torch.no_grad():
            for c0 in range(0, len(pts), 262144):
                xt = torch.from_numpy(pts[c0:c0 + 262144]).to(dev)
                ug[c0:c0 + 262144] = physics.guide_u(xt, ma, xs, Ps, St).cpu().numpy()
        k_far = float((ug[far] * u_ref[far]).sum() / (ug[far] ** 2).sum())
        k_all = float((ug[keep] * u_ref[keep]).sum() / (ug[keep] ** 2).sum())

        def l2re(a, b, sel):
            return float(np.linalg.norm(a[sel] - b[sel]) / np.linalg.norm(b[sel]))
        l_g = l2re(kappa * ug, u_ref, keep)
        l_gs = l2re(k_all * ug, u_ref, keep)
        l_gf = l2re(kappa * ug, u_ref, far)
        l_gfs = l2re(k_far * ug, u_ref, far)
        out[lb] = dict(q=q, kappa_qmc=kappa, kappa_star_far=k_far, kappa_star_all=k_all,
                       l2re_guide=l_g, l2re_guide_recal=l_gs,
                       l2re_guide_far=l_gf, l2re_guide_far_recal=l_gfs)
        print(f"{lb:<6}{kappa:>8.4f}{k_far:>9.4f}{k_all:>9.4f}"
              f"{(k_all / kappa - 1) * 100:>8.1f}%{l_g:>9.3e}{l_gs:>11.3e}"
              f"{l_gf:>11.3e}{l_gfs:>9.3e}")
    with open(os.path.join(HERE, "a2q_data", "kappa_star.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved: a2q_data/kappa_star.json")


if __name__ == "__main__":
    main()

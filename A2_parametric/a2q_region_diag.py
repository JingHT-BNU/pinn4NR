"""a2q_region_diag.py —— L2RE 近场/远场分解(官方口径):定位 3.2e-2 平台误差。

与 a2q_eval.py 完全同口径:
  - 参考解 = SpectralPunctureSolver.from_coefficients(npz).evaluate(pts) 重建;
  - 引导基线 = κ·u_g(含 κ 标定);
  - 81³ 网格 [-30,30]³,rcut=0.3 排除奇点球。
按"到最近孔距离"分带:near<2 / mid2-6 / far>=6。
"""
import os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))
from a2q_model import load_run, predict_a2q
import physics
from spectral_reference import SpectralPunctureSolver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
REFS = os.path.join(ROOT, "tools", "refs_a2")
RCUT = 0.3


def l2re(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = np.linspace(-30, 30, 81)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    r1 = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)
    r2 = np.linalg.norm(pts - np.array([-3.0, 0, 0]), axis=1)
    rmin = np.minimum(r1, r2)
    keep = rmin >= RCUT
    bands = {"near<2": rmin < 2, "mid2-6": (rmin >= 2) & (rmin < 6),
             "far>=6": rmin >= 6}
    evals = {}
    runs = sys.argv[1:] or ["a2q_base", "a2q_champion", "a2q_opv2"]
    for rn in runs:
        rd = os.path.join(RUNS, rn)
        if not os.path.isdir(rd):
            print(f"skip {rn}")
            continue
        model, ck = load_run(rd, dev)
        for lb in ["q10", "q20", "q54", "q100"]:
            src = os.path.join(REFS, f"ref_a2_{lb}.npz")
            if src not in evals:
                evals[src] = SpectralPunctureSolver.from_coefficients(
                    src, device=str(dev), verify=False)
            ev = evals[src]
            u_ref = ev.evaluate(pts, chunk=131072, dtype=torch.float32)
            m = ck["meta"][lb]
            cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
            u = predict_a2q(model, pts.astype(np.float32), cinfo, dev)
            ug = np.empty(len(pts))
            ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64, device=dev)
            xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                               device=dev)
            Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                              dtype=torch.float64, device=dev)
            St = torch.zeros((2, 3), dtype=torch.float64, device=dev)
            with torch.no_grad():
                for c0 in range(0, len(pts), 262144):
                    xt = torch.from_numpy(pts[c0:c0 + 262144]).double().to(dev)
                    ug[c0:c0 + 262144] = (float(m["kappa"]) *
                                          physics.guide_u(xt, ma, xst, Pt, St)
                                          ).cpu().numpy()
            print(f"\n--- {rn} {lb} ---")
            for nm, mband in bands.items():
                sel = keep & mband
                lr = l2re(u[sel], u_ref[sel])
                lg = l2re(ug[sel], u_ref[sel])
                print(f"  {nm:8s} n={sel.sum():6d}  模型={lr:.3e}  引导={lg:.3e}"
                      f"  改进×={lg / lr:.1f}")
            sel = keep
            print(f"  {'ALL':8s} n={sel.sum():6d}  模型={l2re(u[sel], u_ref[sel]):.3e}"
                  f"  引导={l2re(ug[sel], u_ref[sel]):.3e}")


if __name__ == "__main__":
    main()

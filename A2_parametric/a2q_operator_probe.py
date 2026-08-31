"""a2q_operator_probe.py —— 测量算子/基线 ansatz 的修正场实际用量。

corr_eff = (u/(κ·u_g) − 1) / w;  算子变体界为 ±3,基线为 ±c(可学习)。
若 |corr_eff| 远小于上界 → 额外自由度未被使用(与"与基线完全一致"互证)。
"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import sample_ball
import physics
from a2q_model import load_run, predict_a2q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
R_MAX = 30.0


def probe(run_name, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(os.path.join(RUNS, run_name), device)
    cval = None
    for k, v in model.state_dict().items():
        if k.endswith("c"):
            cval = float(v)
    print(f"\n===== {run_name} (variant={ck['variant']}, c={cval}) =====")
    rng = np.random.default_rng(5)
    x = sample_ball(50000, R_MAX, rng).astype(np.float32)
    for lb in labels:
        m = ck["meta"][lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
        u = predict_a2q(model, x, cinfo, device)
        ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64)
        xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
        Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
        St = torch.zeros((2, 3), dtype=torch.float64)
        ug = physics.guide_u(torch.from_numpy(x).double(), ma, xs, Ps, St).numpy()
        w = (ug - m["wmin"]) / (m["wmax"] - m["wmin"] + 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (u / (m["kappa"] * ug + 1e-300) - 1.0) / (w + 1e-12)
        ok = np.isfinite(corr) & (np.abs(ug) > 1e-6)
        q = np.percentile(np.abs(corr[ok]), [50, 90, 99, 100])
        sat = float((np.abs(corr[ok]) > 0.9 * (3.0 if ck["variant"] == "operator" else abs(cval or 3.0))).mean())
        print(f"  {lb:<5} |corr| p50={q[0]:.3f} p90={q[1]:.3f} p99={q[2]:.3f} "
              f"max={q[3]:.2f}  饱和占比={sat:.2%}")


if __name__ == "__main__":
    labs = ["q10", "q20", "q54"]
    probe("a2q_operator", labs)
    probe("a2q_base", labs)

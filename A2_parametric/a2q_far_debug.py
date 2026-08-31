"""a2q_far_debug.py —— 远场带异常诊断:x_ref 网格范围 + 轴线抽样对照。"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "tools", "refs_a2")

z = np.load(os.path.join(REFS, "ref_a2_q10.npz"))
x, ur = z["x_ref"], z["u_ref"]
print("x_ref shape", x.shape, "u_ref shape", ur.shape)
print("x ranges:", [(float(x[:, i].min()), float(x[:, i].max())) for i in range(3)])
r = np.linalg.norm(x, axis=1)
print("r range:", float(r.min()), float(r.max()))
print("u_ref stats:", float(ur.min()), float(ur.max()), float(np.median(ur)))

# 沿 +x 轴抽样(找最近的网格点)
for xt in [6.0, 10.0, 20.0, 29.0]:
    d = np.linalg.norm(x - np.array([xt, 0, 0]), axis=1)
    i = int(np.argmin(d))
    p = x[i]
    ma = torch.tensor([0.5, 0.5], dtype=torch.float64)
    xs = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
    Ps = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    St = torch.zeros((2, 3), dtype=torch.float64)
    ug = float(physics.guide_u(torch.tensor(p, dtype=torch.float64).unsqueeze(0),
                               ma, xs, Ps, St)[0])
    print(f"p=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) r={np.linalg.norm(p):.2f} "
          f"u_ref={ur[i]:.6f}  u_g={ug:.6f}  比值={ur[i] / ug:.4f}")

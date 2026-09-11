# --- path bootstrap (added during repo restructure: shared code lives in core/ and tools/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while True:
    if _os.path.isdir(_os.path.join(_ROOT, "core")):
        break
    _parent = _os.path.dirname(_ROOT)
    if _parent == _ROOT:
        break
    _ROOT = _parent
for _sub in ("core", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
# -*- coding: utf-8 -*-
"""probe_overfit.py -- opv5 最小过拟合探针(CPU)。

单配置全批、普通 MSE、无归一化/无噪声:若不能把 L 压到 ~0,模型或梯度有 bug;
若能,则问题在训练器的损失归一化/采样动态。
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2q_model as A2

z = np.load(os.path.join(_ROOT, "data", "datasets", "a2q_data_v2_smoke",
                         "refsub_q10.npz"))
c = np.load(os.path.join(_ROOT, "data", "datasets", "a2q_data_v2_smoke",
                         "cfg_q10.npz"))
x = torch.from_numpy(z["x"].astype(np.float64))
ur = torch.from_numpy(z["u"].astype(np.float64))
q, m2 = float(c["q"]), float(c["m2"])
kappa, sq = float(c["kappa"]), float(c["sq"])
wmin, wmax = float(c["wmin"]), float(c["wmax"])

for variant in ("opv5", "opv4", "c2"):
    model = A2.make_model(variant, "cpu").double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ma = torch.tensor([0.5, m2], dtype=torch.float64)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    St = torch.zeros((2, 3), dtype=torch.float64)
    pv = A2.param_vec(q, m2, "cpu")
    hist = []
    for s in range(1, 1501):
        opt.zero_grad()
        u = model(x, ma, xst, Pt, St, pv, kappa, wmin, wmax, sq)
        loss = ((u - ur) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        hist.append(float(loss))
        if s % 300 == 0 or s == 1:
            print(f"[{variant}] step {s:4d} mse={loss.item():.4e} "
                  f"rel={np.sqrt(loss.item()/float((ur**2).mean())):.3e}")
    print(f"[{variant}] first={hist[0]:.3e} last={hist[-1]:.3e} "
          f"min={min(hist):.3e}")

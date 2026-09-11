#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""远场(r>10)的**值**误差,直接用远场参考解 refsub 比较。

动机
----
v6e(PDE 残差项)与 v6f(PDE + 远场值监督)的远场 Hamilton 相对残差几乎相同
(1.172e2 vs 1.171e2),但 v6f 的 ADM/偶极好了 38×/11×。要判断"远场残差
是否被别的因素卡住",需要知道**远场的值误差**到底降了多少:

    R_model = Δu + S = Δ(u_ref + err) + S = R_ref + Δ(err)

故远场残差同时取决于 err 的**大小**与**光滑度**。若 v6f 的 err 明显更小
而残差不变,则瓶颈是光滑度(即 ansatz 的表达能力),不是值精度。

用法::

    python a2_q/diagnostics/far_value_error.py \\
        --runs a2q_v6a,a2q_v6e,a2q_v6f --n 2000 --device cpu
"""
# ---- path bootstrap ----------------------------------------------------
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE,):
    while True:
        if _os.path.isdir(_os.path.join(_p, "core")) and \
           _os.path.isdir(_os.path.join(_p, "a2_q")):
            break
        _np = _os.path.dirname(_p)
        if _np == _p:
            break
        _p = _np
    if _os.path.isdir(_os.path.join(_p, "core")):
        import sys as _sys
        for _sub in ("core", "tools", "a2_q"):
            _q = _os.path.join(_p, _sub)
            if _os.path.isdir(_q) and _q not in _sys.path:
                _sys.path.insert(0, _q)
        _ROOT = _p
        break
# ------------------------------------------------------------------------

import argparse
import os
import statistics as st

import numpy as np
import torch

import a2q_model as A2

RUNS = os.path.join(_ROOT, "data", "runs", "a2")
FAR_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v4far")


def rel_err(model, x, u, t, XS3, PS3, ST3, device, chunk=8192):
    """返回 Σ(u-u_ref)² / Σu_ref²(与训练中 far_loss 同口径)。"""
    num = 0.0
    den = float(np.sum(u ** 2))
    with torch.no_grad():
        for i0 in range(0, len(x), chunk):
            xb = torch.from_numpy(x[i0:i0 + chunk]).double().to(device)
            ub = torch.from_numpy(u[i0:i0 + chunk]).double().to(device)
            ma = torch.tensor([0.5, t["m2"]], dtype=torch.float64,
                              device=device)
            pv = A2.param_vec(t["q"], t["m2"], device)
            out = model(xb, ma, XS3, PS3, ST3, pv, t["kappa"], t["wmin"],
                        t["wmax"], t["sq"])
            num += float(((out - ub) ** 2).sum())
    return num / max(den, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--configs",
                    default="q05,q10,q15,q20,q25,q50,q74,q86,q100")
    ap.add_argument("--n", type=int, default=2000, help="每配置采样点数")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    labels = [s for s in args.configs.split(",") if s]
    runs = [s.strip() for s in args.runs.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    print("远场值误差  Σ(u-u_ref)²/Σu_ref²   (r>10, 每配置 %d 点, device=%s)"
          % (args.n, device))
    print("%-10s %s" % ("run", " ".join("%9s" % lb for lb in labels)))
    for r in runs:
        d = os.path.join(RUNS, r)
        mp = os.path.join(d, "model.pt")
        if not os.path.exists(mp):
            print("%-10s (无 model.pt)" % r)
            continue
        ck = torch.load(mp, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        model = A2.make_model("opv6", device,
                              hidden_neurons=a.get("hidden_neurons", 128),
                              n_basis=a.get("n_basis", 128)).double()
        model.load_state_dict(ck["model_state"])
        model.eval()
        meta = ck["meta"]
        XS3 = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                           device=device)
        PS3 = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                           dtype=torch.float64, device=device)
        ST3 = torch.zeros((2, 3), dtype=torch.float64, device=device)
        row = []
        for lb in labels:
            fp_ = os.path.join(FAR_DIR, "refsub_%s.npz" % lb)
            if not os.path.exists(fp_) or lb not in meta:
                row.append(float("nan"))
                continue
            z = np.load(fp_)
            idx = rng.integers(0, len(z["x"]), args.n)
            x = z["x"][idx].astype(np.float64)
            u = z["u"][idx].astype(np.float64)
            row.append((rel_err(model, x, u, meta[lb], XS3, PS3, ST3,
                                device)) ** 0.5)
        print("%-10s %s" % (r, " ".join("%9.3e" % v for v in row)))
        if row and not all(v != v for v in row):
            g = [v for v in row if v == v]
            print("%-10s 均值 %.3e" % ("", sum(g) / len(g)))


if __name__ == "__main__":
    main()

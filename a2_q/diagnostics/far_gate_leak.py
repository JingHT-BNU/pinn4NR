#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""远场通道向近区"泄漏"多少?——**假设已被本脚本否定**。

背景与假设
----------
opv6 的输出为

    u = κ·u_g·(1 + w·ψ_N + χ_far·m_F) + Δ_F·χ_far

near / branch / trunk / mfar 是**四个互不共享参数的子网**,频率编码
_embed_x / _embed_p 也不含可训练参数(仅做 sin/cos,无 Linear)。
故"训练远场会拉坏近区"不可能来自共享权重。曾假设它来自门控泄漏:

    χ_far(ρ) = σ((ρ − 0.8)/0.25),   ρ = min(r1, r2)

χ_far 在 ρ=0.5 处仍有 σ(−1.2)=0.23,看起来远场通道在峰尖附近并未关断。

实测结果(**假设不成立**)
------------------------
q10 / v6g,20000 点:

| ρ 区间 | \|Δu\| rms(关掉远场通道前后的差) | \|u\| rms | 相对 |
|---|---|---|---|
| [0.50,0.80) | 1.179e-04 | 1.406e-02 | **0.84%** |
| [0.80,1.20) | 1.926e-04 | 1.237e-02 | 1.56% |
| [1.20,2.00) | 1.988e-04 | 9.809e-03 | 2.03% |
| [2.00,3.00) | 1.926e-04 | 7.723e-03 | 2.49% |
| [3.00,5.00) | 1.082e-04 | 5.637e-03 | 1.92% |
| [5.00,10.00) | 6.302e-05 | 3.434e-03 | 1.84% |

远场通道在 ρ<0.8 的贡献仅 **0.84%**,χ_far(0.3)=0.119 —— 门控在峰尖
附近实际上已足够干净。**故帕累托代价不是门控泄漏造成的**,收紧 near_cut
无济于事。

那代价从何而来?
----------------
仍待解释。已补一个此前缺失的对照实验(v6k):从 v6a 出发**只继续训练 2000
步、所有物理项关闭**(pde_w=0),看 ring 区是否也会劣化。
- 若 v6k ≈ v6a → 继续训练本身是中性的,帕累托权衡是真实的;
- 若 v6k 的 ring 也劣化 → 劣化源于继续训练(Adam 在 lr 1e-5 下的漂移),
  与物理项无关,"权衡"的说法需要修正。

用法::

    python a2_q/diagnostics/far_gate_leak.py --run a2q_v6g --config q10
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

import numpy as np
import torch

import a2q_model as A2

RUNS = os.path.join(_ROOT, "data", "runs", "a2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a2q_v6g")
    ap.add_argument("--config", default="q10")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = os.path.join(RUNS, args.run)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                    weights_only=False)
    t = ck["meta"][args.config]
    a = ck.get("args", {})
    model = A2.make_model("opv6", device,
                          hidden_neurons=a.get("hidden_neurons", 128),
                          n_basis=a.get("n_basis", 128)).double()
    model.load_state_dict(ck["model_state"])
    model.eval()
    print("run=%s config=%s  near_cut=%.3f near_width=%.3f"
          % (args.run, args.config, model.near_cut, model.near_width))

    ma = torch.tensor([0.5, float(t["m2"])], dtype=torch.float64,
                      device=device)
    XS3 = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                       device=device)
    PS3 = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                       dtype=torch.float64, device=device)
    ST3 = torch.zeros((2, 3), dtype=torch.float64, device=device)
    pv = A2.param_vec(float(t["q"]), float(t["m2"]), device)

    rng = np.random.default_rng(args.seed)
    xs = torch.from_numpy(rng.uniform(-10, 10, size=(args.n * 4, 3))
                          ).double().to(device)
    rho = torch.minimum((xs - XS3[0]).norm(dim=1),
                        (xs - XS3[1]).norm(dim=1))
    keep = rho > 0.05
    x = xs[keep][:args.n]
    rho = torch.minimum((x - XS3[0]).norm(dim=1),
                        (x - XS3[1]).norm(dim=1))

    def fwd(nc):
        old = model.near_cut
        model.near_cut = nc
        with torch.no_grad():
            out = model(x, ma, XS3, PS3, ST3, pv, float(t["kappa"]),
                        float(t["wmin"]), float(t["wmax"]), float(t["sq"]))
        model.near_cut = old
        return out

    u_full = fwd(model.near_cut)
    u_nofar = fwd(1.0e3)                     # χ_far ≈ 0 → 关掉远场通道
    d = (u_full - u_nofar).abs()
    ua = u_full.abs()
    rho_n = rho.cpu().numpy()
    d_n = d.cpu().numpy()
    u_n = ua.cpu().numpy()

    # χ_far 参考
    print("\nχ_far(ρ) = σ((ρ−%.3f)/%.3f):  ρ=0.3→%.4f  ρ=0.5→%.4f  "
          "ρ=0.8→%.4f  ρ=1.0→%.4f  ρ=1.5→%.4f  ρ=2.0→%.4f"
          % (model.near_cut, model.near_width,
             float(torch.sigmoid(torch.tensor((0.3 - model.near_cut)
                                              / model.near_width))),
             float(torch.sigmoid(torch.tensor((0.5 - model.near_cut)
                                              / model.near_width))),
             float(torch.sigmoid(torch.tensor((0.8 - model.near_cut)
                                              / model.near_width))),
             float(torch.sigmoid(torch.tensor((1.0 - model.near_cut)
                                              / model.near_width))),
             float(torch.sigmoid(torch.tensor((1.5 - model.near_cut)
                                              / model.near_width))),
             float(torch.sigmoid(torch.tensor((2.0 - model.near_cut)
                                              / model.near_width)))))

    print("\n=== 远场通道贡献 |Δ_F·χ_far + κu_g·w·χ_far·m_F| 的分布 ===")
    print("%-14s %8s %12s %12s %10s" % ("ρ 区间", "点数",
                                        "|Δu| rms", "|u| rms", "相对"))
    edges = [0.05, 0.2, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 10.0]
    for i in range(len(edges) - 1):
        m = (rho_n >= edges[i]) & (rho_n < edges[i + 1])
        if m.sum() < 5:
            continue
        dr = float(np.sqrt(np.mean(d_n[m] ** 2)))
        ur = float(np.sqrt(np.mean(u_n[m] ** 2)))
        print("%-14s %8d %12.3e %12.3e %9.2f%%"
              % ("[%.2f,%.2f)" % (edges[i], edges[i + 1]), m.sum(), dr, ur,
                 100 * dr / max(ur, 1e-300)))

    # 关键:ring 区(ρ<0.5)的泄漏占比
    ring = rho_n < 0.5
    if ring.sum() > 5:
        print("\nring 区(ρ<0.5): 远场通道贡献 |Δu| rms = %.3e"
              % float(np.sqrt(np.mean(d_n[ring] ** 2))))
        print("  该区内 χ_far 最大值 = %.4f (ρ=0.5 处)"
              % float(torch.sigmoid(torch.tensor(
                  (0.5 - model.near_cut) / model.near_width))))


if __name__ == "__main__":
    main()

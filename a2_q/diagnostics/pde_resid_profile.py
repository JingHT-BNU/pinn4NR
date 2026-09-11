#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hamilton 残差项(--pde-*)的权重结构诊断。

回答的问题
----------
训练里的 PDE 残差项是::

    l_pde = Σ_j w_j (R_j / sig)² / Σ_j w_j

权重 ∝ w·R²。**关键点是 R 在哪个半径最大,而不是 S 在哪个半径最大。**

曾经(2026-09-11 初版)本类的脚本把上式中的 R 误用源项 S 代入来"分析权重
结构",据此得出"远场只占 0.2%、远场未被约束"的结论,并据此给训练器加了
`--pde-norm rel`。改用**真实残差** R 复测后该结论被推翻:

  · `batch` 归一化下 r0≥14 的远场已占 l_pde 的 **98.97%**;
  · 即便把远场放大上限从 1 拉到 1e4,远场份额也只从 99.1% 变到 99.9%。

原因是 |R| 在远场**最大**(q10/v6a:r0≥22 处 rms 2.9e-4,r0≈10 处 5.7e-5),
虽然 S 按 r^{-6} 衰减,但 Δu 的误差增长得更快。所以远场精度上不去是
**收敛速度与批间方差**的问题,不是损失权重的问题。

本脚本用当前模型输出真实 R = Δu + S,按 r0 壳层给出 S、|R| 与损失份额。

用法::

    python a2_q/diagnostics/pde_resid_profile.py --config q10
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

import physics
import a2q_model as A2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/runs/a2/a2q_v6a")
    ap.add_argument("--config", default="q10")
    ap.add_argument("--rmax", type=float, default=28.0)
    ap.add_argument("--rho-min", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--r-start", type=float, default=6.0)
    ap.add_argument("--r-end", type=float, default=26.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.4,
                    help="期望的 r0>=14 远场份额(0~1)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(
        _ROOT, args.run)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                    weights_only=False)
    meta = ck["meta"]
    t = meta[args.config]
    q, m2, kappa = float(t["q"]), float(t["m2"]), float(t["kappa"])
    sq, wmin, wmax = float(t["sq"]), float(t["wmin"]), float(t["wmax"])
    print("run=%s  config=%s  q=%.4g  kappa=%.5f" % (run_dir, args.config, q,
                                                     kappa))

    a = ck.get("args", {})
    model = A2.make_model("opv6", device, hidden_neurons=a.get("hidden_neurons",
                                                               128),
                          n_basis=a.get("n_basis", 128)).double()
    model.load_state_dict(ck["model_state"])
    model.eval()

    ma = torch.tensor([0.5, m2], dtype=torch.float64, device=device)
    XS3 = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                       device=device)
    PS3 = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                       dtype=torch.float64, device=device)
    ST3 = torch.zeros((2, 3), dtype=torch.float64, device=device)
    XA3 = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64, device=device)

    rng = np.random.default_rng(args.seed)
    xs = torch.from_numpy(
        rng.uniform(-args.rmax, args.rmax, size=(max(4 * args.n, 512), 3))
    ).double().to(device)
    rr = torch.minimum((xs - XA3).norm(dim=1), (xs + XA3).norm(dim=1))
    r0 = xs.norm(dim=1)
    x = xs[(rr > args.rho_min) & (r0 <= args.rmax)][:args.n]
    x.requires_grad_(True)
    r0x = x.norm(dim=1)

    pv = A2.param_vec(q, m2, device)
    u = model(x, ma, XS3, PS3, ST3, pv, kappa, wmin, wmax, sq)
    psi_s = physics.psi_sing(x, ma, XS3)
    kk = physics.bowen_york_KK(x, ma, XS3, PS3, ST3)
    R = physics.pde_residual(u, x, psi_s, kk)
    with torch.no_grad():
        psi = torch.clamp(psi_s + u, min=1e-4)
        S = (1.0 / 8.0) * kk / psi ** 7

    R_np = R.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    r_np = r0x.detach().cpu().numpy()

    span = max(args.r_end - args.r_start, 1e-9)
    w = np.clip((r_np - args.r_start) / span, 0.0, 1.0)

    # --- 复刻训练中的壳层标定(用 ψ_sing, 与模型无关) ---
    K = 12
    edges = np.geomspace(args.rho_min, args.rmax, K + 1)
    prof = np.full(K, np.nan)
    for k in range(K):
        m = (r_np >= edges[k]) & (r_np < edges[k + 1])
        if m.sum() >= 8:
            prof[k] = float(np.sqrt(np.mean(S_np[m] ** 2)))
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(K), np.flatnonzero(ok), prof[ok])
    S_ref = float(np.sqrt(np.mean(S_np ** 2)))
    print("\nS_ref = %.4e" % S_ref)
    print("%-14s %10s %12s %12s" % ("r0 壳层", "点数", "S_shell", "|R| rms"))
    for k in range(K):
        m = (r_np >= edges[k]) & (r_np < edges[k + 1])
        if m.sum() == 0:
            continue
        print("%-14s %10d %12.3e %12.3e"
              % ("[%.2f,%.2f)" % (edges[k], edges[k + 1]), m.sum(),
                 prof[k], np.sqrt(np.mean(R_np[m] ** 2))))

    kbin = np.clip(np.searchsorted(edges, r_np, side="right") - 1, 0, K - 1)
    m_ref = r_np >= 14.0

    def shares(eps):
        S_sh = prof[kbin]
        sig = np.sqrt(S_sh ** 2 + (eps * S_ref) ** 2)
        term = w * (R_np / sig) ** 2
        tot = term.sum()
        return term / max(tot, 1e-300), term

    print("\n=== 对照表:不同 eps 下的远场份额(远场 = r0>=14) ===")
    print("%8s %10s %14s %12s" % ("eps", "放大上限", "远场份额", "l_pde 量级"))
    best = None
    for eps in (1.0, 0.5, 0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01):
        sh, term = shares(eps)
        far = sh[m_ref].sum()
        print("%8.3g %10.1e %13.1f%% %12.3e"
              % (eps, 1.0 / eps ** 2, 100 * far, term.mean()))
        if best is None or abs(far - args.target) < abs(best[1] - args.target):
            best = (eps, far)
    print("\n→ 使远场份额最接近 %.0f%% 的 eps ≈ %.3g (实测份额 %.1f%%)"
          % (100 * args.target, best[0], 100 * best[1]))

    # 对照:旧 batch 归一
    sigb = max(S_ref, 1e-3 * sq / 9.0)
    termb = w * (R_np / sigb) ** 2
    print("参照 --pde-norm batch(训练默认): 远场份额 %.2f%%   l_pde 量级 %.3e"
          % (100 * termb[m_ref].sum() / max(termb.sum(), 1e-300),
             termb.mean()))


if __name__ == "__main__":
    main()

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
"""
far_spectrum.py —— 远场误差的空间结构诊断
==========================================

回答:模型与参考解在远场的偏离,是**平滑的大尺度结构**还是**高频振荡**?

方法:沿若干条射线取样 r∈[r0,r1],直接比较 u_θ 与谱参考解 u_ref,报告

  - |u_θ − u_ref| 的均方根(误差幅值);
  - 二阶径向导数 d²(u_θ − u_ref)/dr² 的均方根(误差曲率);
  - 该处源项 |S| 的均方根(物理上 Δu 应当等于 −S 的量级);
  - 去趋势后的主导波长(FFT)。

判读:
  - 曲率 ≫ |S| → Hamilton 约束在该处失衡(残差被误差曲率主导);
  - 主导波长接近射线长度 → 误差是平滑大尺度,可用训练消除;
  - 主导波长远小于 1 → 高频抖动,须改动编码或后处理。

用法:
    python a2_q/diagnostics/far_spectrum.py --run data/runs/a2/a2q_v6a
    python a2_q/diagnostics/far_spectrum.py --run ... --configs q10,q100
"""
import argparse
import os
import sys

import numpy as np
import torch

import physics
import a2q_model as A2

XS = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
PS = np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])


def load_run(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location=device,
                    weights_only=False)
    a = ck.get("args", {})
    m = A2.make_model("opv6", device,
                      hidden_neurons=a.get("hidden_neurons", 128),
                      n_basis=a.get("n_basis", 128))
    m.load_state_dict(ck["model_state"])
    m.double().eval()
    return m, ck["meta"]


def u_model(model, meta, lb, pts, device, chunk=4096):
    t = meta[lb]
    ma = torch.tensor([0.5, t["m2"]], dtype=torch.float64, device=device)
    xs = torch.tensor(XS, dtype=torch.float64, device=device)
    Ps = torch.tensor(PS, dtype=torch.float64, device=device)
    Ss = torch.zeros((2, 3), dtype=torch.float64, device=device)
    p = torch.tensor([[np.log10(t["q"]), t["m2"] / 5.0]],
                     dtype=torch.float64, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            xb = torch.tensor(pts[i:i + chunk], dtype=torch.float64,
                              device=device)
            out.append(model(xb, ma, xs, Ps, Ss, p, float(t["kappa"]),
                             float(t["wmin"]), float(t["wmax"]),
                             float(t["sq"])).cpu().numpy())
    return np.concatenate(out)


def find_ref(lb):
    for d in ("a2v2", "a2"):
        for pat in (f"ref_{lb}.npz", f"ref_a2_{lb}.npz"):
            p = os.path.join(_ROOT, "data", "refs", d, pat)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--configs", default="q10,q50,q100")
    ap.add_argument("--r0", type=float, default=10.0, help="射线起点半径")
    ap.add_argument("--r1", type=float, default=27.0, help="射线终点半径")
    ap.add_argument("--n-ray", type=int, default=2001)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    import spectral_reference as sr

    model, meta = load_run(args.run, device)
    rays = (("+x", np.array([1.0, 0.0, 0.0])),
            ("+z", np.array([0.0, 0.0, 1.0])),
            ("diag", np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)))

    print(f"run = {args.run}")
    print("%-6s %-5s %11s %11s %11s %11s %10s" %
          ("cfg", "ray", "|diff|rms", "|d2diff|rms", "|S|rms", "|diff|max",
           "主波长"))
    for lb in args.configs.split(","):
        lb = lb.strip()
        if lb not in meta:
            print(f"  跳过 {lb}(不在该 run 的配置表中)")
            continue
        src = find_ref(lb)
        if src is None:
            print(f"  跳过 {lb}(无谱参考解)")
            continue
        solver = sr.SpectralPunctureSolver.from_coefficients(
            src, device=str(device), verify=False)
        t = meta[lb]
        for name, dvec in rays:
            r = np.linspace(args.r0, args.r1, args.n_ray)
            pts = r[:, None] * dvec[None, :]
            um = u_model(model, meta, lb, pts, device)
            ur = solver.evaluate(pts, chunk=32768,
                                 dtype=torch.float64).astype(np.float64)
            diff = um - ur
            xt = torch.tensor(pts, dtype=torch.float64)
            kk = physics.bowen_york_KK(
                xt, torch.tensor([0.5, t["m2"]]), torch.tensor(XS),
                torch.tensor(PS), torch.zeros(2, 3)).numpy()
            r1_ = np.linalg.norm(pts - XS[0], axis=1)
            r2_ = np.linalg.norm(pts - XS[1], axis=1)
            psi_s = 1.0 + 0.5 / (2.0 * r1_) + t["m2"] / (2.0 * r2_)
            S = 0.125 * kk / (psi_s + um) ** 7
            h = r[1] - r[0]
            d2 = np.gradient(np.gradient(diff, h), h)
            # FFT:去二次趋势 + 汉宁窗
            x = diff - np.polyval(np.polyfit(r, diff, 2), r)
            F = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            k = np.arange(len(F))
            lam = (r[-1] - r[0]) / np.maximum(k[1:], 1)
            ipk = 1 + int(np.argmax(F[1:]))
            print("%-6s %-5s %11.3e %11.3e %11.3e %11.3e %10.2f" %
                  (lb, name, np.sqrt(np.mean(diff ** 2)),
                   np.sqrt(np.mean(d2 ** 2)), np.sqrt(np.mean(S ** 2)),
                   np.max(np.abs(diff)),
                   lam[ipk - 1] if ipk - 1 < len(lam) else 0.0))


if __name__ == "__main__":
    main()

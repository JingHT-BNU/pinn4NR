r"""大 q 峰区诊断:沿两奇点连线取剖面,比较模型与谱参考解的峰形。

用法::
    python a2_q/diagnostics/needle_profile.py --run a2q_v6l --config tq100
"""
# ---- path bootstrap ----------------------------------------------------
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_p = _HERE
while True:
    if _os.path.isdir(_os.path.join(_p, "core")) and \
       _os.path.isdir(_os.path.join(_p, "a2_q")):
        break
    _np = _os.path.dirname(_p)
    if _np == _p:
        raise SystemExit("找不到仓库根")
    _p = _np
import sys as _sys
_ROOT = _p
for _sub in ("core", "tools", "a2_q"):
    _q = _os.path.join(_p, _sub)
    if _os.path.isdir(_q) and _q not in _sys.path:
        _sys.path.insert(0, _q)
# ------------------------------------------------------------------------

import argparse
import os
import sys
import numpy as np
import torch

import a2q_model as A2
import physics

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="a2q_v6l")
ap.add_argument("--config", default="tq100")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

device = torch.device(args.device if args.device != "auto"
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

run_dir = os.path.join(_ROOT, "data", "runs", "a2", args.run)
ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                weights_only=False)
a = ck["args"]
model = A2.make_model(a.get("variant", "opv6"), device,
                      hidden_neurons=a.get("hidden_neurons", 128),
                      n_basis=a.get("n_basis", 128)).double()
model.load_state_dict(ck["model_state"])
model.eval()

lb = args.config
t = ck["meta"][lb]
ma = torch.tensor([0.5, t["m2"]], dtype=torch.float64, device=device)
pv = A2.param_vec(t["q"], t["m2"], device)
XS = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                  device=device)
PS = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                  device=device)
ST = torch.zeros((2, 3), dtype=torch.float64, device=device)

sys.path.insert(0, os.path.join(_ROOT, "tools"))
import spectral_reference as sr
_c = os.path.join(_ROOT, "data", "refs", "a2", "ref_a2_%s.npz" % lb)
if not os.path.exists(_c):
    _c = os.path.join(_ROOT, "data", "refs", "a2", "ref_%s.npz" % lb)
ref_npz = _c
ev = sr.SpectralPunctureSolver.from_coefficients(ref_npz, device=device,
                                                 verify=False)

# 沿 x 轴剖面(避开奇点 0.05)
xs = np.linspace(-9.9, 9.9, 2000)
xs = xs[np.abs(np.abs(xs) - 3.0) > 0.05]
pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)

with torch.no_grad():
    x_t = torch.from_numpy(pts).double().to(device)
    u_m = model(x_t, ma, XS, PS, ST, pv, t["kappa"], t["wmin"],
                t["wmax"], t["sq"]).cpu().numpy()
u_r = np.asarray(ev.evaluate(pts, chunk=32768, dtype=torch.float64),
                 dtype=np.float64)
err = u_m - u_r

# 各奇点附近的峰:|x-3|<0.5 与 |x+3|<0.5
for name, xc in (("小黑洞 x=+3 (m1=0.5)", 3.0),
                 ("大黑洞 x=-3 (m2=%.2f)" % t["m2"], -3.0)):
    m = np.abs(xs - xc) < 0.5
    if not m.any():
        continue
    im = np.argmax(np.abs(u_r[m]))
    print("%s:" % name)
    print("  ref  峰 |u| = %.4e   model 峰 |u| = %.4e   比值 %.4f"
          % (np.max(np.abs(u_r[m])), np.max(np.abs(u_m[m])),
             np.max(np.abs(u_m[m])) / np.max(np.abs(u_r[m]))))
    print("  峰区 ref rms=%.3e  err rms=%.3e  相对=%.3e"
          % (np.sqrt(np.mean(u_r[m] ** 2)), np.sqrt(np.mean(err[m] ** 2)),
             np.sqrt(np.mean(err[m] ** 2)) / np.sqrt(np.mean(u_r[m] ** 2))))

# 峰形表:沿轴逐点打印(小黑洞侧)
print()
print("小黑洞侧剖面 (x, u_ref, u_model, err):")
sel = (xs > 2.0) & (xs < 4.0)
idx = np.linspace(0, sel.sum() - 1, 18).astype(int)
for i in np.where(sel)[0][idx]:
    print("  x=%+.4f  ref=%+.5e  model=%+.5e  err=%+.3e"
          % (xs[i], u_r[i], u_m[i], err[i]))

# 全域误差随 d(到最近奇点)的分布
rng = np.random.default_rng(3)
n = 40000
xb = torch.from_numpy(rng.uniform(-10, 10, size=(n, 3))).double().to(device)
with torch.no_grad():
    ub = model(xb, ma, XS, PS, ST, pv, t["kappa"], t["wmin"],
               t["wmax"], t["sq"]).cpu().numpy()
urb = np.asarray(ev.evaluate(xb.cpu().numpy(), chunk=32768,
                             dtype=torch.float64), dtype=np.float64)
eb = ub - urb
db = np.minimum(np.linalg.norm(xb.cpu().numpy() - XS[0].cpu().numpy(), axis=1),
                np.linalg.norm(xb.cpu().numpy() - XS[1].cpu().numpy(), axis=1))
print()
print("全域盒内误差 vs 到最近奇点距离 d:")
edges = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (db >= lo) & (db < hi)
    if m.sum() < 10:
        continue
    print("  d in [%4.2f,%5.2f)  n=%6d  |err|rms=%.3e  |u_ref|rms=%.3e  相对=%.3e"
          % (lo, hi, m.sum(), np.sqrt(np.mean(eb[m] ** 2)),
             np.sqrt(np.mean(urb[m] ** 2)),
             np.sqrt(np.mean(eb[m] ** 2)) / np.sqrt(np.mean(urb[m] ** 2))))

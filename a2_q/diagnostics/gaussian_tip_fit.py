r"""离线验证:单高斯凸起能否表达 v6l 在 tq100 针尖的亏损。

若 D(x) = u_ref - u_model 在小黑洞近域可被 a*exp(-(r/sigma)^2) 很好拟合,
则 opv8 的解析凸起基底(可学习振幅/宽度)是正确方向。
"""
# ---- path bootstrap ----------------------------------------------------
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_p = _HERE
while True:
    if _os.path.isdir(_os.path.join(_p, "core")) and \
       _os.path.isdir(_os.path.join(_p, "a2_q")):
        break
    _p = _os.path.dirname(_p)
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
ev = sr.SpectralPunctureSolver.from_coefficients(_c, device=device,
                                                 verify=False)

# 针尖带采样:小黑洞周围 r 0.05-0.8
rng = np.random.default_rng(1)
n = 30000
r = rng.uniform(0.05, 0.8, n)
ct = rng.uniform(-1, 1, n)
ph = rng.uniform(0, 2 * np.pi, n)
st = np.sqrt(1 - ct ** 2)
pts = np.stack([3 + r * st * np.cos(ph), r * st * np.sin(ph), r * ct],
               axis=1)
with torch.no_grad():
    x_t = torch.from_numpy(pts).double().to(device)
    u_m = model(x_t, ma, XS, PS, ST, pv, t["kappa"], t["wmin"],
                t["wmax"], t["sq"]).cpu().numpy()
u_r = np.asarray(ev.evaluate(pts, chunk=32768, dtype=torch.float64),
                 dtype=np.float64)
D = u_r - u_m                       # 亏损(模型需补上的量)
ri = np.linalg.norm(pts - np.array([3.0, 0, 0]), axis=1)

# 拟合 D ≈ a*(1 + d*nx)*exp(-(r/s)^2) + c, 线性最小二乘(对 a, a*d, c), 扫 s
# nx = 从小黑洞指向场点的单位矢量的 x 分量(一阶角调制)
nx = (pts[:, 0] - 3.0) / np.maximum(ri, 1e-12)
print("拟合 D = a*(1+d*nx)*exp(-(r/s)^2) + c   (n=%d)" % n)
best = None
for s in np.geomspace(0.05, 1.0, 40):
    G = np.stack([np.exp(-(ri / s) ** 2),
                  nx * np.exp(-(ri / s) ** 2),
                  np.ones_like(ri)], axis=1)
    coef, res, *_ = np.linalg.lstsq(G, D, rcond=None)
    pred = G @ coef
    rms = float(np.sqrt(np.mean((D - pred) ** 2)))
    if best is None or rms < best[0]:
        best = (rms, s, coef)
rms, s, (a_fit, ad_fit, c_fit) = best
d_fit = ad_fit / a_fit
D_rms = float(np.sqrt(np.mean(D ** 2)))
print("最优: sigma=%.4f  a=%.4e  d=%.4f  c=%.4e" % (s, a_fit, d_fit, c_fit))
print("亏损 D rms = %.4e   拟合残差 rms = %.4e   (解释 %.1f%%)"
      % (D_rms, rms, 100 * (1 - rms ** 2 / D_rms ** 2)))

# 加上凸起后的峰高比(沿 x 轴)
xs = np.linspace(2.2, 3.9, 1700)
xs = xs[np.abs(xs - 3.0) > 0.05]
axial = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
with torch.no_grad():
    u_ax = model(torch.from_numpy(axial).double().to(device), ma, XS, PS,
                 ST, pv, t["kappa"], t["wmin"], t["wmax"],
                 t["sq"]).cpu().numpy()
u_ax_ref = np.asarray(ev.evaluate(axial, chunk=32768, dtype=torch.float64),
                      dtype=np.float64)
rax = np.abs(xs - 3.0)
bump = a_fit * (1 + d_fit * (xs - 3.0) / np.maximum(rax, 1e-12)) \
    * np.exp(-(rax / s) ** 2) + c_fit
h0 = np.max(np.abs(u_ax)) / np.max(np.abs(u_ax_ref))
h1 = np.max(np.abs(u_ax + bump)) / np.max(np.abs(u_ax_ref))
print("沿轴峰高比:  v6l=%.4f   +高斯=%.4f   ref=%.4e"
      % (h0, h1, np.max(np.abs(u_ax_ref))))
e0 = u_ax - u_ax_ref
e1 = e0 - bump
print("峰区(|x-3|<0.5) err rms:  v6l=%.3e  +高斯=%.3e"
      % (np.sqrt(np.mean(e0 ** 2)), np.sqrt(np.mean(e1 ** 2))))

# 纯径向(无角调制)对照
G0 = np.stack([np.exp(-(ri / s) ** 2), np.ones_like(ri)], axis=1)
coef0, *_ = np.linalg.lstsq(G0, D, rcond=None)
rms0 = float(np.sqrt(np.mean((D - G0 @ coef0) ** 2)))
print("对照(无角调制): 解释 %.1f%%" % (100 * (1 - rms0 ** 2 / D_rms ** 2)))

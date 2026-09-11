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
"""smoke_train_v4.py -- a2q_train_v4.py 冒烟:合成数据 2 配置 × 3 变体 × 60 步。

验证:数据加载/加权采样/参数噪声/三 ansatz 前向反传/续训指纹路径无运行时错误,
L_ref 量级下降。不验证拟合能力(真实数据另行)。
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics

SMOKE = os.path.join(_ROOT, "data", "datasets", "a2q_data_v2_smoke")
PY = sys.executable


def make_fake():
    if os.path.isdir(SMOKE):
        shutil.rmtree(SMOKE)
    os.makedirs(SMOKE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(3)
    for q, lb in ((1.0, "q10"), (1.5, "q15")):
        m2 = 0.5 * q
        ma = torch.tensor([0.5, m2], dtype=torch.float64, device=device)
        xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                           device=device)
        Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
                          dtype=torch.float64, device=device)
        St = torch.zeros((2, 3), dtype=torch.float64, device=device)
        n = 1024
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x *= (10.0 * rng.random(n) ** (1 / 3))[:, None]
        xt = torch.from_numpy(x).double().to(device)
        with torch.no_grad():
            ug = physics.guide_u(xt, ma, xst, Pt, St).cpu().numpy()
        kappa = float((ug ** 2).sum() ** 0.5) * 0 + 0.5  # 任意常数
        # 合成"参考解":κ·u_g + 一个光滑各向异性畸变(网络需拟合)
        distort = 0.05 * np.exp(-np.linalg.norm(x - np.array([3.0, 0, 0]),
                                                axis=1) ** 2 / 4.0)
        u = kappa * ug + distort * ug
        sq = float(np.sqrt(np.mean(ug ** 2)) + 1e-30)
        np.savez(os.path.join(SMOKE, f"refsub_{lb}.npz"),
                 x=x.astype(np.float32), u=u, w=np.ones(n))
        np.savez(os.path.join(SMOKE, f"cfg_{lb}.npz"),
                 q=q, m1=0.5, m2=m2, kappa=kappa, sq=sq,
                 wmin=float(ug.min()), wmax=float(ug.max()),
                 x_int=x.astype(np.float32))
    print("fake data ready:", SMOKE)


def run_variant(variant):
    out = os.path.join(_ROOT, "data", "runs", "a2", f"smoke_{variant}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    r = subprocess.run([PY, "-u", os.path.join(HERE, "a2q_train_v4.py"),
                        "--variant", variant, "--exp-name", f"smoke_{variant}",
                        "--steps", "300", "--pts-per-cfg", "256",
                        "--data-dir", SMOKE, "--heldout", "q15"],
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    hist_path = os.path.join(out, "history.json")
    ok = r.returncode == 0 and os.path.exists(hist_path)
    if ok:
        h = json.load(open(hist_path))["L_ref"]
        ok = h[-1] < 0.7 * h[0] and np.isfinite(h).all()
        print(f"[{variant}] L_ref {h[0]:.3e} -> {h[-1]:.3e} "
              f"{'OK' if ok else 'NOT DECREASING'}")
    else:
        print(f"[{variant}] FAILED rc={r.returncode}\n{tail}")
    return ok


if __name__ == "__main__":
    make_fake()
    good = all(run_variant(v) for v in ("opv5", "opv4", "c2"))
    print("SMOKE", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)

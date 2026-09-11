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
"""post_refsub_v3.py -- refsub_v3:远场加权重采样(R4,针对 G1 global/far 瓶颈)。

诊断(2026-09-02):v4b 容量×2.5 后 global/far 纹丝不动(3.072e-2 vs 3.061e-2@q1)
→ 瓶颈是损失权重结构而非容量:far(r0>5)相对误差是 global 唯一显著来源
(峰/环/谷全 ≤1e-2),但 refsub_v2 里远场损失权重占比低,且相对 L2 的分母
被峰区 u² 淹没,远场梯度信号过弱。误差面板(q50/q100)显示远场误差为
大尺度光滑瓣状结构,可拟合。

v3 采样(每配置):
  球内 r<10 ×8192(w=1)+ 立方体远场 r0>5 ×8192(w=4,与 eval2 far 对齐,
  覆盖球采样够不到的立方体角区)+ 球面 r=10 ×2048(w=1)
  + 双孔近峰壳 r∈[0.1,1.5] ×2×4096(w=2)+ 谷区 r<1.5 ×2048(w=1)
cfg_*.npz 直接复制 v2(κ/sq/wmin/wmax 不变)。幂等:已有 refsub 跳过。

用法: python post_refsub_v3.py
"""
import logging
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools"))
from logutil import setup_logging
from data import sample_ball, sample_sphere_surface
from post_refs_v2 import label_src_map, RCUT, R_MAX
from spectral_reference import SpectralPunctureSolver

log = logging.getLogger("paper.A2.post_refsub_v3")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v2")
DST_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v3")


def sample_far_cube(n, rng):
    """立方体 [-10,10]³ 均匀采样,拒绝法取 r0=min(r1,r2)>5(接受率约 87%)。"""
    out = []
    got = 0
    while got < n:
        m = max(int((n - got) * 1.3), 1024)
        x = rng.uniform(-R_MAX, R_MAX, size=(m, 3))
        r1 = np.linalg.norm(x - np.array([3.0, 0, 0]), axis=1)
        r2 = np.linalg.norm(x - np.array([-3.0, 0, 0]), axis=1)
        keep = np.minimum(r1, r2) > 5.0
        out.append(x[keep])
        got += int(keep.sum())
    return np.concatenate(out, axis=0)[:n]


def make_refsub_v3(ev, lb, device):
    rng = np.random.default_rng(888)
    parts, wts = [], []
    xb = sample_ball(8192, R_MAX, rng).astype(np.float64)
    parts.append(xb)
    wts.append(np.ones(len(xb)))
    xf = sample_far_cube(8192, rng)
    parts.append(xf)
    wts.append(np.full(len(xf), 4.0))
    xs_ = sample_sphere_surface(2048, R_MAX, rng).astype(np.float64)
    parts.append(xs_)
    wts.append(np.ones(len(xs_)))
    d = rng.normal(size=(4096, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = (RCUT ** 3 + (1.5 ** 3 - RCUT ** 3) *
         rng.random(4096)) ** (1.0 / 3.0)
    shell = d * r[:, None]
    for c in ((3.0, 0, 0), (-3.0, 0, 0)):
        parts.append(shell + np.asarray(c))
        wts.append(np.full(len(shell), 2.0))
    dv = rng.normal(size=(2048, 3))
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    parts.append(dv * (1.5 * rng.random(2048) ** (1.0 / 3.0))[:, None])
    wts.append(np.ones(2048))
    pts = np.concatenate(parts, axis=0)
    w = np.concatenate(wts, axis=0)
    u = np.asarray(ev.evaluate(pts, chunk=32768, dtype=torch.float64))
    np.savez(os.path.join(DST_DIR, f"refsub_{lb}.npz"),
             x=pts.astype(np.float32), u=u.astype(np.float64),
             w=w.astype(np.float64))


def main():
    setup_logging("A2", "post_refsub_v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DST_DIR, exist_ok=True)
    srcs = label_src_map()
    log.info("共 %d 配置;输出 %s", len(srcs), DST_DIR)
    t0 = time.time()
    n_new = 0
    for i, (lb, src) in enumerate(sorted(srcs.items()), 1):
        dst = os.path.join(DST_DIR, f"refsub_{lb}.npz")
        cfg_dst = os.path.join(DST_DIR, f"cfg_{lb}.npz")
        if not os.path.exists(cfg_dst):
            shutil.copy(os.path.join(SRC_DIR, f"cfg_{lb}.npz"), cfg_dst)
        if os.path.exists(dst):
            continue
        z = np.load(os.path.join(SRC_DIR, f"cfg_{lb}.npz"))
        m2 = float(z["masses"][1])
        ev = SpectralPunctureSolver.from_coefficients(src, device=str(device),
                                                      verify=False)
        make_refsub_v3(ev, lb, device)
        n_new += 1
        if i % 10 == 0:
            log.info("  %d/%d (新增 %d, %.1f min)", i, len(srcs), n_new,
                     (time.time() - t0) / 60)
    log.info("完成:68 配置 refsub_v3,新增 %d,用时 %.1f min",
             n_new, (time.time() - t0) / 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("refsub_v3 生成失败")
        raise

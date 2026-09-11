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
"""post_refs_v2.py -- a2v2 参考解后处理:refsub_v2 + cfg_v2 + κ*_spec_v2。

与 post_refs_opv3.py 的差别(2026-09-01 新口径):
  - 参考解来源:优先 data/refs/a2v2(新解),回退 data/refs/a2(旧 L48);
  - κ*_spec 在新口径窗口([-10,10]³,N=161,rcut=0.1)上最小二乘,并与
    eval2 的参考网格缓存共用同一次求值(data/refs/a2v2/evalcache/w10_n161);
  - refsub_v2 近峰加密+逐点权重(指令 #1"重视峰值处误差"):
      球内 r<10 ×8192(w=1)+ 球面 r=10 ×2048(w=1)
      + 双孔近峰壳 r∈[0.1,1.5] ×2×4096(w=2)+ 谷区 r<1.5 ×2048(w=1)
  - cfg_v2 的 sq/wmin/wmax 按 R=10 窗口重算。
幂等:已有输出跳过;--force 重算。

用法: python post_refs_v2.py [--force]
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools"))
from logutil import setup_logging
from data import sample_ball, sample_sphere_surface
import physics
from spectral_reference import SpectralPunctureSolver

log = logging.getLogger("paper.A2.post_refs_v2")

HERE = os.path.dirname(os.path.abspath(__file__))
REFS2 = os.path.join(_ROOT, "data", "refs", "a2v2")
REFS1 = os.path.join(_ROOT, "data", "refs", "a2")
DATA_DIR = os.path.join(_ROOT, "data", "datasets", "a2q_data_v2")
M1 = 0.5
RCUT = 0.1
GRID_N = 81   # κ*_spec_v2 拟合网格([-10,10]³,53 万点;κ 统计量对网格分辨率不敏感,
              # N=161 的细网格缓存留给 eval2 按需构建)
R_MAX = 10.0
CACHE_DIR = os.path.join(REFS2, "evalcache", f"w{R_MAX:g}_n{GRID_N}")


def label_src_map():
    """label -> 参考解 npz 路径(a2v2 优先,a2 回退)。"""
    out = {}
    for d, strip in ((REFS2, "ref_"), (REFS1, "ref_a2_")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz") or not fn.startswith("ref_"):
                continue
            lb = fn[len(strip):-4] if fn.startswith(strip) else fn[4:-4]
            p = os.path.join(d, fn)
            if fn.startswith("ref_tq"):
                lb = fn[4:-4]
            out.setdefault(lb, p)
    return out


def grid_cache(src, lb, device):
    """[-10,10]³ N=161 全网格 u_ref(缓存共用,供 κ 拟合与 eval2)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dst = os.path.join(CACHE_DIR, f"ref_{lb}.npz")
    if os.path.exists(dst):
        return np.load(dst)["u_ref"]
    ev = SpectralPunctureSolver.from_coefficients(src, device=str(device),
                                                  verify=False)
    g = np.linspace(-R_MAX, R_MAX, GRID_N)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    u_full = np.empty(len(pts), dtype=np.float32)
    B = 32768   # 谱求值显存 ~chunk×K×16B;262144 在 16GB 卡上 OOM
    t0 = time.time()
    for i in range(0, len(pts), B):
        u_full[i:i + B] = ev.evaluate(pts[i:i + B], chunk=B,
                                      dtype=torch.float64).astype(np.float32)
    np.savez(dst, u_ref=u_full, grid_n=GRID_N, rmax=R_MAX, rcut=RCUT,
             src=os.path.relpath(src, HERE))
    log.info("  网格缓存 %s (%.0fs)", lb, time.time() - t0)
    return u_full


def guide_on_grid(lb, q, m2, device):
    """u_g 在同网格上的值(逐块解析求值)。"""
    g = np.linspace(-R_MAX, R_MAX, GRID_N)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ma = torch.tensor([M1, m2], dtype=torch.float64, device=device)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                       device=device)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                      device=device)
    St = torch.zeros((2, 3), dtype=torch.float64, device=device)
    ug = np.empty(len(pts))
    with torch.no_grad():
        for c0 in range(0, len(pts), 262144):
            xt = torch.from_numpy(pts[c0:c0 + 262144]).double().to(device)
            ug[c0:c0 + 262144] = physics.guide_u(xt, ma, xst, Pt,
                                                 St).cpu().numpy()
    return ug


def make_refsub(ev, lb, device):
    rng = np.random.default_rng(777)
    parts, wts = [], []
    xb = sample_ball(8192, R_MAX, rng).astype(np.float64)
    parts.append(xb)
    wts.append(np.ones(len(xb)))
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
    u = np.asarray(ev.evaluate(pts, chunk=131072, dtype=torch.float64))
    np.savez(os.path.join(DATA_DIR, f"refsub_{lb}.npz"),
             x=pts.astype(np.float32), u=u.astype(np.float64),
             w=w.astype(np.float64))


def make_cfg(lb, q, k, device):
    m2 = M1 * q
    ma = torch.tensor([M1, m2], dtype=torch.float64, device=device)
    xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64,
                       device=device)
    Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64,
                      device=device)
    St = torch.zeros((2, 3), dtype=torch.float64, device=device)
    N_INT, N_BND = 12000, 4000
    rng = np.random.default_rng(abs(hash(lb)) % (2 ** 31))
    xi = sample_ball(N_INT, R_MAX, rng).astype(np.float32)
    xbn = sample_sphere_surface(N_BND, R_MAX, rng).astype(np.float32)

    def _c(x):
        xt = torch.from_numpy(x).double().to(device)
        return (physics.psi_sing(xt, ma, xst).cpu().numpy().astype(np.float32),
                physics.bowen_york_KK(xt, ma, xst, Pt, St).cpu().numpy()
                .astype(np.float32),
                physics.guide_u(xt, ma, xst, Pt, St).cpu().numpy()
                .astype(np.float32))

    pi, ki, ui = _c(xi)
    pb, kb, ub = _c(xbn)
    sq = float(np.sqrt(np.mean(ui.astype(np.float64) ** 2)) + 1e-30)
    au = np.concatenate([ui, ub]).astype(np.float64)
    np.savez(os.path.join(DATA_DIR, f"cfg_{lb}.npz"),
             x_int=xi, x_bnd=xbn, ps_int=pi, kk_int=ki, ug_int=ui,
             ps_bnd=pb, kk_bnd=kb, ug_bnd=ub,
             masses=ma.cpu().numpy().astype(np.float32),
             xs=xst.cpu().numpy().astype(np.float32),
             Ps=Pt.cpu().numpy().astype(np.float32),
             Ss=St.cpu().numpy().astype(np.float32),
             sq=sq, wmin=float(au.min()), wmax=float(au.max()),
             kappa=float(k), q=float(q), m1=M1, m2=float(m2), heldout=0)


def main():
    setup_logging("A2", "post_refs_v2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REFS2, exist_ok=True)
    ks_path = os.path.join(DATA_DIR, "kappa_spec_v2.json")
    ks = json.load(open(ks_path)) if os.path.exists(ks_path) else {}
    srcs = label_src_map()
    log.info("参考解 %d 个(新窗口后处理)", len(srcs))
    t0 = time.time()
    for i, (lb, src) in enumerate(sorted(srcs.items()), 1):
        ref_p = os.path.join(DATA_DIR, f"refsub_{lb}.npz")
        cfg_p = os.path.join(DATA_DIR, f"cfg_{lb}.npz")
        if (not args.force and os.path.exists(ref_p) and os.path.exists(cfg_p)
                and lb in ks):
            log.info("[skip] %s", lb)
            continue
        z = np.load(src)
        raw = z["raw"]
        q = float(raw[1] / raw[0])
        m2 = float(raw[1])
        u_full = grid_cache(src, lb, device)
        r1 = np.linalg.norm(
            _grid_pts() - np.array([3.0, 0, 0]), axis=1)
        r2 = np.linalg.norm(
            _grid_pts() - np.array([-3.0, 0, 0]), axis=1)
        keep = (r1 > RCUT) & (r2 > RCUT)
        if lb not in ks or args.force:
            ug = guide_on_grid(lb, q, m2, device)
            k = float((ug[keep] * u_full[keep].astype(np.float64)).sum()
                      / (ug[keep] ** 2).sum())
            ks[lb] = dict(q=q, kappa_star_spec_v2=k, src=os.path.relpath(src,
                                                                         HERE))
            json.dump(ks, open(ks_path, "w"), indent=1)
        else:
            k = float(ks[lb]["kappa_star_spec_v2"])
        if args.force or not os.path.exists(ref_p):
            ev = SpectralPunctureSolver.from_coefficients(
                src, device=str(device), verify=False)
            make_refsub(ev, lb, device)
            del ev
            torch.cuda.empty_cache()
        if args.force or not os.path.exists(cfg_p):
            make_cfg(lb, q, k, device)
        log.info("[%d/%d] %s: q=%g κ*_v2=%.4f (%.0fs)",
                 i, len(srcs), lb, q, k, time.time() - t0)
    log.info("后处理完成,用时 %.1f min → %s", (time.time() - t0) / 60.0,
             DATA_DIR)


_GPTS = None


def _grid_pts():
    global _GPTS
    if _GPTS is None:
        g = np.linspace(-R_MAX, R_MAX, GRID_N)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        _GPTS = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return _GPTS


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("post_refs_v2 failed")
        raise

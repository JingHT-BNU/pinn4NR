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
"""check_ref_conv.py -- hi-res spectral solve vs existing L48 ref: peak-region
convergence check. Decides whether the whole reference set must be regenerated
at higher resolution before retraining.

For one config (default q=1 -> old ref ref_a2_q10.npz):
  - solve fresh SpectralPunctureSolver(N_r=768, L=64, N_th=96, N_ph=192, R0=15)
  - evaluate BOTH (old coefficients, new solve) on dense probes:
      ball r<10 (200k), peak shells r in [0.1,1.5] around each puncture
      (60k each), valley ball r<1.5 at origin (20k), dense x-axis (2001)
  - region-wise L2RE(new vs old); also new solve spectral residual.
Output JSON: data/refs/conv_check_<tag>.json
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from logutil import setup_logging
import spectral_reference as sr

log = logging.getLogger("paper.A2.check_ref_conv")

REFS = os.path.join(_ROOT, "data", "refs", "a2")


def l2re(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30)))


def sample_regions(rng):
    out = {}
    n = 200000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 10.0 * rng.random(n) ** (1.0 / 3.0)
    out["ball10"] = d * r[:, None]
    n = 60000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = (0.1 ** 3 + (1.5 ** 3 - 0.1 ** 3) * rng.random(n)) ** (1.0 / 3.0)
    shell = d * r[:, None]
    out["shell_p"] = shell + np.array([3.0, 0, 0])
    out["shell_m"] = shell - np.array([3.0, 0, 0])
    n = 20000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    out["valley"] = d * (1.5 * rng.random(n) ** (1.0 / 3.0))[:, None]
    xs = np.linspace(-10, 10, 2001)
    ax = np.zeros((len(xs), 3))
    ax[:, 0] = xs
    out["axis"] = ax
    return out


def main():
    setup_logging("A2", "check_ref_conv")
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--old", default=None, help="existing ref npz (default by q)")
    ap.add_argument("--n-r", type=int, default=768)
    ap.add_argument("--lmax", type=int, default=64)
    ap.add_argument("--n-theta", type=int, default=96)
    ap.add_argument("--n-phi", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"q{args.q:g}"
    m2 = 0.5 / args.q
    raw = np.array([0.5, m2, 3.0, -3.0, 0.2, -0.2, 0.0, 0.0])
    old_path = args.old or os.path.join(
        REFS, f"ref_a2_{tag.replace('.', 'p')}.npz")
    if not os.path.exists(old_path):
        cand = os.path.join(REFS, f"ref_tq{str(args.q).replace('.', 'p')}.npz")
        if os.path.exists(cand):
            old_path = cand
        else:
            raise SystemExit(f"no old ref for q={args.q}: {old_path}")
    log.info("[%s] old ref: %s", tag, old_path)
    old = sr.SpectralPunctureSolver.from_coefficients(old_path, device=device,
                                                      verify=False)
    log.info("[%s] old ref: L=%d N_r=%d R0=%g", tag, old.L, old.N_r, old.R0)

    t0 = time.time()
    log.info("[%s] solving hi-res (N_r=%d L=%d Nth=%d Nph=%d)...",
             tag, args.n_r, args.lmax, args.n_theta, args.n_phi)
    new = sr.SpectralPunctureSolver(raw, N_r=args.n_r, L=args.lmax,
                                    N_th=args.n_theta, N_ph=args.n_phi,
                                    R0=15.0, device=device)
    it, stats = new.solve(maxit_final=200)
    log.info("[%s] solved: it=%d res_max=%.3e res_rms=%.3e (%.1f min)",
             tag, it, stats["res_max"], stats["res_rms"],
             (time.time() - t0) / 60.0)

    rng = np.random.default_rng(args.seed)
    regions = sample_regions(rng)
    res = {"tag": tag, "q": args.q, "old": {"L": old.L, "N_r": old.N_r,
                                            "R0": old.R0, "path": old_path},
           "new": {"N_r": args.n_r, "L": args.lmax,
                   "N_theta": args.n_theta, "N_phi": args.n_phi,
                   "res_max": float(stats["res_max"]),
                   "res_rms": float(stats["res_rms"]),
                   "iters": int(it), "solve_min": (time.time() - t0) / 60.0},
           "regions": {}}
    for name, pts in regions.items():
        uo = old.evaluate(pts, chunk=131072, dtype=torch.float64)
        un = new.evaluate(pts, chunk=131072, dtype=torch.float64)
        e = l2re(un, uo)
        res["regions"][name] = {"l2re_new_vs_old": e, "n": int(len(pts))}
        extra = ""
        if name == "axis":
            rr = np.linalg.norm(pts[:, :1] - np.array([[3.0, 0, 0]]), axis=1)
            dx = np.minimum(np.abs(pts[:, 0] - 3), np.abs(pts[:, 0] + 3))
            m = (dx >= 0.1) & (dx <= 1.5)
            e_pk = l2re(un[m], uo[m])
            res["regions"][name]["peak_win_l2re"] = e_pk
            extra = f" peak_win={e_pk:.3e}"
        log.info("[%s] %-8s new-vs-old L2RE=%.3e (%d pts)%s",
                 tag, name, e, len(pts), extra)
    out = os.path.join(_ROOT, "data", "refs", f"conv_check_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    log.info("[%s] wrote %s (%.1f min total)", tag, out,
             (time.time() - t0) / 60.0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("conv check failed")
        raise

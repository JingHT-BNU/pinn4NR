# -*- coding: utf-8 -*-
"""a2q_head2head.py -- champion2 vs opv3 peak/bulk dual-metric head-to-head.

Motivation: the official L2RE (81^3 grid on [-30,30]^3, rcut=0.3 balls removed,
grid spacing 0.75) is bulk-dominated; the near-puncture peak zone holds only a
handful of grid points, so the metric is nearly blind to it. champion2 fits the
peaks well but has larger bulk error; opv3 halves the bulk error but visibly
undershoots the peaks. This script quantifies BOTH on identical spectral refs:

  1. bulk L2RE   : taken verbatim from each run's eval.json (same refs, verified
                   bit-identical via identical l2re_guide values)
  2. peak L2RE   : dense boxes around both punctures, half-width 2.7, spacing
                   0.15 (37^3 each), r<0.3 balls removed
  3. ring L2RE   : subset r in [0.3, 1.0] (the ring just outside the metric hole)
  4. axial L2RE  : dense x-axis (2401 pts), window |x -+3| in [0.05, 1.5],
                   plus peak-height ratio max(u)/max(u_ref) in the window

Outputs: data/runs/a2/head2head.json
         reports/figs/fig7_head2head_profiles.png
         reports/figs/fig8_dual_metric_per_config.png
"""
import json
import logging
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from logutil import setup_logging
import physics
from a2q_model import load_run, predict_a2q
from a2q_eval import l2re, _bridge, REFS
from spectral_reference import SpectralPunctureSolver

log = logging.getLogger("paper.A2.a2q_head2head")

RUNS = os.path.join(HERE, "data", "runs", "a2")
REPORTS = os.path.normpath(os.path.join(HERE, "..", "reports"))
RUN_NAMES = ["a2q_champion2", "a2q_opv3"]
RUN_LABEL = {"a2q_champion2": "champion2", "a2q_opv3": "opv3"}

PEAK_HALF = 2.7
PEAK_N = 37
RCUT = 0.3
AX_N = 2401
CENTERS = ((3.0, 0.0, 0.0), (-3.0, 0.0, 0.0))


def build_peak_pts():
    g = np.linspace(-PEAK_HALF, PEAK_HALF, PEAK_N)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    out, rmin = [], []
    for c in CENTERS:
        p = pts + np.asarray(c, dtype=float)
        r1 = np.linalg.norm(p - np.asarray(CENTERS[0]), axis=1)
        r2 = np.linalg.norm(p - np.asarray(CENTERS[1]), axis=1)
        rm = np.minimum(r1, r2)
        keep = rm > RCUT
        out.append(p[keep])
        rmin.append(rm[keep])
    return (np.concatenate(out, axis=0).astype(np.float64),
            np.concatenate(rmin, axis=0))


def main():
    setup_logging("A2", "a2q_head2head")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(REPORTS, "figs"), exist_ok=True)

    models = {}
    for rn in RUN_NAMES:
        models[rn] = load_run(os.path.join(RUNS, rn), device)
        log.info("loaded %s (variant=%s)", rn, models[rn][1].get("variant"))
    base_meta = models["a2q_champion2"][1]["meta"]
    heldout = set(models["a2q_champion2"][1].get("heldout_labels", []))
    labels = sorted(base_meta.keys(), key=lambda lb: base_meta[lb]["q"])
    for rn in RUN_NAMES:
        assert set(models[rn][1]["meta"].keys()) == set(base_meta.keys()), rn

    peak_pts, peak_rmin = build_peak_pts()
    log.info("peak box points: %d (r in [%.2f, %.2f])",
             len(peak_pts), peak_rmin.min(), peak_rmin.max())

    xs_ax = np.linspace(-28.0, 28.0, AX_N)
    ax_pts = np.zeros((AX_N, 3))
    ax_pts[:, 0] = xs_ax
    dx_ax = np.minimum(np.abs(xs_ax - 3.0), np.abs(xs_ax + 3.0))
    win = (dx_ax >= 0.05) & (dx_ax <= 1.5)
    ring = peak_rmin <= 1.0

    results = {lb: {"q": float(base_meta[lb]["q"]),
                    "heldout": lb in heldout} for lb in labels}
    fig_cache = {}
    t0 = time.time()
    for lb in labels:
        m = base_meta[lb]
        cinfo = {k: m[k] for k in ("q", "m2", "kappa", "sq", "wmin", "wmax")}
        cinfo["m1"] = 0.5
        src = os.path.join(REFS, f"ref_a2_{lb}.npz")
        if not os.path.exists(src):
            log.warning("%s: ref missing, skipped", lb)
            continue
        ev = SpectralPunctureSolver.from_coefficients(src, device=str(device),
                                                      verify=False)
        u_ref_peak = ev.evaluate(peak_pts, chunk=131072, dtype=torch.float64)
        u_ref_ax = ev.evaluate(ax_pts.astype(np.float32), chunk=131072,
                               dtype=torch.float64).astype(np.float64)
        denom_peak = float(np.sum(u_ref_peak ** 2))
        denom_ax = float(np.sum(u_ref_ax[win] ** 2))

        ma = torch.tensor([0.5, m["m2"]], dtype=torch.float64)
        xst = torch.tensor([[3.0, 0, 0], [-3.0, 0, 0]], dtype=torch.float64)
        Pt = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
        St = torch.zeros((2, 3), dtype=torch.float64)
        ug_ax = (float(m["kappa"]) *
                 physics.guide_u(torch.from_numpy(ax_pts), ma, xst, Pt, St).numpy())

        entry = results[lb]
        curves = {"ref": u_ref_ax, "guide": ug_ax, "xs": xs_ax}
        for rn in RUN_NAMES:
            model, _ = models[rn]
            u_peak = predict_a2q(model, peak_pts, cinfo, device,
                                 chunk=32768).astype(np.float64)
            u_ax = predict_a2q(model, ax_pts.astype(np.float32), cinfo, device,
                               chunk=65536).astype(np.float64)
            e = {
                "peak_l2re": l2re(u_peak, u_ref_peak),
                "ring_l2re": l2re(u_peak[ring], u_ref_peak[ring]),
                "ax_l2re": l2re(u_ax[win], u_ref_ax[win]),
                "ax_maxerr": float(np.max(np.abs(u_ax[win] - u_ref_ax[win]))),
                "peak_height_ratio": float(u_ax[win].max() /
                                           max(u_ref_ax[win].max(), 1e-30)),
                "bulk_l2re": json.load(open(os.path.join(
                    RUNS, rn, "eval.json")))["results"][lb]["l2re"],
            }
            entry[RUN_LABEL[rn]] = e
            curves[RUN_LABEL[rn]] = u_ax
            log.info("%-5s %-9s peak=%.3e ring=%.3e ax=%.3e axmax=%.3e "
                     "hratio=%.4f (%.0fs)",
                     lb, RUN_LABEL[rn], e["peak_l2re"], e["ring_l2re"],
                     e["ax_l2re"], e["ax_maxerr"], e["peak_height_ratio"],
                     time.time() - t0)
        fig_cache[lb] = curves

    out_json = os.path.join(RUNS, "head2head.json")
    with open(out_json, "w") as f:
        json.dump({"results": results,
                   "peak_box": {"half": PEAK_HALF, "n": PEAK_N,
                                "spacing": 2 * PEAK_HALF / (PEAK_N - 1),
                                "rcut": RCUT},
                   "axial_window": [0.05, 1.5]}, f, indent=1)
    log.info("wrote %s", out_json)

    # ---- summary table ----
    rows = ["config      " + "".join(f"{c:>28}" for c in
                                     ("champ_peak", "opv3_peak", "champ_ring",
                                      "opv3_ring", "champ_ax", "opv3_ax",
                                      "champ_bulk", "opv3_bulk"))]
    agg = {rn: {k: [] for k in ("peak_l2re", "ring_l2re", "ax_l2re",
                                "bulk_l2re", "peak_height_ratio")}
           for rn in RUN_NAMES}
    for lb in labels:
        e = results[lb]
        if RUN_LABEL[RUN_NAMES[0]] not in e:
            continue
        vals = []
        for key in ("peak_l2re", "ring_l2re", "ax_l2re"):
            for rn in RUN_NAMES:
                v = e[RUN_LABEL[rn]][key]
                vals.append(v)
                agg[rn][key].append(v)
        for rn in RUN_NAMES:
            vals.append(e[RUN_LABEL[rn]]["bulk_l2re"])
            agg[rn]["bulk_l2re"].append(e[RUN_LABEL[rn]]["bulk_l2re"])
        for rn in RUN_NAMES:
            agg[rn]["peak_height_ratio"].append(
                e[RUN_LABEL[rn]]["peak_height_ratio"])
        rows.append(f"{lb:<12}" + "".join(f"{v:28.4e}" for v in vals))
    log.info("per-config table:\n%s", "\n".join(rows))
    means = {RUN_LABEL[rn]: {k: float(np.mean(v)) for k, v in d.items()}
             for rn, d in agg.items()}
    log.info("MEANS over %d configs: %s", len(labels), json.dumps(means))

    # ---- fig7: head-to-head profiles ----
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    show = [lb for lb in ("q10", "q20", "q50", "q100") if lb in fig_cache]
    fig, axes = plt.subplots(len(show), 2, figsize=(13.5, 3.2 * len(show)),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    axes = np.atleast_2d(axes)
    for i, lb in enumerate(show):
        cv = fig_cache[lb]
        e = results[lb]
        tag = " (heldout)" if e["heldout"] else ""
        for j, xlim in enumerate(((-28, 28), (1.3, 4.7))):
            ax = axes[i][j]
            msk = (cv["xs"] >= xlim[0]) & (cv["xs"] <= xlim[1])
            ax.plot(cv["xs"][msk], _bridge(cv["xs"], cv["ref"])[msk], "-",
                    lw=1.6, color="C3", label="spectral reference")
            ax.plot(cv["xs"][msk], _bridge(cv["xs"], cv["guide"])[msk], "--",
                    lw=0.8, color="gray", label="guide k·u_g")
            for rn in RUN_NAMES:
                nm = RUN_LABEL[rn]
                ax.plot(cv["xs"][msk], _bridge(cv["xs"], cv[nm])[msk], "-",
                        lw=1.0, color={"a2q_champion2": "C0",
                                       "a2q_opv3": "C2"}[rn], label=nm)
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"q={e['q']:g}{tag}" + ("  full axis" if j == 0
                                                 else "  zoom @ x=+3"), fontsize=10)
            if j == 1:
                txt = "\n".join(
                    f"{RUN_LABEL[rn]}: axL2RE={e[RUN_LABEL[rn]]['ax_l2re']:.2e}, "
                    f"h/h_ref={e[RUN_LABEL[rn]]['peak_height_ratio']:.3f}"
                    for rn in RUN_NAMES)
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8,
                        va="top", ha="left",
                        bbox=dict(fc="white", ec="0.7", alpha=0.85))
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("champion2 vs opv3 vs spectral reference: x-axis profiles "
                 "(y=z=0; same refs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f7 = os.path.join(REPORTS, "figs", "fig7_head2head_profiles.png")
    fig.savefig(f7, dpi=300)
    plt.close(fig)
    log.info("[fig] %s", f7)

    # ---- fig8: dual metric per config ----
    labs = [lb for lb in labels if RUN_LABEL[RUN_NAMES[0]] in results[lb]]
    x = np.arange(len(labs))
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, key, title in ((axs[0], "peak_l2re", "peak box L2RE (r<2.7, "
                            "spacing 0.15)"),
                           (axs[1], "ring_l2re", "ring L2RE (r<1.0)")):
        for off, (rn, col) in enumerate((("a2q_champion2", "C0"),
                                         ("a2q_opv3", "C2"))):
            v = [results[lb][RUN_LABEL[rn]][key] for lb in labs]
            hs = [results[lb]["heldout"] for lb in labs]
            bars = ax.bar(x + (off - 0.5) * 0.4, v, width=0.4, color=col,
                          label=RUN_LABEL[rn])
            for b, h in zip(bars, hs):
                if h:
                    b.set_hatch("//")
                    b.set_edgecolor("white")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"q{results[lb]['q']:g}" for lb in labs],
                           rotation=45, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    ax = axs[2]
    for rn, col in (("a2q_champion2", "C0"), ("a2q_opv3", "C2")):
        bx = [results[lb][RUN_LABEL[rn]]["bulk_l2re"] for lb in labs]
        by = [results[lb][RUN_LABEL[rn]]["peak_l2re"] for lb in labs]
        ax.scatter(bx, by, c=col, s=28, label=RUN_LABEL[rn])
    lo, hi = 4e-3, 5e-2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("bulk L2RE (official 81^3 metric)")
    ax.set_ylabel("peak box L2RE")
    ax.set_title("bulk vs peak: above diagonal = peak worse than bulk",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("champion2 vs opv3: bulk breakthrough vs peak regression "
                 "(hatched = heldout)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f8 = os.path.join(REPORTS, "figs", "fig8_dual_metric_per_config.png")
    fig.savefig(f8, dpi=300)
    plt.close(fig)
    log.info("[fig] %s", f8)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("head2head failed")
        raise

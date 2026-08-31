# A2-opv2 — Shape-Correcting Operator with Functional Inputs

Companion code for the **operator v2 variant** of our parametric PINN for
binary black hole (BBH) initial data (paper: arXiv:2607.06002v1): a
correction field that reads **true functional inputs** — samples of the
guide solution in a neighborhood of each query point.

## Method

The v1 operator ([`A2-operator`](../../tree/A2-operator)) reads only per-point
features and degenerates to a constant rescaling. v2 gives each query point a
**local patch** of the guide solution:

    u = κ·u_g·(1 + w·G_θ(x, p, f_patch)),   G_θ ∈ [-3, 3]

    f_patch = { log1p(|u_g(x + r_i·d_j)| / σ_g) }   for i = 1..3 radii
                                                      (0.5, 1.5, 4.0),
                                                      j = 1..8 Fibonacci
                                                      directions → 24 features

The radii span three shape scales of the guide field (inside the peaks, the
inter-peak valley, the global envelope). The correction field can now
represent *shape* modifications that follow the local geometry of the guide
solution, shared across all 15 training configurations. Patch features are
precomputed once per fixed point set and indexed per step (8 ms overhead).

Training recipe: identical to [`A2-champion`](../../tree/A2-champion)
(corrected κ + all-configuration reference supervision + parameter noise) —
a clean ablation against champion.

## Results (15000 steps, 81³ grid, L=48 spectral reference)

| q | 1.0 | 1.5 (held-out) | 2.0 | 5.0 (held-out) | 10 |
|---|-----|-----|-----|-----|-----|
| opv2 | 2.63e-2 | 2.90e-2 | 3.21e-2 | 3.35e-2 | 3.10e-2 |
| champion | 2.65e-2 | 3.03e-2 | 3.20e-2 | 3.35e-2 | 2.87e-2 |

Both success criteria evaluated, both **negative**:

1. L2RE statistically identical to champion — the 3e-2 plateau stands;
2. The structure probe (`a2q_opv2_probe.py`) shows ψ = corr_eff still
   collapses to a **constant** (all percentiles = 0.133, relative IQR = 0,
   geometry correlation ≈ 0) — even with full neighborhood information, the
   training signal rewards only a constant rescaling.

**Conclusion**: the bottleneck is neither network capacity nor input
information — it is the loss structure. This completes the evidence chain
that led to the supervision redesign of
[`A2-champion2`](../../tree/A2-champion2) (−24% without any ansatz change)
and identifies the far-field guide-shape error as the remaining frontier
(requiring a free-field correction `u = κ·u_g + Δ_θ`).

## Repository layout

```
a2q_model.py     # OperatorV2Ansatz (this variant), patch_offsets/fibonacci_dirs
a2q_train.py     # unified trainer; run with --variant opv2
a2q_opv2_probe.py # correction-field structure probe (percentiles/IQR/correlations)
a2q_opv2_smoke.py # patch-feature timing + numerical health check
a2q_eval.py      # 81³ L2RE evaluation + axis-profile figures
a2q_runner.cmd   # crash-restart wrapper with checkpoint resumption
```

## Running

```cmd
:: data pipeline (see A2-base branch README)
python a2q_prep.py && python a2q_prep2.py && python a2q_refsub.py && python a2q_kappa2.py

:: train (15000 steps, ~135 min on a 16 GB GPU)
cmd /c a2q_runner.cmd opv2 a2q_opv2 15000

:: evaluate + structure probe
python a2q_eval.py --run runs\a2q_opv2 --grid-n 81
python a2q_opv2_probe.py a2q_opv2 a2q_champion
```

# A3 — Multi-Parametric PINN for Binary Black Hole Initial Data

Companion code for the **multi-parametric** stage of our study on BBH initial
data via physics-informed neural networks (paper: arXiv:2607.06002v1): one
network covering an **8-dimensional configuration space** — two puncture
masses, positions, momenta and spins.

## Method

- **Configuration sampling**: Latin Hypercube Sampling over the 8-D box
  (400 configurations: 300 training + 100 validation);
- **Network**: FiLM-conditioned MLP (6×256, ~726K parameters); parameters are
  sinusoidally encoded and modulate hidden layers;
- **Ansatz & supervision**: same guided hard-constraint framework as the
  single-parameter study,
  $u = \kappa\,u_g\,(1 + c\,w\,\tanh h_\theta)$, with reference-solution
  supervision + PDE regularization + curriculum learning;
- κ precomputed for all LHS configurations (`multi_param_precompute.py`).

## Status

Training variant **v5** reaches **L2RE ≈ 0.52%** on the base configuration
(our full-reference metric). Earlier variants (v1–v4) and their component-level
diagnoses are documented in `multi_param_experiments.md` (see also
`README_legacy.md` for the original run instructions).

## Carrying over the single-parameter lessons

The single-parameter study (branches `A2-base` … `A2-champion2`) established
several methodological facts that directly apply here:

1. **κ caches are noise-sensitive**: per-configuration random seeds + 200k
   QMC points make κ(q) jump by up to ±21%, coinciding with model failure
   regions. Use a fixed seed with ~2M points (see `A2-base`:
   `a2q_kappa2.py`); the bundled `multi_param_kappa_cache.json` predates
   this fix;
2. **Per-configuration window normalization** (not global min-max);
3. **All-configuration reference supervision at every step** — sampling a few
   configurations per step lets the PDE batch-mean gradient drown the
   reference signal (see `A2-champion2` for the decisive experiment);
4. **Well-scaled parameter encoding** (`log10`-style), not raw-value
   sinusoidal encoding;
5. Far-field guide-shape error is the remaining frontier for sub-2% targets.

## Files

```
multi_param_model.py       # 8-D FiLM-conditioned MLP
multi_param_train.py       # training loop (variants v1–v5)
multi_param_eval.py        # evaluation + parameter sensitivity
multi_param_viz.py         # visualization
multi_param_precompute.py  # κ precomputation (LHS 400 configurations)
multi_param_experiments.md # experiment log (v1–v5 diagnoses)
README_legacy.md           # original run instructions
```

## Running

```cmd
python multi_param_precompute.py
python multi_param_train.py --steps 15000
python multi_param_eval.py --run runs\multi_param_v5
```

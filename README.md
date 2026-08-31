# A2-parametric-legacy — First Parametric Study (q ∈ [0.5, 2])

**Legacy branch**, preserved for historical reference. This was our first
extension of the single-configuration PINN (paper: arXiv:2607.06002v1) to a
parametric model: one network covering mass ratios q ∈ [0.5, 2]. It was
superseded by the redesigned study in the [`A2-base`](../../tree/A2-base) and
subsequent branches.

## Method

- **Network**: FiLM-conditioned MLP (4×128, ~112K parameters); the mass
  parameter is sinusoidally encoded and modulates hidden layers;
- **Ansatz**: $u = \kappa\,u_g\,(1 + c\,w\,\tanh h_\theta(x, p))$ with a
  **global** min-max window (all configurations share one normalization);
- **Supervision**: reference-solution supervision + PDE regularization +
  curriculum learning;
- κ precomputed for 19 mass configurations (`precompute_kappa.py` →
  `kappa_cache.json`).

## Results and lessons (why it was redesigned)

1. Interpolation near $q = 1$ worked, but **light configurations had ~50×
   worse residuals**: the global min-max window compresses the correction of
   light configurations to ~1% (guide-solution peaks scale as $m^2$, a 16×
   range across configurations). → the redesigned study uses
   **per-configuration window normalization**;
2. Extrapolation outside the training interval failed outright
   (generalization study); the new study trains on $q \in [1,10]$ where
   physical interest lies ($q$ and $1/q$ are mirror-equivalent
   configurations);
3. Sinusoidal encoding of raw $m_2$ aliases badly at $m_2 \to 5$ (frequency
   $e^7$). → the redesigned study encodes $[\log_{10} q,\ m_2/5]$.

These three lessons directly shaped the `A2-base` branch design.

## Files

```
parametric_model.py    # FiLM-conditioned MLP ansatz
parametric_train.py    # training: reference supervision + curriculum
parametric_eval.py     # L2RE (dual metrics) + physical validation
parametric_viz.py      # per-parameter-axis profiles / loss curves
precompute_kappa.py    # κ lookup-table precomputation
```

## Running

```cmd
python precompute_kappa.py
python parametric_train.py --steps 15000
python parametric_eval.py --run runs\parametric_a1
```

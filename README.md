# A2-base — Parametric PINN for BBH Initial Data (Baseline)

Companion code for the **baseline parametric model** of our study on binary
black hole (BBH) initial data via physics-informed neural networks. The
published paper **Solving Hamiltonian Constraint Equation with
Physics-Informed Neural Networks** (arXiv:2607.06002v1) is the *reference
method*; this branch is the starting point of our parametric extension.

## Task

One network solves the Hamiltonian constraint across a **family** of BBH
configurations parameterized by the mass ratio

$$q = m_2/m_1 \in [1, 10], \qquad m_1 = 0.5 \text{ fixed},$$

with punctures at $x = \pm 3$, momenta $P = (0, \pm 0.2, 0)$, zero spin.
15 training configurations (geometric spacing) + 3 held out
($q = 1.5, 2.5, 5$). Since $q$ and $1/q$ are mirror-equivalent configurations,
only $q \ge 1$ is covered.

## Method

- **Ansatz** (guided hard-constraint, following the reference paper):

$$u = \kappa\, u_g\,(1 + c\,w\,\tanh h_\theta(x, p)),$$

  with the analytic guide solution $u_g$ (Lousto–Zlochower 2008), a
  learnable scalar $c$, and a **per-configuration window** $w$ (fixes the
  global-min-max flaw of a first study, which suppressed light configurations
  by ~50×);
- **Parameter encoding** $[\log_{10} q,\ m_2/5]$ (raw-value sinusoidal
  encoding aliases at large $m_2$);
- **Fast residual path**: $u_g$, $\nabla u_g$, $\Delta u_g$ precomputed per
  configuration; autograd only through the MLP (verified equivalent to full
  autograd to ~1e-6; ~20× speedup);
- **Per-configuration residual normalization** by $\sqrt{\langle u_g^2\rangle}$
  so configurations contribute comparably;
- Training: 3 random configurations per step, Adam + cosine annealing, EMA
  loss balancing, checkpoint resumption.

The κ calibration is solved per configuration by QMC — **fixed seed + 2M
points** (see `a2q_kappa2.py`; per-configuration random seeds at 200k points
make κ(q) jump by up to ±21% and coincide exactly with model failure regions).

## Results (81³ grid vs L=48 spectral reference, r > 0.3)

| q | 1.0 | 1.2 | 1.5 (held out) | 2.0 | 2.5 (held out) | 3.3 |
|---|-----|-----|-----|-----|-----|-----|
| L2RE | **5.9e-3** | 2.1e-2 | 1.3e-1 | **2.1e-1** | 1.9e-2 | 3.2e-2 |

- q = 1 (strong single-configuration supervision) **reaches the paper's base
  accuracy** (5.9e-3 ≈ 0.0067);
- q ≈ 2 was a catastrophic region in early runs — traced to the κ QMC noise
  above and later fixed;
- The remaining uniform error (~3e-2) is analyzed in the follow-up branches:
  [`A2-operator`](../../tree/A2-operator) →
  [`A2-champion`](../../tree/A2-champion) →
  [`A2-champion2`](../../tree/A2-champion2) (best model, −24%).

## Repository layout

```
a2q_model.py    # BaselineAnsatz / OperatorAnsatz / OperatorV2Ansatz
a2q_train.py    # unified trainer (variants: base/operator/refweight/rar/
                #  noise/champion/opv2/champion2); this branch documents base
a2q_eval.py     # 81³ L2RE (dual metrics) + per-axis profile figures
a2q_prep.py / a2q_prep2.py / a2q_refsub.py   # dataset pipeline
a2q_make_refs.py / a2q_kappa2.py             # reference solutions + κ re-solve
a2q_kappa_probe.py / a2q_kappa_fit.py        # κ diagnostics + κ* calibration
a2q_check_fast.py  # fast-path equivalence check (~1e-6)
a2q_runner.cmd     # crash-restart wrapper with checkpoint resumption
```

Large artifacts live in `data/` (git-ignored): `data/refs/a2/` (reference
solutions), `data/datasets/a2q_data/` (training datasets), `data/runs/a2/`
(outputs).

## Running

```cmd
:: 1. reference solutions (18 configurations, ~45 min on GPU)
python a2q_make_refs.py
:: 2. datasets + κ
python a2q_prep.py && python a2q_prep2.py && python a2q_refsub.py && python a2q_kappa2.py
:: 3. train baseline (10000 steps, ~70 min)
cmd /c a2q_runner.cmd base a2q_base 10000
:: 4. evaluate
python a2q_eval.py --run runs\a2q_base --grid-n 81
```

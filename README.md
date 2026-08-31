# A2-champion — Combined Recipe for Parametric BBH Initial Data

Companion code for the **champion variant** of our parametric PINN for binary
black hole (BBH) initial data: the best-performing combination of supervision
and training-schedule improvements identified across the A2 study, applied to
the baseline ansatz. The published paper **Solving Hamiltonian Constraint
Equation with Physics-Informed Neural Networks** (arXiv:2607.06002v1) is the
*reference method*.

## Method

Baseline ansatz (unchanged — the
[`A2-operator`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-operator) study
showed network capacity is not the bottleneck):

$$u = \kappa\, u_g\,(1 + c\,w\,\tanh h_\theta(x, p)).$$

Champion recipe on top of it:

1. **Corrected κ**: the cached κ values used by earlier variants suffered from
   QMC sampling noise — jumps up to $\pm 21\%$ in $\kappa(q)$, coinciding
   exactly with the model's failure regions. We re-solve κ with a fixed seed
   and 2M volume points (`a2q_kappa2.py`); the resulting $\kappa(q)$ is smooth
   and monotone;
2. **All-configuration reference supervision** with $3\times$ weight and a
   floor on the reference-loss coefficient (counteracts batch-mean dilution);
3. **Progressive parameter noise**:

$$\tilde q = q\,e^{\sigma_t \xi}, \qquad \sigma_t:\ 0 \to 0.06 \text{ (first half of training)},$$

   with the guide field and κ recomputed online ($\xi \sim \mathcal N(0,1)$);
4. 15,000 steps (vs. 10,000 for the single-variant studies).

Task setup: mass ratio $q \in [1,10]$, 15 training + 3 held-out configurations
(see [`A2-base`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-base)).

## Results (81³ grid, L=48 spectral reference, r > 0.3)

| q | 1.0 | 1.5 (held-out) | 2.0 | 5.0 (held-out) | 10 |
|---|-----|-----|-----|-----|-----|
| champion | 2.65e-2 | 3.03e-2 | 3.20e-2 | 3.35e-2 | 2.87e-2 |
| guide baseline | 3.48e-2 | 3.92e-2 | 4.13e-2 | 3.97e-2 | 3.16e-2 |

Global (train) mean 3.15e-2, max 3.50e-2 — a **uniform** error level: the
catastrophic region of the baseline ($q \approx 2$: L2RE 0.21) is fully cured.

**Key diagnostic**: the reference loss stays flat for all 15,000 steps. This
is *not* undertraining — see
[`A2-champion2`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-champion2) for
the decisive experiment (pure-reference fine-tuning drives the same network
2.4× lower) and the fix (all-configuration full-batch reference supervision:
global mean 3.15e-2 → 2.40e-2).

## Repository layout

```
a2q_train.py   # unified trainer; run with --variant champion
a2q_model.py   # BaselineAnsatz (champion uses the baseline ansatz)
a2q_kappa2.py  # fixed-seed high-density κ re-solve (the QMC-noise fix)
a2q_eval.py    # 81³ L2RE evaluation + axis-profile figures
a2q_region_diag.py  # near/mid/far-field error decomposition
a2q_runner.cmd # crash-restart wrapper with checkpoint resumption
```

## Running

```cmd
:: data pipeline (see the A2-base branch README)
python a2q_prep.py && python a2q_prep2.py && python a2q_refsub.py && python a2q_kappa2.py

:: train (15000 steps, ~105 min on a 16 GB GPU)
cmd /c a2q_runner.cmd champion a2q_champion 15000

:: evaluate
python a2q_eval.py --run runs\a2q_champion --grid-n 81
```

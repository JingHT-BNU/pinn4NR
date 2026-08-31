# A2-champion2 — Two-Stage Training for Parametric BBH Initial Data

Companion code for the **best-performing model** of our parametric PINN study
on binary black hole (BBH) initial data. The published paper **Solving
Hamiltonian Constraint Equation with Physics-Informed Neural Networks**
(arXiv:2607.06002v1) is the *reference method*. This branch is stage two of a
two-stage training scheme; stage one is the
[`A2-champion`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-champion)
branch.

## Method

Stage two fine-tunes the champion checkpoint with two targeted fixes derived
from an error decomposition of the uniform $3.2\times10^{-2}$ plateau:

1. **κ recalibration (κ\*)**: the QMC-solved κ carries a systematic bias of
   $0.8$–$1.5\%$ versus the least-squares-optimal value measured against the
   spectral reference (`a2q_kappa_fit.py` → `kappa_star.json`):

$$\kappa^* = \arg\min_\kappa \sum_i \left(\kappa\, u_g(\mathbf{x}_i) - u_{\rm ref}(\mathbf{x}_i)\right)^2.$$

   Training and evaluation use $\kappa^*$;
2. **All-configuration full-batch reference supervision**: at every step, the
   reference-regression term includes **all 15 training configurations** at
   full point count (10,240 points each), instead of a random 3. The
   motivation is a decisive experiment (`a2q_ft_test.py`): fine-tuning the
   very same network on *pure* reference regression lowers the reference loss
   $2.4\times$ below the champion plateau — proving the plateau was caused by
   the PDE batch-mean gradient drowning the reference signal, not by network
   capacity.

## Results (81³ grid, L=48 spectral reference, r > 0.3)

| q | 1.0 | 1.5 (held-out) | 2.0 | 2.5 (held-out) | 5.0 (held-out) | 10 |
|---|-----|-----|-----|-----|-----|-----|
| champion2 | **1.91e-2** | 3.16e-2 | **2.57e-2** | 3.71e-2 | 3.45e-2 | **1.86e-2** |
| champion | 2.65e-2 | 3.03e-2 | 3.20e-2 | 3.52e-2 | 3.35e-2 | 2.87e-2 |

Training-configuration mean **$3.15\times10^{-2} \to 2.40\times10^{-2}$
(−24%)**; best results at large mass ratios ($q = 10$: 1.86e-2, 1.5× better
than the guide solution).

**Remaining bottleneck** (error decomposition, `a2q_region_diag.py`): the
near/mid field already matches the paper's base-case benchmark; the residual
error lives almost entirely in the **far field** (99.3% of grid points), where
the window-modulated correction $w\,\psi$ loses its leverage ($w \to 0$) and
the model can only track the guide solution. Breaking this floor requires a
free-field correction ansatz $u = \kappa\,u_g + \Delta_\theta$ — future work.

*Note*: held-out configurations regress slightly ($\kappa^*$ is fitted on the
15 training references); use the smooth $\kappa^*(q)$ curve for interpolation
when evaluating unseen $q$.

## Repository layout

```
a2q_train.py      # unified trainer; run with --variant champion2
                  # (loads runs/a2q_champion/model.pt + a2q_data/kappa_star.json)
a2q_kappa_fit.py  # κ* calibration against the spectral references
a2q_ft_test.py    # the pure-reference fine-tuning experiment (plateau diagnosis)
a2q_region_diag.py # near/mid/far-field error decomposition
a2q_eval.py       # 81³ L2RE evaluation + axis-profile figures
a2q_runner.cmd    # crash-restart wrapper with checkpoint resumption
```

## Running

```cmd
:: prerequisites: stage-one checkpoint (A2-champion branch) + kappa_star.json
python a2q_prep.py && python a2q_prep2.py && python a2q_refsub.py
python a2q_kappa2.py && python a2q_kappa_fit.py

:: stage-two fine-tuning (6000 steps, ~75 min on a 16 GB GPU)
cmd /c a2q_runner.cmd champion2 a2q_champion2 6000

:: evaluate + error decomposition
python a2q_eval.py --run runs\a2q_champion2 --grid-n 81
python a2q_region_diag.py a2q_champion2
```

# A2-operator — Neural-Operator Correction Field for Parametric BBH Initial Data

Companion code for the **neural-operator variant** of our parametric PINN for
binary black hole (BBH) initial data. The published paper **Solving
Hamiltonian Constraint Equation with Physics-Informed Neural Networks**
(arXiv:2607.06002v1) is the *reference method*; this branch extends its
guided hard-constraint ansatz with a **conditional operator network** that
reads guide-solution features.

## Method

The baseline parametric ansatz (see
[`A2-base`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-base)) uses a
scalar-scaled correction

$$u = \kappa\, u_g\,(1 + c\,w\,\tanh h_\theta(x, p)).$$

This variant replaces the scalar scale with a **bounded operator field**:

$$u = \kappa\, u_g\,\big(1 + w\,G_\theta(x, p, f)\big), \qquad G_\theta \in [-3, 3],$$

where $G_\theta$ is a FiLM-conditioned MLP that additionally reads per-point
guide-solution features

$$f = \big[\,\log(1+|u_g|/\sigma_g),\ w\,\big],$$

with a zero-initialized final layer (training starts exactly from the guide
solution). Task setup identical to A2-base: $q \in [1,10]$, 15 training +
3 held-out configurations, fast residual path, fixed-seed high-density κ.

## Result (and a useful negative finding)

Evaluated on an 81³ grid vs. an L=48 spectral reference ($r > 0.3$):

| q | 1.0 | 1.5 (held-out) | 2.0 | 5.0 (held-out) | 10 |
|---|-----|-----|-----|-----|-----|
| L2RE | 5.9e-3 | 1.35e-1 | 2.1e-1 | 5.5e-2 | — |

The operator variant converges to **exactly the same solution as the scalar
baseline** (per-configuration difference < 1%). A correction-field usage probe
(`a2q_operator_probe.py`) shows why: $|{\rm corr}|$ sits at a constant
$\approx 0.21$ across all percentiles (p50 = p90 = p99, zero saturation) —
the training signal (PDE batch mean + single-configuration reference) is
satisfied by a trivial **constant rescaling** of the guide solution, so the
network never uses its shape degrees of freedom.

**Takeaway**: the expressive power of the ansatz is not the bottleneck; the
bottleneck lies in the loss structure and supervision placement. This
motivated the supervision-side redesign in
[`A2-champion`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-champion) and
[`A2-champion2`](https://github.com/JingHT-BNU/pinn4NR/tree/A2-champion2),
which reduced the global error by 24% without changing the network.

## Repository layout

```
a2q_model.py   # BaselineAnsatz / OperatorAnsatz (this variant) / OperatorV2Ansatz
a2q_train.py   # unified trainer; run with --variant operator
a2q_eval.py    # 81³ L2RE evaluation + per-parameter-axis profile figures
a2q_operator_probe.py  # correction-field usage probe (the analysis above)
a2q_prep.py, a2q_prep2.py, a2q_refsub.py, a2q_kappa2.py  # data pipeline
a2q_runner.cmd # crash-restart wrapper with checkpoint resumption
```

## Running

```cmd
:: data pipeline (see the A2-base branch README for details)
python a2q_prep.py && python a2q_prep2.py && python a2q_refsub.py && python a2q_kappa2.py

:: train (10000 steps, ~70 min on a 16 GB GPU)
cmd /c a2q_runner.cmd operator a2q_operator 10000

:: evaluate + probe
python a2q_eval.py --run runs\a2q_operator --grid-n 81
python a2q_operator_probe.py
```

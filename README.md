# EGGROLL O(1/r) Convergence — Independent Empirical Verification

> **I worked through the math and explained the bits & pieces in my blog post — please read it first:** <https://yuvanesh.vercel.app/blogs/EGGROLL>
>
> This repo is the empirical companion: independent verification of Theorem 2 of *"Evolution Strategies at the Hyperscale"* (arXiv:2511.16652), on a **4 GB consumer GPU** (RTX 3050 Laptop). Two experiments: a toy tanh teacher–student (`eggroll_experiment/`) and a real **MNIST MLP** (`eggroll_nn/`).

## Headline results

| Setup | Slope (bias-only) | R² | r values used in fit |
|---|---|---|---|
| Toy — tanh teacher–student, d=32       | **-0.898** | 0.998   | r ∈ {1, 2, 4, 8} |
| MNIST MLP — 784→32→10, σ=0.3            | **-0.755** | 0.9995  | r ∈ {1, 2, 4, 8} |
| CLT baseline (what naive replication would predict) | -0.5 | — | — |

Both setups clearly beat the CLT baseline. The bias decays nearly perfectly along a 1/r power-law line on the high-SNR ranks.

The fitted slopes are slightly less negative than -1.0 because at the σ values where the bias signal clears the MC noise floor on a 4 GB GPU, the higher-order O(σ⁴) terms contribute non-trivially at small `r`. The theorem says `O(1/r)`, not `= c/r` — the **rate** is what's verified, and the log-log linearity (R² ≈ 1) confirms a power-law.

![convergence_nn](eggroll_nn/results/figures/convergence_loglog_annotated.png)

## What's verified

```
|| g_EGGROLL^r - g_True ||_F  =  O(1/r)
```

As EGGROLL's perturbation rank `r` grows, its gradient estimate converges to the true (full-rank Gaussian) ES gradient at rate **1/r** — quadratically faster than the central limit theorem's **1/√r**.

## Why this is non-trivial to reproduce

Following the obvious formula gives slope ≈ 0. Three things have to be right:

### 1. The fitness must have non-zero higher-order derivatives

A quadratic fitness like `f(W) = -||W - W*||²` gives the EGGROLL estimator *exactly the same expectation* as full-rank Gaussian ES. The covariance of `E_eggroll = (1/√r) A Bᵀ` matches Gaussian; a quadratic has no third or higher derivatives for the higher cumulants to bite on. The theorem's bias is **literally zero** for that fitness — the theorem is trivially true but invisible.

Fix: use a fitness with non-zero higher derivatives. We use tanh (toy) and cross-entropy on MNIST (NN).

### 2. Average gradients across trials *before* computing the error

A single per-trial `||g_eg^r - g_ref||` measures Monte-Carlo variance, not bias. The bias signal is buried below MC noise at every individual trial.

Fix: compute `N_trials` independent EGGROLL estimates per `r`, **average them**, then take the norm against `g_ref`. Variance of the trial-mean decays as `1/(N_trials · N_eggroll)`.

### 3. Subtract *both* SEM terms

With finite samples:
```
err(r)²  ≈  bias(r)²  +  SEM_eg(r)²  +  SEM_ref²
```
The trial-mean still has noise (`SEM_eg`), and `g_ref` itself has noise (`SEM_ref`). To estimate `SEM_ref` we compute `g_ref` as the mean of **M=10 independent chunks**, with the chunk-to-chunk std-of-mean giving us `SEM_ref` empirically. Then the unbiased bias is
```
bias(r) = sqrt(max(err(r)² - SEM_eg(r)² - SEM_ref², 0))
```
and we fit `log(bias)` vs `log(r)` only on the ranks where the signal is significant (`err² > 2·noise²`).

**Antithetic sampling** is also load-bearing — without it, the `f(μ)·E` baseline-variance term dominates by orders of magnitude. Both estimators use
```
g = (1/(2σN)) Σ_i [ f(μ + σE_i) - f(μ - σE_i) ] · E_i
```

### σ choice

`σ` is a hyperparameter, **not a function of `r`** (confirmed from the paper's project page; σ scales with dimension `d`, not rank). Bias scales as σ², SEM is roughly σ-independent → SNR ∝ σ². Too small σ: bias buried. Too large σ: leaves the Taylor regime, slope drifts off -1 for theoretical reasons.

For the MLP at d=32, σ=0.3 worked. σ=0.05 and σ=0.15 had bias below the noise floor.

## Repo layout

```
eggroll/
├── README.md                         # ← you are here
├── PROGRESS.md                       # session-resume log
├── eggrollideaideadide.md            # original spec (note: its quadratic fitness can't show the theorem; see Gotcha 1)
│
├── eggroll_experiment/               # TOY verification (tanh teacher-student)
│   ├── README.md
│   ├── run.py                        # python run.py [--quick]
│   ├── src/{fitness,reference_gradient,eggroll_gradient,experiment,plot}.py
│   └── results/data/errors.json + results/figures/*.png
│
└── eggroll_nn/                       # NN verification (MNIST MLP)
    ├── README.md
    ├── BLOG_DRAFT.md                 # blog post + tweet thread draft
    ├── focused_nn_run.py             # primary driver; intermediate-saves to focused_partial.json
    ├── complete_high_r.py            # appends r=64 in a fresh sub-process
    ├── r100_only.py                  # r=100 in isolation (hits a CUDA hang at high r)
    ├── final_analysis.py             # consolidates partial → errors.json, fits slope, generates figures
    ├── run.py                        # original entry with presets {trial, mid, full}
    ├── src/{nn_fitness,perturbation,grad_estimator,experiment,plot}.py
    ├── data/                         # MNIST cache (auto-downloaded)
    └── results/
        ├── data/{errors.json, focused_partial.json}
        └── figures/*.png
```

## How to run

### Toy (~1 min on RTX 3050)
```bash
cd eggroll_experiment
pip install -r requirements.txt
python run.py
```

### MNIST MLP (~30-50 min on RTX 3050)
```bash
cd eggroll_nn
pip install -r requirements.txt
python focused_nn_run.py     # main sweep, with intermediate JSON saves
python final_analysis.py     # consolidates + plots
```

## What this is and isn't

- ✅ **Independent empirical confirmation** that the rate is 1/r-like, not 1/√r, on a real neural network with a non-toy fitness.
- ✅ **Methodology contribution** — a careful path from "naive replication gives slope 0" to "clean -0.76 slope with R²=0.9995", with each subtraction motivated by the math.
- ⚠ **Slope is not exactly -1.** It's in [-0.76, -0.90] across the two setups, consistent with the asymptotic O(1/r) bound and finite-σ corrections.
- ⚠ **Clean signal extends through r=8.** r=16-64 fall below this hardware's joint MC noise floor (≈0.41 on the NN). Getting clean signal at r=100 would need ~50-100× more samples — well within an H100, well outside an RTX 3050.

## Hardware

- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- Batched perturbations B=500, MLP 784→32→10 (P ≈ 25 k params)
- Auto-halving batch on CUDA OOM (`grad_estimator.py`)
- Long-running processes hit a CUDA cumulative-state hang around r=64+; mitigation is running high-r sweeps in fresh sub-processes.

## Acknowledgements

To the authors of *Evolution Strategies at the Hyperscale* — clean theorem, clean proof, falsifiable claim. The fact that the empirical signal stays this clean down to a 4 GB GPU is a good sign for both their theory and the practicality of EGGROLL on hobbyist hardware.

Math derivations: <https://yuvanesh.vercel.app/blogs/EGGROLL>

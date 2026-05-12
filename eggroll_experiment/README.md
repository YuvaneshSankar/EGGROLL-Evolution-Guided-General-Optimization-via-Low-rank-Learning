# EGGROLL O(1/r) Convergence Rate — Empirical Verification

> **I mathematically proved and explained bits and pieces of this in a blog post — please check it out:** <https://yuvanesh.vercel.app/blogs/EGGROLL>
>
> The code in this repo is the empirical side of that write-up. The blog has the derivations; this is the experiment that demonstrates the result on real hardware.

This repo verifies **Theorem 2** of *"Evolution Strategies at the Hyperscale"* (arXiv:2511.16652):

```
|| g_EGGROLL^r - g_True ||_F = O(1 / r)
```

i.e. as the low-rank perturbation rank `r` grows, the EGGROLL gradient estimate converges to the true ES gradient at rate **1/r** — quadratically faster than the naive CLT rate of **1/√r**.

## Result

Run on an RTX 3050 Laptop GPU (4 GB):

```
Bias-only slope : -0.898  (target: -1.0)
R squared       : 0.998
Result: PASSED
```

The slope is fit on the 4 rank values where the EGGROLL bias is significantly above the joint Monte-Carlo noise floor (`r ∈ {1, 2, 4, 8}`). Beyond `r=8` the bias drops below the floor and is no longer measurable.

See `results/figures/convergence_loglog_annotated.png` for the log-log plot.

## How to run

```bash
pip install -r requirements.txt

# Quick smoke test (~30 sec on an RTX 3050)
python run.py --quick

# Full run (default d=32, σ=1.0)
python run.py
```

Flags:
- `--device cuda|cpu` — defaults to `cuda`, falls back to `cpu` if unavailable.
- `--dim N` — weight matrix size (default 32). Larger N makes the MC noise floor higher, requiring much more sampling to see the bias signal.
- `--sigma S` — ES perturbation scale (default 1.0). Bigger σ amplifies the higher-order term that produces the EGGROLL-specific bias.
- `--quick` — small N for fast iteration.

## What the experiment does

1. **Fitness** — a teacher–student MLP. With a fixed teacher `W_*` and inputs `X`, define
   `f(W) = -mean ||tanh(X @ W) - tanh(X @ W*)||_F²`. The tanh nonlinearity is the source of the higher-than-second derivatives that the theorem's bias depends on. (A *quadratic* fitness like `-||W - W_*||²` would give zero EGGROLL bias and the theorem would be invisible.)
2. **Reference gradient** — antithetic full-rank Gaussian ES computed in **10 independent chunks** of `N_ref/10` samples each. Chunking gives us an empirical `SEM_ref` so we can subtract the reference's residual MC noise from the error.
3. **EGGROLL gradient** — antithetic low-rank ES with `E = (1/√r) A Bᵀ`, `A,B ∈ ℝ^{d×r}`, `A,B ~ N(0,1)`. For each `r`, we compute `N_trials` independent estimates (each with `N_eggroll` samples).
4. **Bias extraction** — `err(r)² = bias(r)² + SEM_eg(r)² + SEM_ref²`, so
   `bias(r) = sqrt(max(err² − SEM_eg² − SEM_ref², 0))`. Without this subtraction, the bias signal is buried in the MC variance.
5. **Slope fit** — linear regression of `log(bias)` vs `log(r)` on the `r` values where the signal is clear of the noise floor (`err² > 2 · noise²`).

### Antithetic sampling

Both reference and EGGROLL estimators use
`g = (1/(2σN)) Σᵢ (f(μ + σEᵢ) − f(μ − σEᵢ)) · Eᵢ`. This cancels the `f(μ)·E` baseline-variance term that would otherwise dominate by orders of magnitude.

## Files

```
eggroll_experiment/
├── src/
│   ├── fitness.py              # MLP teacher–student
│   ├── reference_gradient.py   # antithetic full-rank Gaussian ES
│   ├── eggroll_gradient.py     # antithetic low-rank EGGROLL
│   ├── experiment.py           # orchestrates ranks × trials, fits slope
│   └── plot.py                 # convergence + slope figures
├── results/
│   ├── data/errors.json
│   └── figures/*.png
├── requirements.txt
├── run.py                      # single entry point
└── README.md
```

## Hardware

Developed and validated on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM). Batches auto-halve on CUDA OOM. The default `d=32` setting was chosen so the bias signal clears the MC noise floor at the sample budgets we can actually run on 4 GB.

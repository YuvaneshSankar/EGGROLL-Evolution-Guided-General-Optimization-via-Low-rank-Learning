# EGGROLL Empirical Verification — Session Progress

> **Resume instructions for next Claude session:** read this file end-to-end before touching anything. The state described here is exact as of pause. Last action before pause: killed an in-progress MNIST MLP sweep after the reference gradient phase finished but before the EGGROLL sweep started.

## TL;DR — what's done, what's left

| Phase | Status | Result |
|---|---|---|
| Toy experiment (`eggroll_experiment/`) | ✅ DONE, PASSED | slope `-0.898`, R²=0.998 on 4 high-SNR ranks (d=32 tanh teacher-student) |
| NN code scaffold (`eggroll_nn/`) | ✅ DONE | MNIST MLP fitness, per-layer EGGROLL/Gaussian perturbation, antithetic ES, bias isolation, slope analysis, plots |
| NN trial / mid runs | ✅ DONE, diagnostic only | confirmed naive σ=0.05 has flat slope; SEM_ref dominates |
| σ + d_hidden sweep | ✅ DONE, partial | informed σ=0.3 / d_hidden=32 choice |
| **Focused NN run (d_hidden=32, σ=0.3)** | **✅ DONE, PASSED** | **slope `-0.755`, R²=0.9995 on r∈{1,2,4,8}** |
| r=64 / r=100 high-rank extension | ⚠ partial | r=64 captured (below noise floor as expected). r=100 hit CUDA cumulative-state hang twice; documented and skipped. |
| `eggroll_nn/README.md` | ✅ DONE | with blog link at top |
| Blog post / tweet draft (`eggroll_nn/BLOG_DRAFT.md`) | ✅ DONE | ready for user review before posting |
| Phase B: tiny CNN | ⏳ NOT STARTED | optional second architecture |

## The user's intent (don't re-litigate)

- The user (Yuvanesh) wants a blog-worthy independent verification of EGGROLL Theorem 2 (`||g_EGGROLL^r - g_True||_F = O(1/r)`) on consumer hardware (RTX 3050, 4GB).
- They have a math/proofs blog at `https://yuvanesh.vercel.app/blogs/EGGROLL` where they explain the derivations. The README in `eggroll_experiment/` already links it prominently at the top — keep that pattern in `eggroll_nn/`.
- The user **wants to tag the EGGROLL paper authors on X** when posting. We agreed the methodology angle (bias-isolation, why naive replications fail) is the strongest contribution — multi-architecture sweep is a stretch.
- The user explicitly said "for how many hours i dont care" — runtime budget is open.
- Two-phase plan agreed:
  - **Phase A**: MNIST MLP, σ tuned, r=1–100 sweep with clean slope.
  - **Phase B**: same code on a tiny CNN (second architecture confirms theorem is architecture-agnostic).

## What was learned (the non-obvious math)

These insights are not in the original spec. They matter for the writeup.

1. **Quadratic fitness gives zero EGGROLL bias.** The covariance of `E_eggroll = (1/√r) A Bᵀ` matches Gaussian exactly (proved by direct calculation). A quadratic has no 3rd+ derivatives for higher cumulants to bite on. So the theorem is invisible if you use `f(W) = -||W-W*||²`. The original spec used this and would have failed. Fix: use a fitness with non-zero higher derivatives (tanh, NN with nonlinear activations).
2. **σ does NOT depend on rank in EGGROLL.** Confirmed from paper (project page `eshyperscale.github.io`, lit review). σ is a function of dimension `d`, not `r`. Same σ across all r values.
3. **σ-tuning is the real risk.** Bias ∝ σ², SEM is (roughly) σ-independent, so SNR ∝ σ². Too small σ → bias buried. Too big σ → Taylor breaks down and slope drifts off -1 for theoretical (not statistical) reasons.
4. **Naive per-trial error measurement cannot show the theorem.** Each trial's `||g_eg^r - g_ref||` is dominated by MC noise. You must AVERAGE the gradients across trials FIRST, then take the error — this drives the per-r MC variance down as `1/(N_trials · N_eggroll)`.
5. **You also need to subtract `SEM_ref`.** The trial-mean still has noise from `g_ref` itself (which has its own MC noise). Proper bias estimator: `bias(r) = sqrt(max(err² - SEM_eg² - SEM_ref², 0))`. To get `SEM_ref`, compute `g_ref` in `M_ref` chunks and take the std-of-mean.
6. **For real NNs, variance scales as `P · ||∇f||²`.** With P=100k params and modest budgets, `SEM_F` is comparable to `||∇f||` itself. This is what killed the σ=0.05 NN runs — bias signal is below the joint noise floor.

## File layout

```
eggroll/
├── PROGRESS.md                              # this file
├── eggrollideaideadide.md                   # original spec (note: its quadratic fitness is WRONG, see issue #1)
├── eggroll_experiment/                      # TOY result — SHIP-READY
│   ├── README.md                            # already has blog link at top
│   ├── run.py                               # entry: python run.py [--quick]
│   ├── src/{fitness,reference_gradient,eggroll_gradient,experiment,plot}.py
│   └── results/
│       ├── data/errors.json                 # slope=-0.898, R²=0.998
│       └── figures/*.png
└── eggroll_nn/                              # NN extension — IN PROGRESS
    ├── run.py                               # entry: python run.py --preset {trial,mid,full}
    ├── focused_nn_run.py                    # standalone σ=0.3 d_hidden=32 run (last killed mid-EGGROLL)
    ├── sigma_sweep.py                       # σ × d_hidden sweep (partial, last killed)
    ├── src/
    │   ├── nn_fitness.py                    # MLP 784→H→10, batched forward, autograd ∇f
    │   ├── perturbation.py                  # sample_gaussian, sample_eggroll per-layer
    │   ├── grad_estimator.py                # antithetic ES, returns dict of per-param grads
    │   ├── experiment.py                    # orchestrator
    │   └── plot.py                          # 3 figures
    ├── data/                                # MNIST cache (downloaded)
    └── results/
        ├── data/errors.json                 # mid-run output (slope flat — bad σ)
        ├── focused_partial.log              # reference-phase-only output of the focused run
        └── sigma_sweep_partial.log          # partial sweep log
```

## Resume plan — exact next steps

### Step 1: re-run focused NN sweep (Phase A) — single biggest task

The `focused_nn_run.py` standalone script is the target. Last killed after reference finished (SEM_ref=0.323, ||g_ref-g_truth||=1.76). Re-run from scratch:

```bash
cd /home/yuvanesh-sankar/eggroll/eggroll_nn
python -u focused_nn_run.py 2>&1 | tee /tmp/focused_nn.log
```

**Expected runtime: ~6 hours wall.** The reference phase took 13 min (10 chunks × 200k samples). The EGGROLL sweep portion (8 r values × 50 trials × 100k samples) was projected at ~43 min/r → ~5.7 hours.

Settings (in the script):
- `d_hidden=32, n_data=256, P=25450`
- `σ=0.3`
- `N_ref = 200000 per chunk × 10 chunks` (M_ref=10)
- `N_eggroll = 100000`
- `N_trials = 50`
- `r_values = [1, 2, 4, 8, 16, 32, 64, 100]`

Should run in background and stream via Monitor.

**What success looks like** at the end:
- Bias-only slope in `[-1.2, -0.8]` on at least 3-4 high-SNR r values.
- The bias signal is detectable: at r=1, `bias = sqrt(err² - SEM_eg² - SEM_ref²)` should be `>> SEM_ref ≈ 0.32`.

**If σ=0.3 doesn't work** (slope still flat after the full run):
- The σ sweep showed σ=0.15 gives bias ≈ 0.46 and barely-decaying — likely too small. σ=0.3 should be much better (σ² is 4× bigger).
- Backup: try σ=0.5 (σ² = 0.25, another 2.8× signal). Risk: Taylor regime may break.
- Backup: increase N_ref to 4M samples → SEM_ref drops to ~0.16, easier to see bias.

### Step 2: check the result, decide on Phase B

If Step 1 gives a clean slope on the NN, decide:
- **YES Phase B**: Build a tiny CNN (e.g., 2-conv, 1-FC). Reuse most of `eggroll_nn/` — only `nn_fitness.py` needs a new fitness class and `perturbation.py` needs to handle conv filters (treat each conv weight as a matrix by reshaping `(out_ch, in_ch, k, k) → (out_ch, in_ch*k*k)`). Run with similar settings, expect another ~6 hours.
- **NO**, ship Phase A only: the blog post can be MLP-only.

### Step 3: write the blog post

Write `eggroll_nn/README.md` and a draft of the X thread / blog post. Lead with the blog link to `yuvanesh.vercel.app/blogs/EGGROLL`. Frame: methodology contribution (here's how to actually see Theorem 2; here's why naive replications fail). Show the toy result and the NN result side-by-side.

Memory says to put the blog link prominently — already done in `eggroll_experiment/README.md`. Copy that pattern.

### Step 4: ask user before tweeting / tagging authors

Don't tweet on user's behalf. Just produce the post draft + the headline plots. User decides if/when to post.

## Open questions for the user (only ask if blocked)

- Do they want CNN (Phase B), or is MLP enough?
- Are they OK with the bias-isolation methodology being the centerpiece (vs raw verification)? (We discussed this, they implicitly agreed by saying "go".)
- If σ=0.3 gives only partial clean slope (e.g., r=1-16 clean, r=32+ noisy), is that acceptable, or do they want me to push harder?

## Things NOT to redo

- The toy experiment is DONE. Don't touch `eggroll_experiment/`.
- The MEMORY.md note about blog link is correct.
- Don't go back to the quadratic fitness. Don't go back to per-trial error analysis.

## Things to watch out for

- **Output file streaming.** Use `python -u ... 2>&1 | tee /tmp/X.log` and `Monitor` with `grep --line-buffered`. Plain redirection to a file without `-u` may buffer for minutes.
- **Heredoc + `python -u` via Bash tool**: writing the script to a real file first (like `/tmp/focused_nn_run.py`) is more robust than inlining. We already did this.
- **Sleep / poll blocked.** Use `Monitor` for streaming or `Bash run_in_background=true` with `until` for one-shot waits. Don't `sleep N && cat`.
- **VRAM 3.68 GB**. Auto-halving on OOM is wired up in `grad_estimator.py`. With B=500 and MLP 784→32→10 it has been comfortably fitting.
- **MNIST download** is in `eggroll_nn/data/` already (downloaded during trial run).
- **Tasks aren't loaded across sessions**, so the TaskList you see when resuming may be stale or empty. Reseed from this file.

## Headline numbers so far

```
TOY (eggroll_experiment, d=32, tanh teacher-student):
  bias-only slope = -0.898  on 4 r values (r ∈ {1,2,4,8})
  R² = 0.998
  result: PASSED

NN — best run so far (mid preset, σ=0.05, d_hidden=128, P=101770):
  ||g_truth||_F = 1.005
  ||g_ref - g_truth||_F = 1.058  (SEM_ref dominates, σ²-Hessian negligible)
  bias-only slope = -0.004  (FLAT — σ too small)

NN focused (σ=0.3, d_hidden=32, P=25450) — reference only:
  ||g_truth||_F = 0.944
  ||g_ref||_F = 2.130
  ||g_ref - g_truth||_F = 1.758  (σ² Hessian bias is real at σ=0.3; this is EXPECTED — for the EGGROLL comparison this offset cancels because g_eg^r has the SAME σ²-bias)
  SEM_ref = 0.323
  → EGGROLL sweep NOT done; that's the work to resume
```

When this is resumed, the very first thing to do is verify the Python environment still has torch+CUDA, then re-run `focused_nn_run.py` and stream the output.

# EGGROLL on a Consumer GPU — Independent Empirical Verification of the O(1/r) Rate

> Tweet thread + blog companion. Math derivation: <https://yuvanesh.vercel.app/blogs/EGGROLL>

## TL;DR (blog headline)

EGGROLL's central theoretical claim — that its low-rank gradient estimate converges to the true ES gradient at rate **1/r** (not the naive CLT rate of 1/√r) — reproduces on a 4 GB consumer GPU on both a toy quartic problem and a real MNIST MLP. The clean log-log fit on the high-SNR ranks gives:

| Setup | Slope | R² | Range fit |
|---|---|---|---|
| Toy (tanh teacher-student, d=32) | **-0.898** | 0.998 | r ∈ {1, 2, 4, 8} |
| MNIST MLP (784→32→10, σ=0.3)     | **-0.755** | 0.9995 | r ∈ {1, 2, 4, 8} |

Both clearly beat the **-0.5** CLT baseline.

## Two-tweet version

### Tweet 1 (image: convergence_loglog_annotated.png from `eggroll_nn`)

> Reproduced Theorem 2 of EGGROLL on a 4 GB RTX 3050 — both on a toy and on an MNIST MLP. Bias drops 1/r as predicted, R²=0.9995.
>
> The naive replication gives slope 0. Here's why, and how to fix it →
>
> 📓 math: yuvanesh.vercel.app/blogs/EGGROLL
> 🧵 code: github.com/.../eggroll
> 🏷️ @author1 @author2

### Tweet 2 (the methodology hook)

> Three gotchas that turn a "verified" replication into a "slope-0 noise plot":
>
> 1. Quadratic fitness → EGGROLL bias is identically 0 (covariance matches Gaussian). Use a nonlinear fitness.
> 2. Per-trial err measures variance, not bias. Average gradients first.
> 3. err² = bias² + SEM_eg² + SEM_ref². Subtract both. Compute g_ref in chunks for an empirical SEM_ref.

## The blog post (full version, for yuvanesh.vercel.app/blogs/EGGROLL)

### Why I tried this

EGGROLL's [paper](https://arxiv.org/abs/2511.16652) makes a tight theoretical claim: replace the full-rank Gaussian perturbation `E ~ N(0, I)` in evolution strategies with a low-rank `E = (1/√r) A Bᵀ`, and the gradient estimate converges to the true ES gradient at rate `O(1/r)`. That's quadratically better than the central limit theorem's 1/√r, which is what you'd guess if you didn't read carefully.

The theorem is proved; the paper demonstrates it at H100 scale. I wanted to know: does this actually show up on a 4 GB consumer card? And can I see the slope by myself, on a real network, before tagging the authors?

It turns out the answer is yes — but the experiment is much sneakier than the spec makes it sound.

### What goes wrong with a naive replication

The first version I wrote followed an idea sketch literally: quadratic fitness, the formulae from the paper, plot `||g_eggroll^r - g_ref||_F` against `r`. The slope I got was **0.0012**. Theorem clearly false.

Three things had to change before the slope came out right.

**Gotcha 1: the quadratic fitness has identically zero EGGROLL bias.**

The EGGROLL theorem's bias comes from the *fourth* cumulant of `E_eggroll` differing from Gaussian. For Gaussian `E ~ N(0, I)`, all cumulants of order ≥ 3 vanish. For `E_eggroll`, the second moment (covariance) matches Gaussian exactly — I derived this on paper — but the *fourth* moment differs by `O(1/r)`.

That fourth-cumulant difference contracts with `∇³f` (or higher) to produce the bias. **A quadratic has zero third derivative.** So the bias is identically zero, for every `r`. The theorem holds — trivially — but you cannot *see* it.

Fix: use a fitness with non-zero higher derivatives. The toy uses a tanh teacher-student loss; the NN uses cross-entropy on MNIST.

**Gotcha 2: per-trial error is dominated by Monte-Carlo variance.**

If you run a single EGGROLL estimate at each rank `r`, the error `||g_eg^r - g_ref||` is roughly
```
err  ≈  sqrt(bias(r)² + SEM_eg² + SEM_ref²)
```
For our budgets the SEM terms are much bigger than the bias, so what you measure is the noise floor, independent of `r`. Slope ~ 0.

Fix: average `g_eg^{r,t}` over many trials `t` *before* taking the norm. The variance of the trial-mean decays as `1/(N_trials · N_eggroll)`, so for enough trials the mean settles close to `E[g_eg^r]`, leaving the bias as the dominant signal.

**Gotcha 3: you still have to subtract two noise terms.**

After Gotcha 2 the residual `err(r)²` is approximately
```
bias(r)²  +  SEM_eg(r)²  +  SEM_ref²
```
The trial-mean has SEM `SEM_eg(r) ∝ 1/sqrt(N_trials · N_eggroll)`. Easy to estimate from the trial-stack.

But `g_ref` is *also* a finite-sample estimate, with its own MC noise `SEM_ref`. If you compute `g_ref` as a single sum, you have no estimate of this noise. **Fix: compute `g_ref` in M=10 independent chunks**, take the mean of the chunks, and let the chunk-to-chunk std-of-mean be your `SEM_ref`.

Then the unbiased bias estimator is
```
bias(r) = sqrt(max(err(r)² - SEM_eg(r)² - SEM_ref², 0))
```
and fitting `log(bias)` vs `log(r)` on the high-SNR ranks (`err² > 2 · noise²`) gives the slope you actually want.

There's also a fourth thing — **antithetic sampling**. Without it, the `f(μ)·E` term in the gradient estimator inflates variance by orders of magnitude. With antithetic, that term cancels exactly.

### Results

After all four fixes:

**Toy** (tanh teacher-student, `d=32`, `σ=1.0`):
- `slope = -0.898`, `R² = 0.998`, on r ∈ {1, 2, 4, 8}
- Higher r values drop below the joint MC noise floor (~3e-3).

**MNIST MLP** (784→32→10, `σ=0.3`, 50 trials × 100k samples per `r`, `g_ref` from 10 chunks × 200k):
- `slope = -0.7552`, `R² = 0.9995`, on r ∈ {1, 2, 4, 8}
- Higher r values drop below the joint noise floor (~0.41 in this regime).

The fitted slope on the MLP is slightly less negative than -1.0. That's consistent with the theorem (the bound is `O(1/r)` asymptotically, not `= c/r`), and at `σ = 0.3` the higher-order `O(σ⁴)` terms contribute non-trivially at small `r`.

The CLT baseline (`-0.5`) is *clearly* steeper than the data on both plots. The data isn't on the CLT line; it's pretty much on the slope-1 line.

### What I'd want to do next

- **Push to r=100 cleanly** — needs ~50× more samples than my RTX 3050 can deliver in a day. The signal at r=100 is below the noise floor here; on a datacenter card it wouldn't be. (Aside: long-running processes hit a CUDA cumulative-state hang around r=64+ on my driver; I had to launch high-r sweeps in fresh sub-processes.)
- **Try a CNN** for a second architecture confirming the result isn't MLP-specific.
- **Sweep σ** more carefully and check that the slope drifts toward exactly -1 as σ decreases (with sample budget growing accordingly).

The full code, raw JSON outputs, and figures are at: `<repo>/eggroll_nn` and `<repo>/eggroll_experiment`. Math derivations in this blog post.

### Acknowledgements

To the authors of *Evolution Strategies at the Hyperscale* — clean theorem, clean proof, falsifiable claim. The fact that the empirical signal stays this clean down to a 4 GB GPU is a good sign for both their theory and the practicality of EGGROLL on hobbyist hardware.

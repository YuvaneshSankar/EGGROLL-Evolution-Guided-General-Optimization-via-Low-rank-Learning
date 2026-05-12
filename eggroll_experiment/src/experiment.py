"""Empirical verification of Theorem 2 (||g_EGGROLL^r - g_True||_F = O(1/r)).

Design notes
------------
- Fitness: small tanh teacher–student. f(W) = -mean ||tanh(X W) - tanh(X W*)||².
- ES gradient: antithetic full-rank Gaussian, N_ref samples → `g_ref`.
- EGGROLL gradient: antithetic low-rank E = (1/√r) A Bᵀ.
- Ground truth: PyTorch autograd of f at μ → `g_truth` (exact).
- For each r, we AVERAGE the EGGROLL gradient estimator across N_trials
  independent trials (each with N_eggroll samples). The mean-of-trials drives
  MC variance down as 1/(N_trials·N_eggroll), so ||g_eg_mean^r - g_ref|| is
  dominantly the *bias* — exactly what Theorem 2 bounds.
- Slope fit reports both raw slope and **noise-floor-corrected** slope. With
  finite N the trial-mean still has SEM > 0 and we measure
  err² ≈ bias² + SEM². The noise-corrected bias is sqrt(max(err² - SEM², 0));
  fitting that vs log r recovers the underlying O(1/r) law.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import linregress
from tqdm import tqdm

from .eggroll_gradient import compute_eggroll_gradient
from .fitness import MLPFitness
from .reference_gradient import compute_reference_gradient


DEFAULT_R_VALUES = [1, 2, 4, 8, 16, 32, 64, 100]


def run_experiment(
    *,
    device,
    matrix_size=(32, 32),
    sigma=1.0,
    N_ref=200000,
    N_eggroll=5000,
    N_trials=20,
    r_values=None,
    batch_size_ref=2000,
    batch_size_eggroll=2000,
    n_data=32,
    seed=0,
    out_json_path="results/data/errors.json",
):
    r_values = list(r_values) if r_values is not None else list(DEFAULT_R_VALUES)
    torch.manual_seed(seed)
    np.random.seed(seed)

    d1, d2 = matrix_size
    assert d1 == d2, "MLP fitness assumes square W"
    d = d1

    fitness = MLPFitness(d=d, n_data=n_data, device=device, seed=seed)
    mu = (torch.randn(d, d, device=device) / (d ** 0.5)).contiguous()

    print(f"\nFitness: MLP teacher–student, d={d}, n_data={n_data}, sigma={sigma}")
    print(f"  f(μ)               = {float(fitness(mu).item()):.6f}")

    g_truth = fitness.analytical_gradient(mu)
    truth_norm = float(torch.linalg.norm(g_truth).item())
    print(f"  ||g_autograd||_F   = {truth_norm:.6f}")

    # Compute the reference gradient as a mean of M_ref independent chunks.
    # We need an empirical SEM_ref to properly subtract the g_ref noise from
    # ||g_eg_mean - g_ref|| later — otherwise the bias signal is masked by it.
    M_ref = 10
    N_ref_per_chunk = N_ref // M_ref
    print(f"\nComputing reference ES gradient: {M_ref} antithetic chunks × {N_ref_per_chunk} samples ...")
    t0 = time.time()
    g_ref_chunks = []
    for k in range(M_ref):
        g_k = compute_reference_gradient(
            mu, fitness, sigma, N_ref_per_chunk, batch_size_ref, device,
            desc=f"ref chunk {k+1}/{M_ref}",
        )
        g_ref_chunks.append(g_k)
    g_ref_stack = torch.stack(g_ref_chunks, dim=0)
    g_ref = g_ref_stack.mean(dim=0)
    sem_ref_F = float(torch.sqrt(g_ref_stack.var(dim=0, unbiased=True).sum() / M_ref).item())
    t_ref = time.time() - t0
    print(f"  done in {t_ref:.1f}s")
    ref_norm = float(torch.linalg.norm(g_ref).item())
    ref_vs_truth = float(torch.linalg.norm(g_ref - g_truth).item())
    print(f"  ||g_ref||_F              = {ref_norm:.6f}")
    print(f"  ||g_ref - g_autograd||_F = {ref_vs_truth:.6f}  (full-rank ES σ²-bias + N_ref MC)")
    print(f"  SEM_ref                  = {sem_ref_F:.6f}  (MC noise of g_ref mean)")
    sanity_passed = ref_vs_truth < truth_norm
    if not sanity_passed:
        print(f"  [WARN] ||g_ref - g_autograd|| exceeds ||g_truth|| — consider bigger N_ref / smaller σ.")

    print(f"\nComputing EGGROLL estimates for r in {r_values} ({N_trials} trials each) ...")
    err_to_ref = []
    err_to_truth = []
    sem_estimates = []
    per_trial_to_ref = []

    for r in r_values:
        stack = []
        per_trial = []
        for t in tqdm(range(N_trials), desc=f"r={r}", leave=False):
            g_eg = compute_eggroll_gradient(
                mu, fitness, sigma, r, N_eggroll, batch_size_eggroll, device,
                desc=f"r={r} t={t}",
            )
            stack.append(g_eg)
            per_trial.append(float(torch.linalg.norm(g_eg - g_ref).item()))

        stack_t = torch.stack(stack, dim=0)
        g_mean = stack_t.mean(dim=0)
        sem_F = float(torch.sqrt(stack_t.var(dim=0, unbiased=True).sum() / N_trials).item())
        err_ref = float(torch.linalg.norm(g_mean - g_ref).item())
        err_truth = float(torch.linalg.norm(g_mean - g_truth).item())
        err_to_ref.append(err_ref)
        err_to_truth.append(err_truth)
        sem_estimates.append(sem_F)
        per_trial_to_ref.append(per_trial)
        print(f"  r={r:4d}   err_to_ref={err_ref:.4e}   "
              f"err_to_truth={err_truth:.4e}   SEM={sem_F:.4e}")

    # Bias extraction.
    #   err² = bias² + SEM_eg² + SEM_ref²    (independent noise terms)
    # so the unbiased bias estimate is sqrt(max(err² - SEM_eg² - SEM_ref², 0)).
    err_arr = np.array(err_to_ref, dtype=float)
    sem_arr = np.array(sem_estimates, dtype=float)
    noise_sq = sem_arr ** 2 + sem_ref_F ** 2
    bias_sq = err_arr ** 2 - noise_sq
    # When err² is below the noise floor (≈ 0 within noise), report the noise
    # floor itself as the upper-bound on bias. Avoids absurd 1e-15 plot points.
    floor_proxy = np.sqrt(noise_sq) * 0.1
    bias_only = np.sqrt(np.maximum(bias_sq, floor_proxy ** 2))

    # Only fit slope on r values where the bias is significantly above the
    # joint noise floor. Else we're fitting noise.
    significance = err_arr ** 2 > 2.0 * noise_sq
    if significance.sum() < 3:
        # Fall back: fit on first 4 r values, the natural high-signal regime.
        significance = np.zeros_like(significance)
        significance[:4] = True

    log_r_all = np.log(np.array(r_values, dtype=float))
    log_r_fit = log_r_all[significance]
    log_b_fit = np.log(bias_only[significance])

    fit_raw = linregress(log_r_all, np.log(err_arr))
    fit_bias = linregress(log_r_fit, log_b_fit)

    slope_raw = float(fit_raw.slope)
    r2_raw = float(fit_raw.rvalue ** 2)
    slope_bias = float(fit_bias.slope)
    r2_bias = float(fit_bias.rvalue ** 2)

    print(f"\nSlope fits:")
    print(f"  Raw (||mean - g_ref||):                   slope={slope_raw:.4f}  R²={r2_raw:.4f}")
    print(f"  Bias-only on {significance.sum()}/{len(r_values)} high-signal r values:    "
          f"slope={slope_bias:.4f}  R²={r2_bias:.4f}")
    print(f"  (expected: -1.0)")

    err_at_r1 = err_to_ref[0]
    clt_baseline = [err_at_r1 / (r ** 0.5) for r in r_values]

    config = {
        "matrix_size": list(matrix_size),
        "sigma": sigma,
        "N_ref": N_ref,
        "N_eggroll": N_eggroll,
        "N_trials": N_trials,
        "r_values": r_values,
        "device": str(device),
        "fitness_function": "MLP teacher-student: -mean ||tanh(X W) - tanh(X W*)||^2",
        "n_data": n_data,
        "error_metric": "||mean_t g_eg^{r,t} - g_ref||_F  (trial-averaged to isolate bias)",
    }

    out = {
        "experiment_config": config,
        "sanity_check": {
            "ref_vs_autograd_error": ref_vs_truth,
            "autograd_norm": truth_norm,
            "ref_norm": ref_norm,
            "sem_ref": sem_ref_F,
            "passed": bool(sanity_passed),
        },
        "results": {
            "r_values": r_values,
            "mean_errors": err_to_ref,
            "std_errors": sem_estimates,
            "sem_ref": sem_ref_F,
            "bias_only": bias_only.tolist(),
            "fit_mask": significance.tolist(),
            "err_to_truth": err_to_truth,
            "clt_baseline": clt_baseline,
            "per_trial_errors": per_trial_to_ref,
        },
        "slope_fit": {
            "fitted_slope_raw": slope_raw,
            "r_squared_raw": r2_raw,
            "fitted_slope": slope_bias,
            "r_squared": r2_bias,
            "expected_slope": -1.0,
            "std_err": float(fit_bias.stderr),
            "intercept": float(fit_bias.intercept),
            "n_points_fit": int(significance.sum()),
        },
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json_path}")

    return out

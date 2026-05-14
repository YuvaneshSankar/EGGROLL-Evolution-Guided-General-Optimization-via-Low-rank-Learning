"""Empirical verification of EGGROLL Theorem 2 on a real MNIST MLP.

For each r ∈ r_values:
  - Run N_trials independent EGGROLL gradient estimates (each with N_eggroll samples).
  - Take the trial-mean g_eg_mean^r.
  - Compute err(r) = ||g_eg_mean^r - g_ref||_F.
g_ref is computed as the mean of M_ref independent chunks of full-rank antithetic
Gaussian ES (gives us SEM_ref).

Slope fit: bias(r)² = err(r)² - SEM_eg(r)² - SEM_ref², then linregress log(bias) vs log(r)
on r values where the bias is significantly above the noise floor.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import linregress
from tqdm import tqdm

from .grad_estimator import (
    diff,
    estimate_gradient,
    flatten_grad,
    mean_grad,
    norm_F,
    sem_grad,
    stack_grads,
)
from .nn_fitness import Fitness, load_mnist_batch
from .perturbation import sample_eggroll, sample_gaussian


DEFAULT_R_VALUES = [1, 2, 4, 8, 16, 32, 64, 100]


def run_experiment(
    *,
    device,
    sigma=0.05,
    d_hidden=128,
    n_data=256,
    N_ref=100000,
    M_ref=10,
    N_eggroll=20000,
    N_trials=80,
    r_values=None,
    batch_size_ref=500,
    batch_size_eggroll=500,
    seed=0,
    data_dir="data",
    out_json_path="results/data/errors.json",
):
    r_values = list(r_values) if r_values is not None else list(DEFAULT_R_VALUES)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nLoading MNIST batch (n_data={n_data}) ...")
    X, Y = load_mnist_batch(n_data, device, data_dir)
    fitness = Fitness(X, Y, d_hidden=d_hidden, device=device, seed=seed)
    print(f"  Network: 784 → {d_hidden} → 10  ({sum(v.numel() for v in fitness.params.values())} params)")

    f0 = float(fitness(fitness.params).item())
    print(f"  f(μ) = {f0:.6f}")

    g_truth = fitness.analytical_gradient()
    truth_norm = norm_F(g_truth)
    print(f"  ||∇f||_F (autograd) = {truth_norm:.6e}")

    # -------- Reference ES gradient: chunked so we can estimate SEM_ref --------
    N_ref_per_chunk = N_ref // M_ref
    print(f"\nReference ES (antithetic Gaussian) — {M_ref} chunks × {N_ref_per_chunk} samples, σ={sigma}")
    t0 = time.time()
    g_ref_chunks = []
    for k in range(M_ref):
        g_k = estimate_gradient(
            fitness, sigma, N_ref_per_chunk, batch_size_ref,
            perturb_sampler=lambda B: sample_gaussian(fitness.param_shapes, B, device),
            device=device, desc=f"ref chunk {k+1}/{M_ref}",
        )
        g_ref_chunks.append(g_k)
    g_ref_stack = stack_grads(g_ref_chunks)
    g_ref = mean_grad(g_ref_stack)
    sem_ref_F = sem_grad(g_ref_stack)
    t_ref = time.time() - t0
    print(f"  done in {t_ref:.1f}s  "
          f"||g_ref||_F={norm_F(g_ref):.6e}  ||g_ref-g_truth||_F={norm_F(diff(g_ref, g_truth)):.6e}  "
          f"SEM_ref={sem_ref_F:.6e}")

    # -------- EGGROLL estimates at each r --------
    err_to_ref = []
    err_to_truth = []
    sem_estimates = []

    print(f"\nEGGROLL sweep: r ∈ {r_values}, N_trials={N_trials}, N_eggroll={N_eggroll}")
    for r in r_values:
        t0 = time.time()
        trials = []
        for t in tqdm(range(N_trials), desc=f"r={r}", leave=False):
            g_t = estimate_gradient(
                fitness, sigma, N_eggroll, batch_size_eggroll,
                perturb_sampler=lambda B: sample_eggroll(fitness.param_shapes, B, r, device),
                device=device, pbar=False,
            )
            trials.append(g_t)
        trials_stack = stack_grads(trials)
        g_mean = mean_grad(trials_stack)
        sem_F = sem_grad(trials_stack)
        err_ref = norm_F(diff(g_mean, g_ref))
        err_truth = norm_F(diff(g_mean, g_truth))
        err_to_ref.append(err_ref)
        err_to_truth.append(err_truth)
        sem_estimates.append(sem_F)
        elapsed = time.time() - t0
        print(f"  r={r:4d}  err_to_ref={err_ref:.4e}  err_to_truth={err_truth:.4e}  "
              f"SEM_eg={sem_F:.4e}  [{elapsed:.0f}s]")

    # -------- Bias extraction & slope fit --------
    err_arr = np.array(err_to_ref, dtype=float)
    sem_arr = np.array(sem_estimates, dtype=float)
    noise_sq = sem_arr ** 2 + sem_ref_F ** 2
    bias_sq = err_arr ** 2 - noise_sq
    floor_proxy = np.sqrt(noise_sq) * 0.1
    bias_only = np.sqrt(np.maximum(bias_sq, floor_proxy ** 2))

    significance = err_arr ** 2 > 2.0 * noise_sq
    if significance.sum() < 3:
        significance = np.zeros_like(significance)
        significance[: max(4, int(np.argmax(err_arr ** 2 < 2.0 * noise_sq)) or len(r_values))] = True

    log_r_all = np.log(np.array(r_values, dtype=float))
    fit_raw = linregress(log_r_all, np.log(err_arr))
    log_r_fit = log_r_all[significance]
    log_b_fit = np.log(bias_only[significance])
    fit_bias = linregress(log_r_fit, log_b_fit)

    print(f"\nSlope fits:")
    print(f"  Raw                                    slope={fit_raw.slope:.4f}  R²={fit_raw.rvalue**2:.4f}")
    print(f"  Bias-only on {significance.sum()}/{len(r_values)} high-SNR r:        "
          f"slope={fit_bias.slope:.4f}  R²={fit_bias.rvalue**2:.4f}")
    print(f"  (expected: -1.0)")

    err_at_r1 = err_to_ref[0]
    clt_baseline = [err_at_r1 / (r ** 0.5) for r in r_values]

    out = {
        "experiment_config": {
            "sigma": sigma,
            "d_hidden": d_hidden,
            "n_data": n_data,
            "N_ref": N_ref,
            "M_ref": M_ref,
            "N_eggroll": N_eggroll,
            "N_trials": N_trials,
            "r_values": r_values,
            "device": str(device),
            "fitness_function": "MNIST MLP: -CrossEntropy on fixed batch",
        },
        "sanity_check": {
            "autograd_norm": truth_norm,
            "ref_norm": norm_F(g_ref),
            "ref_vs_autograd_error": norm_F(diff(g_ref, g_truth)),
            "sem_ref": sem_ref_F,
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
        },
        "slope_fit": {
            "fitted_slope": float(fit_bias.slope),
            "r_squared": float(fit_bias.rvalue ** 2),
            "fitted_slope_raw": float(fit_raw.slope),
            "r_squared_raw": float(fit_raw.rvalue ** 2),
            "expected_slope": -1.0,
            "intercept": float(fit_bias.intercept),
            "std_err": float(fit_bias.stderr),
            "n_points_fit": int(significance.sum()),
        },
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json_path}")
    return out

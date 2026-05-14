"""Focused EGGROLL verification on MNIST MLP.

- d_hidden=32 (P~25k) keeps SEM_F manageable on a 4GB GPU
- σ=0.3 amplifies the EGGROLL bias (signal ∝ σ²) while staying in Taylor regime
- g_ref via 10 chunks → empirical SEM_ref to subtract from err²
- 50 trials × 100k samples per r → SEM_eg empirically estimated and subtracted

Total: ~50 min wall on RTX 3050.
"""
import os
import sys
import time
import json

import numpy as np
import torch
from scipy.stats import linregress
from tqdm import tqdm

sys.path.insert(0, "/home/yuvanesh-sankar/eggroll/eggroll_nn")

from src.nn_fitness import Fitness, load_mnist_batch
from src.perturbation import sample_eggroll, sample_gaussian
from src.grad_estimator import (
    diff, estimate_gradient, mean_grad, norm_F, sem_grad, stack_grads,
)


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    d_hidden = 32
    n_data = 256
    sigma = 0.3
    N_ref_per_chunk = 200000
    M_ref = 10
    N_eggroll = 100000
    N_trials = 50
    r_values = [1, 2, 4, 8, 16, 32, 64, 100]
    batch_size = 500

    out_dir = "/home/yuvanesh-sankar/eggroll/eggroll_nn/results"
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    print(f"Setup: d_hidden={d_hidden}, σ={sigma}, "
          f"N_ref={N_ref_per_chunk}×{M_ref}, N_eg={N_eggroll}, N_trials={N_trials}", flush=True)
    X, Y = load_mnist_batch(n_data, device, "/home/yuvanesh-sankar/eggroll/eggroll_nn/data")
    fitness = Fitness(X, Y, d_hidden=d_hidden, device=device, seed=0)
    P = sum(v.numel() for v in fitness.params.values())
    g_truth = fitness.analytical_gradient()
    g_truth_norm = norm_F(g_truth)
    print(f"P={P}  ||∇f||_F = {g_truth_norm:.4f}", flush=True)

    # ---- chunked reference ----
    t0 = time.time()
    print(f"\nReference: {M_ref} chunks × {N_ref_per_chunk} samples", flush=True)
    g_ref_chunks = []
    for k in range(M_ref):
        g_k = estimate_gradient(
            fitness, sigma, N_ref_per_chunk, batch_size,
            perturb_sampler=lambda B: sample_gaussian(fitness.param_shapes, B, device),
            device=device, pbar=False,
        )
        g_ref_chunks.append(g_k)
        print(f"  chunk {k+1}/{M_ref} done ({time.time()-t0:.0f}s)", flush=True)
    g_ref_stack = stack_grads(g_ref_chunks)
    g_ref = mean_grad(g_ref_stack)
    sem_ref = sem_grad(g_ref_stack)
    print(f"||g_ref||={norm_F(g_ref):.4e}  ||g_ref-g_truth||={norm_F(diff(g_ref, g_truth)):.4e}  "
          f"SEM_ref={sem_ref:.4e}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- per-r EGGROLL sweep ----
    err_to_ref = []
    err_to_truth = []
    sem_estimates = []
    partial_path = os.path.join(out_dir, "data", "focused_partial.json")
    for r in r_values:
        tr = time.time()
        trials = []
        for t in range(N_trials):
            g_t = estimate_gradient(
                fitness, sigma, N_eggroll, batch_size,
                perturb_sampler=lambda B: sample_eggroll(fitness.param_shapes, B, r, device),
                device=device, pbar=False,
            )
            trials.append(g_t)
        ts = stack_grads(trials)
        g_mean = mean_grad(ts)
        sem_eg = sem_grad(ts)
        err_ref = norm_F(diff(g_mean, g_ref))
        err_truth = norm_F(diff(g_mean, g_truth))
        err_to_ref.append(err_ref)
        err_to_truth.append(err_truth)
        sem_estimates.append(sem_eg)
        print(f"  r={r:4d}  err_to_ref={err_ref:.4e}  err_to_truth={err_truth:.4e}  "
              f"SEM_eg={sem_eg:.4e}  [{time.time()-tr:.0f}s, total={time.time()-t0:.0f}s]",
              flush=True)
        # Intermediate save after each completed r — recoverable if killed.
        with open(partial_path, "w") as f:
            json.dump({
                "r_done": r_values[:len(err_to_ref)],
                "err_to_ref": err_to_ref,
                "err_to_truth": err_to_truth,
                "sem_eg": sem_estimates,
                "sem_ref": sem_ref,
                "g_truth_norm": g_truth_norm,
                "g_ref_norm": norm_F(g_ref),
                "ref_vs_truth": norm_F(diff(g_ref, g_truth)),
                "config": {
                    "d_hidden": d_hidden, "sigma": sigma, "P": P,
                    "N_ref_per_chunk": N_ref_per_chunk, "M_ref": M_ref,
                    "N_eggroll": N_eggroll, "N_trials": N_trials,
                    "r_values": r_values,
                },
            }, f, indent=2)

    # ---- analysis ----
    err_arr = np.array(err_to_ref, dtype=float)
    sem_arr = np.array(sem_estimates, dtype=float)
    noise_sq = sem_arr ** 2 + sem_ref ** 2
    bias_sq = err_arr ** 2 - noise_sq
    floor = np.sqrt(noise_sq) * 0.1
    bias_only = np.sqrt(np.maximum(bias_sq, floor ** 2))
    significance = err_arr ** 2 > 2.0 * noise_sq
    if significance.sum() < 3:
        significance = np.zeros_like(significance)
        significance[: max(3, int(np.argmax(err_arr ** 2 < 2.0 * noise_sq)) or len(r_values))] = True

    log_r = np.log(np.array(r_values, dtype=float))
    fit_raw = linregress(log_r, np.log(err_arr))
    fit_bias = linregress(log_r[significance], np.log(bias_only[significance]))
    print(f"\nSlope: raw={fit_raw.slope:.4f} R²={fit_raw.rvalue**2:.4f}", flush=True)
    print(f"Slope: bias-only on {significance.sum()}/{len(r_values)} r: "
          f"{fit_bias.slope:.4f} R²={fit_bias.rvalue**2:.4f}", flush=True)

    out = {
        "config": {
            "d_hidden": d_hidden, "n_data": n_data, "sigma": sigma,
            "N_ref_per_chunk": N_ref_per_chunk, "M_ref": M_ref,
            "N_eggroll": N_eggroll, "N_trials": N_trials, "r_values": r_values, "P": P,
        },
        "sanity": {
            "g_truth_norm": g_truth_norm,
            "g_ref_norm": norm_F(g_ref),
            "ref_vs_truth": norm_F(diff(g_ref, g_truth)),
            "sem_ref": sem_ref,
        },
        "results": {
            "r_values": r_values,
            "err_to_ref": err_to_ref,
            "err_to_truth": err_to_truth,
            "sem_eg": sem_estimates,
            "sem_ref": sem_ref,
            "bias_only": bias_only.tolist(),
            "fit_mask": significance.tolist(),
        },
        "slope_fit": {
            "raw_slope": float(fit_raw.slope), "raw_r2": float(fit_raw.rvalue ** 2),
            "bias_slope": float(fit_bias.slope), "bias_r2": float(fit_bias.rvalue ** 2),
            "intercept": float(fit_bias.intercept),
            "n_fit": int(significance.sum()),
        },
    }
    with open(os.path.join(out_dir, "data", "focused.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s. Saved focused.json", flush=True)


if __name__ == "__main__":
    main()

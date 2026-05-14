"""Complete r=64 and r=100 in a fresh process (the long-running focused_nn_run.py
hung on these due to cumulative CUDA state — confirmed they work in isolation).

Strategy:
- Recompute g_ref with the same seed (deterministic, will produce identical numbers).
- Run r=64 and r=100 sweeps using the same N_trials and N_eggroll as the main run.
- Merge into focused_partial.json so we have all 8 r values.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import linregress

sys.path.insert(0, "/home/yuvanesh-sankar/eggroll/eggroll_nn")
from src.grad_estimator import (
    diff, estimate_gradient, mean_grad, norm_F, sem_grad, stack_grads,
)
from src.nn_fitness import Fitness, load_mnist_batch
from src.perturbation import sample_eggroll, sample_gaussian


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    d_hidden = 32
    sigma = 0.3
    N_ref_per_chunk = 200000
    M_ref = 10
    N_eggroll = 100000
    N_trials = 50
    r_remaining = [64, 100]
    batch_size = 500

    out_dir = "/home/yuvanesh-sankar/eggroll/eggroll_nn/results"
    partial_path = os.path.join(out_dir, "data", "focused_partial.json")

    print("Loading partial state...", flush=True)
    with open(partial_path) as f:
        partial = json.load(f)
    print(f"  r_done so far: {partial['r_done']}", flush=True)

    print(f"\nRebuilding fitness + g_ref (seed=0, deterministic) ...", flush=True)
    X, Y = load_mnist_batch(256, device, "/home/yuvanesh-sankar/eggroll/eggroll_nn/data")
    fitness = Fitness(X, Y, d_hidden=d_hidden, device=device, seed=0)
    P = sum(v.numel() for v in fitness.params.values())
    g_truth = fitness.analytical_gradient()
    g_truth_norm = norm_F(g_truth)
    print(f"  P={P}  ||∇f||={g_truth_norm:.4f}", flush=True)

    t0 = time.time()
    g_ref_chunks = []
    for k in range(M_ref):
        g_k = estimate_gradient(
            fitness, sigma, N_ref_per_chunk, batch_size,
            perturb_sampler=lambda B: sample_gaussian(fitness.param_shapes, B, device),
            device=device, pbar=False,
        )
        g_ref_chunks.append(g_k)
    g_ref_stack = stack_grads(g_ref_chunks)
    g_ref = mean_grad(g_ref_stack)
    sem_ref = sem_grad(g_ref_stack)
    print(f"  g_ref done in {time.time()-t0:.0f}s  SEM_ref={sem_ref:.4e}", flush=True)
    assert abs(sem_ref - partial["sem_ref"]) < 1e-3, \
        f"SEM_ref mismatch: got {sem_ref}, partial says {partial['sem_ref']}"

    # Sweep remaining r values.
    for r in r_remaining:
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
        partial["r_done"].append(r)
        partial["err_to_ref"].append(err_ref)
        partial["err_to_truth"].append(err_truth)
        partial["sem_eg"].append(sem_eg)
        print(f"  r={r:4d}  err_to_ref={err_ref:.4e}  err_to_truth={err_truth:.4e}  "
              f"SEM_eg={sem_eg:.4e}  [{time.time()-tr:.0f}s]", flush=True)
        with open(partial_path, "w") as f:
            json.dump(partial, f, indent=2)

    # ---- final analysis on the full r sweep ----
    r_values = partial["r_done"]
    err_arr = np.array(partial["err_to_ref"], dtype=float)
    sem_arr = np.array(partial["sem_eg"], dtype=float)
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
    print(f"\nFinal slope fits:", flush=True)
    print(f"  Raw:        slope={fit_raw.slope:.4f}  R²={fit_raw.rvalue**2:.4f}", flush=True)
    print(f"  Bias-only on {significance.sum()}/{len(r_values)} r values:  "
          f"slope={fit_bias.slope:.4f}  R²={fit_bias.rvalue**2:.4f}", flush=True)
    print(f"  (target: -1.0)", flush=True)

    # Write the final consolidated result.
    final = {
        "config": {
            "d_hidden": d_hidden, "sigma": sigma, "P": P,
            "N_ref_per_chunk": N_ref_per_chunk, "M_ref": M_ref,
            "N_eggroll": N_eggroll, "N_trials": N_trials, "r_values": r_values,
        },
        "sanity": {
            "g_truth_norm": g_truth_norm,
            "g_ref_norm": norm_F(g_ref),
            "ref_vs_truth": norm_F(diff(g_ref, g_truth)),
            "sem_ref": sem_ref,
        },
        "results": {
            "r_values": r_values,
            "err_to_ref": partial["err_to_ref"],
            "err_to_truth": partial["err_to_truth"],
            "sem_eg": partial["sem_eg"],
            "sem_ref": sem_ref,
            "bias_only": bias_only.tolist(),
            "fit_mask": significance.tolist(),
            "clt_baseline": [partial["err_to_ref"][0] / (r ** 0.5) for r in r_values],
        },
        "slope_fit": {
            "raw_slope": float(fit_raw.slope), "raw_r2": float(fit_raw.rvalue ** 2),
            "bias_slope": float(fit_bias.slope), "bias_r2": float(fit_bias.rvalue ** 2),
            "intercept": float(fit_bias.intercept),
            "n_fit": int(significance.sum()),
        },
    }
    with open(os.path.join(out_dir, "data", "focused.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved focused.json", flush=True)


if __name__ == "__main__":
    main()

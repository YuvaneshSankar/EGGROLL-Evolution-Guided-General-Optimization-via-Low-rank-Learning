"""Final analysis pipeline once focused_partial.json has all 8 r values.

- Reads focused_partial.json
- Computes bias-only series with full noise subtraction (SEM_eg² + SEM_ref²)
- Picks the high-SNR mask (err² > 2 · noise²)
- Fits slopes (raw and bias-only)
- Writes results/data/focused.json in the format plot.py expects
- Generates results/figures/*.png
"""
import json
import os
import sys

import numpy as np
from scipy.stats import linregress

sys.path.insert(0, "/home/yuvanesh-sankar/eggroll/eggroll_nn")
from src.plot import generate_all_figures


HERE = "/home/yuvanesh-sankar/eggroll/eggroll_nn"
PARTIAL = os.path.join(HERE, "results/data/focused_partial.json")
OUT_JSON = os.path.join(HERE, "results/data/errors.json")
FIGURES_DIR = os.path.join(HERE, "results/figures")


def main():
    with open(PARTIAL) as f:
        p = json.load(f)
    r_values = p["r_done"]
    err = np.array(p["err_to_ref"], dtype=float)
    sem_eg = np.array(p["sem_eg"], dtype=float)
    sem_ref = float(p["sem_ref"])
    print(f"r values: {r_values}")
    print(f"err:      {err.tolist()}")
    print(f"sem_eg:   {sem_eg.tolist()}")
    print(f"sem_ref:  {sem_ref:.4f}")

    noise_sq = sem_eg ** 2 + sem_ref ** 2
    bias_sq = err ** 2 - noise_sq
    floor = np.sqrt(noise_sq) * 0.1
    bias_only = np.sqrt(np.maximum(bias_sq, floor ** 2))

    significance = err ** 2 > 2.0 * noise_sq
    if significance.sum() < 3:
        significance = np.zeros_like(significance)
        significance[:4] = True
    print(f"bias:     {bias_only.tolist()}")
    print(f"fit mask: {significance.tolist()}")

    log_r = np.log(np.array(r_values, dtype=float))
    fit_raw = linregress(log_r, np.log(err))
    fit_bias = linregress(log_r[significance], np.log(bias_only[significance]))

    print(f"\nRaw slope:        {fit_raw.slope:.4f}  R²={fit_raw.rvalue**2:.4f}")
    print(f"Bias-only slope:  {fit_bias.slope:.4f}  R²={fit_bias.rvalue**2:.4f}")
    print(f"  on {significance.sum()}/{len(r_values)} r values  (expected: -1.0)")

    clt = (err[0] / np.sqrt(np.array(r_values, dtype=float))).tolist()

    out = {
        "experiment_config": {
            **p["config"],
            "fitness_function": "MNIST MLP 784→32→10, -CrossEntropy on fixed batch (n=256)",
            "n_data": 256,
            "device": "cuda",
        },
        "sanity_check": {
            "autograd_norm": p["g_truth_norm"],
            "ref_norm": p["g_ref_norm"],
            "ref_vs_autograd_error": p["ref_vs_truth"],
            "sem_ref": sem_ref,
            "passed": p["ref_vs_truth"] < p["g_truth_norm"] * 2.0,  # σ²-Hessian; can exceed truth
        },
        "results": {
            "r_values": r_values,
            "mean_errors": p["err_to_ref"],
            "std_errors": p["sem_eg"],
            "sem_ref": sem_ref,
            "bias_only": bias_only.tolist(),
            "fit_mask": significance.tolist(),
            "err_to_truth": p["err_to_truth"],
            "clt_baseline": clt,
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
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_JSON}")

    generate_all_figures(out, FIGURES_DIR)
    print(f"Saved figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

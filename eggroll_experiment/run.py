"""Single entry point for the EGGROLL O(1/r) convergence experiment."""
import argparse
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.experiment import run_experiment, DEFAULT_R_VALUES
from src.plot import generate_all_figures


RESULTS_DIR = os.path.join(HERE, "results")
DATA_PATH = os.path.join(RESULTS_DIR, "data", "errors.json")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def parse_args():
    p = argparse.ArgumentParser(description="EGGROLL O(1/r) convergence verification")
    p.add_argument("--device", default="cuda",
                   help="device to use; falls back to cpu if cuda unavailable")
    p.add_argument("--quick", action="store_true",
                   help="quick run: smaller N for fast iteration")
    p.add_argument("--dim", type=int, default=32,
                   help="weight matrix size d (W is d×d). Default 32. Larger d "
                        "needs much more sampling for the bias signal to clear the MC floor.")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="ES perturbation scale")
    return p.parse_args()


def pick_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def print_env(device):
    print(f"Device: {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"GPU:    {name}")
        print(f"VRAM:   {total_mem:.2f} GB")


def main():
    args = parse_args()

    print("Setting up experiment...")
    os.makedirs(os.path.join(RESULTS_DIR, "data"), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = pick_device(args.device)
    print_env(device)

    if args.quick:
        print(f"Mode:   QUICK (d={args.dim}, σ={args.sigma}, N_ref=20k, N_eggroll=2000, N_trials=10)")
        print("Estimated runtime: ~2-3 minutes on RTX 3050")
        kwargs = dict(N_ref=20000, N_eggroll=2000, N_trials=10,
                      batch_size_ref=2000, batch_size_eggroll=2000)
    else:
        print(f"Mode:   FULL (d={args.dim}, σ={args.sigma}, N_ref=200k, N_eggroll=5000, N_trials=40)")
        print("Estimated runtime: ~15-30 minutes on RTX 3050")
        kwargs = dict(N_ref=200000, N_eggroll=5000, N_trials=40,
                      batch_size_ref=2000, batch_size_eggroll=2000)

    data = None
    try:
        data = run_experiment(
            device=device,
            matrix_size=(args.dim, args.dim),
            sigma=args.sigma,
            r_values=DEFAULT_R_VALUES,
            out_json_path=DATA_PATH,
            **kwargs,
        )
        generate_all_figures(data, FIGURES_DIR)
    except Exception:
        traceback.print_exc()
        if data is not None:
            try:
                generate_all_figures(data, FIGURES_DIR)
                print("[info] Saved partial figures despite failure.")
            except Exception:
                traceback.print_exc()
        print("Done (with errors). Partial results saved to results/")
        sys.exit(1)

    slope = data["slope_fit"]["fitted_slope"]
    slope_raw = data["slope_fit"]["fitted_slope_raw"]
    r2 = data["slope_fit"]["r_squared"]
    passed = -1.2 <= slope <= -0.8
    verdict = "PASSED" if passed else "CHECK MANUALLY"

    print("============================================")
    print("EXPERIMENT COMPLETE")
    print(f"Bias-only slope : {slope:.4f}  (target: -1.0)")
    print(f"Raw slope       : {slope_raw:.4f}  (decays only until SEM floor)")
    print( "CLT baseline    : -0.5")
    print(f"R squared       : {r2:.4f}")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Data saved to   : {DATA_PATH}")
    print("============================================")
    print(f"Result: {verdict}")

    print("Done. Results saved to results/")


if __name__ == "__main__":
    main()

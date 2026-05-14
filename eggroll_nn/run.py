"""Entry point: EGGROLL O(1/r) verification on a real MNIST MLP."""
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
DATA_DIR = os.path.join(HERE, "data")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--preset", choices=["trial", "mid", "full"], default="trial",
                   help="trial=~5min smoke, mid=~1h check, full=multi-hour final sweep")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--d-hidden", type=int, default=128)
    p.add_argument("--n-data", type=int, default=256)
    return p.parse_args()


def pick_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def print_env(device):
    print(f"Device: {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"GPU:    {name}")
        print(f"VRAM:   {total_mem:.2f} GB")


PRESETS = {
    "trial": dict(N_ref=20000,  M_ref=5,  N_eggroll=2000,  N_trials=20,
                  batch_size_ref=500, batch_size_eggroll=500),
    "mid":   dict(N_ref=100000, M_ref=10, N_eggroll=10000, N_trials=50,
                  batch_size_ref=500, batch_size_eggroll=500),
    "full":  dict(N_ref=300000, M_ref=15, N_eggroll=40000, N_trials=120,
                  batch_size_ref=500, batch_size_eggroll=500),
}


def main():
    args = parse_args()
    print(f"Setting up — preset={args.preset}, σ={args.sigma}, d_hidden={args.d_hidden}")
    os.makedirs(os.path.join(RESULTS_DIR, "data"), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    device = pick_device(args.device)
    print_env(device)

    cfg = PRESETS[args.preset]
    data = None
    try:
        data = run_experiment(
            device=device, sigma=args.sigma, d_hidden=args.d_hidden, n_data=args.n_data,
            r_values=DEFAULT_R_VALUES, out_json_path=DATA_PATH, data_dir=DATA_DIR,
            **cfg,
        )
        generate_all_figures(data, FIGURES_DIR)
    except Exception:
        traceback.print_exc()
        if data is not None:
            try:
                generate_all_figures(data, FIGURES_DIR)
            except Exception:
                traceback.print_exc()
        sys.exit(1)

    slope = data["slope_fit"]["fitted_slope"]
    slope_raw = data["slope_fit"]["fitted_slope_raw"]
    r2 = data["slope_fit"]["r_squared"]
    npts = data["slope_fit"]["n_points_fit"]
    passed = -1.2 <= slope <= -0.8
    print("============================================")
    print("EXPERIMENT COMPLETE")
    print(f"Bias-only slope : {slope:.4f}  on {npts} r values  (target: -1.0)")
    print(f"Raw slope       : {slope_raw:.4f}")
    print(f"R²              : {r2:.4f}")
    print(f"Result          : {'PASSED' if passed else 'CHECK MANUALLY'}")
    print(f"Figures         : {FIGURES_DIR}/")
    print(f"Data            : {DATA_PATH}")
    print("============================================")


if __name__ == "__main__":
    main()

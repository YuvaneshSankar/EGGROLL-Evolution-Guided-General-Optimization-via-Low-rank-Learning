"""Convergence figures for the NN experiment."""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _setup():
    sns.set_style("whitegrid")


def plot_convergence_loglog(data, out_path, annotated=False, hw_label="RTX 3050, 4GB VRAM"):
    _setup()
    cfg = data["experiment_config"]
    res = data["results"]
    fit = data["slope_fit"]

    r_values = np.array(res["r_values"], dtype=float)
    raw_err = np.array(res["mean_errors"], dtype=float)
    sem = np.array(res["std_errors"], dtype=float)
    sem_ref = float(res.get("sem_ref", 0.0))
    bias_only = np.array(res["bias_only"], dtype=float)
    fit_mask = np.array(res.get("fit_mask", [True] * len(r_values)), dtype=bool)
    clt = np.array(res["clt_baseline"], dtype=float)

    slope = fit["fitted_slope"]
    intercept = fit["intercept"]
    r_fit = r_values[fit_mask]
    fitted_line = np.exp(intercept) * r_fit ** slope

    ref_minus1 = bias_only[fit_mask][0] * r_values ** (-1.0)
    noise_floor = float(np.sqrt(sem.mean() ** 2 + sem_ref ** 2))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(r_values, raw_err, marker="o", color="#4a6fa5", linewidth=1.5,
            alpha=0.7, label=r"$\|\bar g_{r} - g_{ref}\|_F$ (raw)")
    ax.fill_between(r_values, np.maximum(raw_err - sem, 1e-12), raw_err + sem,
                    color="#4a6fa5", alpha=0.12, label="±SEM (trial-mean)")
    ax.plot(r_values[fit_mask], bias_only[fit_mask], marker="s", color="#1f3a93",
            linewidth=2, label=r"bias $\sqrt{\mathrm{err}^2 - \mathrm{SEM}_{eg}^2 - \mathrm{SEM}_{ref}^2}$")
    if (~fit_mask).any():
        ax.plot(r_values[~fit_mask], bias_only[~fit_mask], marker="s", color="#1f3a93",
                linewidth=1, linestyle=":", alpha=0.5, label="bias (below noise floor)")
    ax.plot(r_fit, fitted_line, linestyle="--", color="orange", linewidth=2,
            label=f"fitted O(1/r) on high-SNR r (slope={slope:.3f})")
    ax.plot(r_values, clt, linestyle="--", color="red", linewidth=2,
            label=r"CLT baseline $O(1/\sqrt{r})$")
    ax.plot(r_values, ref_minus1, linestyle=":", color="gray", linewidth=2,
            label="reference slope = -1")
    ax.axhline(noise_floor, color="black", linestyle="-.", linewidth=1, alpha=0.4,
               label=f"joint noise floor ≈ {noise_floor:.2e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank r")
    ax.set_ylabel("error (Frobenius)")
    ax.set_title("EGGROLL on MNIST MLP — bias vs rank r")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    if annotated:
        txt = (
            f"Bias-only slope: {slope:.3f}\n"
            f"Expected: -1.0\n"
            f"R² = {fit['r_squared']:.3f}\n"
            f"Raw slope: {fit['fitted_slope_raw']:.3f}\n"
            f"{hw_label}\n"
            f"MNIST MLP 784→{cfg['d_hidden']}→10, σ={cfg['sigma']}\n"
            f"N_eggroll={cfg['N_eggroll']}, N_trials={cfg['N_trials']}\n"
            f"N_ref={cfg.get('N_ref', cfg.get('N_ref_per_chunk', 0) * cfg.get('M_ref', 1))} "
            f"({cfg.get('M_ref', '?')} chunks)"
        )
        ax.text(
            0.98, 0.98, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_slope_fit(data, out_path):
    _setup()
    res = data["results"]
    fit = data["slope_fit"]

    r_values = np.array(res["r_values"], dtype=float)
    bias_only = np.array(res["bias_only"], dtype=float)
    fit_mask = np.array(res.get("fit_mask", [True] * len(r_values)), dtype=bool)
    log_r = np.log(r_values)
    log_b = np.log(bias_only)

    slope = fit["fitted_slope"]
    intercept = fit["intercept"]
    r2 = fit["r_squared"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_r[fit_mask], log_b[fit_mask], color="#1f3a93", s=60,
               label="high-SNR points (fit)", zorder=3)
    if (~fit_mask).any():
        ax.scatter(log_r[~fit_mask], log_b[~fit_mask], color="lightgray", s=60,
                   label="below noise floor (excluded)", zorder=3)
    xs = np.linspace(log_r[fit_mask].min(), log_r[fit_mask].max(), 100)
    ax.plot(xs, intercept + slope * xs, color="orange", linewidth=2,
            label=f"fit: y = {slope:.3f}·log r + {intercept:.3f}")

    ax.set_xlabel("log(r)")
    ax.set_ylabel("log(bias)")
    ax.set_title(f"Slope fit on noise-corrected bias: slope = {slope:.4f}, R² = {r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_figures(data, figures_dir):
    plot_convergence_loglog(data, os.path.join(figures_dir, "convergence_loglog.png"),
                            annotated=False)
    plot_convergence_loglog(data, os.path.join(figures_dir, "convergence_loglog_annotated.png"),
                            annotated=True)
    plot_slope_fit(data, os.path.join(figures_dir, "slope_fit.png"))

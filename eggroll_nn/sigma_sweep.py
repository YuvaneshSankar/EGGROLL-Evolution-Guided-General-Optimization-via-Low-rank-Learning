"""Sweep sigma to find a regime where bias signal is comfortably above MC noise."""
import sys, time, torch, numpy as np
sys.path.insert(0, '.')
from src.nn_fitness import Fitness, load_mnist_batch
from src.perturbation import sample_eggroll, sample_gaussian
from src.grad_estimator import estimate_gradient, stack_grads, mean_grad, sem_grad, norm_F, diff

device = torch.device("cuda")

for d_hidden in [32, 128]:
    X, Y = load_mnist_batch(256, device, "data")
    fitness = Fitness(X, Y, d_hidden=d_hidden, device=device)
    P = sum(v.numel() for v in fitness.params.values())
    g_truth = fitness.analytical_gradient()
    g_truth_norm = norm_F(g_truth)
    print(f"=== d_hidden={d_hidden}, P={P}, ||g_truth||={g_truth_norm:.3f} ===", flush=True)

    for sigma in [0.05, 0.15, 0.3, 0.5]:
        N_ref = 200000
        N_eg = 50000
        T = 10
        g_ref = estimate_gradient(fitness, sigma, N_ref, 500,
            perturb_sampler=lambda B: sample_gaussian(fitness.param_shapes, B, device),
            device=device, pbar=False)
        for r in [1, 4, 16]:
            trials = [estimate_gradient(fitness, sigma, N_eg, 500,
                perturb_sampler=lambda B: sample_eggroll(fitness.param_shapes, B, r, device),
                device=device, pbar=False) for _ in range(T)]
            ts = stack_grads(trials)
            g_mean = mean_grad(ts)
            sem = sem_grad(ts)
            err = norm_F(diff(g_mean, g_ref))
            bias_sq = max(err**2 - sem**2, 0)
            bias = bias_sq**0.5
            print(f"  σ={sigma:.2f}  r={r:3d}  err={err:.3e}  sem_eg={sem:.3e}  est_bias={bias:.3e}  bias/sem={bias/max(sem,1e-9):.2f}", flush=True)

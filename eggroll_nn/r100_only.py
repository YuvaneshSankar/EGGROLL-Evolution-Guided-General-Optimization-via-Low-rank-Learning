"""Run JUST r=100 in a totally fresh process to dodge the cumulative-state hang.
Saves into focused_partial.json. Final analysis happens in a separate step."""
import json, os, sys, time
import torch
sys.path.insert(0, "/home/yuvanesh-sankar/eggroll/eggroll_nn")
from src.grad_estimator import (
    diff, estimate_gradient, mean_grad, norm_F, sem_grad, stack_grads,
)
from src.nn_fitness import Fitness, load_mnist_batch
from src.perturbation import sample_eggroll, sample_gaussian

device = torch.device("cuda")
torch.manual_seed(0)

partial_path = "/home/yuvanesh-sankar/eggroll/eggroll_nn/results/data/focused_partial.json"
with open(partial_path) as f:
    partial = json.load(f)
print(f"r_done so far: {partial['r_done']}", flush=True)

print("Rebuilding fitness + g_ref ...", flush=True)
X, Y = load_mnist_batch(256, device, "/home/yuvanesh-sankar/eggroll/eggroll_nn/data")
fitness = Fitness(X, Y, d_hidden=32, device=device, seed=0)
t0 = time.time()
g_ref_chunks = [estimate_gradient(fitness, 0.3, 200000, 500,
    perturb_sampler=lambda B: sample_gaussian(fitness.param_shapes, B, device),
    device=device, pbar=False) for _ in range(10)]
g_ref = mean_grad(stack_grads(g_ref_chunks))
sem_ref = sem_grad(stack_grads(g_ref_chunks))
print(f"g_ref in {time.time()-t0:.0f}s, SEM_ref={sem_ref:.4e}", flush=True)

# Sweep r=100
tr = time.time()
trials = []
for t in range(50):
    g_t = estimate_gradient(fitness, 0.3, 100000, 500,
        perturb_sampler=lambda B: sample_eggroll(fitness.param_shapes, B, 100, device),
        device=device, pbar=False)
    trials.append(g_t)
    if (t+1) % 10 == 0:
        print(f"  trial {t+1}/50  [{time.time()-tr:.0f}s]", flush=True)
ts = stack_grads(trials)
g_mean = mean_grad(ts)
sem_eg = sem_grad(ts)
err_ref = norm_F(diff(g_mean, g_ref))
g_truth = fitness.analytical_gradient()
err_truth = norm_F(diff(g_mean, g_truth))
print(f"r=100  err_to_ref={err_ref:.4e}  err_to_truth={err_truth:.4e}  SEM_eg={sem_eg:.4e}  [{time.time()-tr:.0f}s]", flush=True)

partial["r_done"].append(100)
partial["err_to_ref"].append(err_ref)
partial["err_to_truth"].append(err_truth)
partial["sem_eg"].append(sem_eg)
with open(partial_path, "w") as f:
    json.dump(partial, f, indent=2)
print("Saved r=100 to partial", flush=True)

"""Reference (ground-truth) ES gradient using antithetic full-rank Gaussian perturbations.

    g = (1 / (2σ N)) Σ_i [ f(μ + σE_i) - f(μ - σE_i) ] · E_i,   E_i ~ N(0, I).

Antithetic eliminates the f(μ)·E baseline-noise term, which otherwise dominates
variance by orders of magnitude.
"""
import torch
from tqdm import tqdm


def compute_reference_gradient(mu, fitness, sigma, N_ref, batch_size, device, desc="reference"):
    d1, d2 = mu.shape
    total = torch.zeros_like(mu)

    pbar = tqdm(total=N_ref, desc=desc, leave=False)
    cur_bs = batch_size
    i = 0
    while i < N_ref:
        bs = min(cur_bs, N_ref - i)
        try:
            E = torch.randn(bs, d1, d2, device=device)
            W_plus = mu.unsqueeze(0) + sigma * E
            W_minus = mu.unsqueeze(0) - sigma * E
            f_plus = fitness(W_plus)
            f_minus = fitness(W_minus)
            diff = (f_plus - f_minus).view(-1, 1, 1)
            total = total + (diff * E).sum(dim=0)
            i += bs
            pbar.update(bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cur_bs = max(1, cur_bs // 2)
            print(f"\n[warn] CUDA OOM in reference gradient — reducing batch size to {cur_bs}")
            if cur_bs == 1 and bs == 1:
                raise
    pbar.close()

    return total / (2.0 * sigma * N_ref)

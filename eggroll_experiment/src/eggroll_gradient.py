"""EGGROLL gradient with low-rank perturbations E = (1/√r) A Bᵀ and antithetic sampling.

    g = (1 / (2σ N)) Σ_i [ f(μ + σE_i) - f(μ - σE_i) ] · E_i.
"""
import torch
from tqdm import tqdm


def compute_eggroll_gradient(mu, fitness, sigma, rank, N_eggroll, batch_size, device, desc=None):
    d1, d2 = mu.shape
    r = rank
    scale = 1.0 / (r ** 0.5)
    total = torch.zeros_like(mu)

    pbar = tqdm(total=N_eggroll, desc=desc or f"eggroll r={r}", leave=False)
    cur_bs = batch_size
    i = 0
    while i < N_eggroll:
        bs = min(cur_bs, N_eggroll - i)
        try:
            A = torch.randn(bs, d1, r, device=device)
            B = torch.randn(bs, d2, r, device=device)
            E = scale * torch.matmul(A, B.transpose(-1, -2))
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
            print(f"\n[warn] CUDA OOM in eggroll r={r} — reducing batch size to {cur_bs}")
            if cur_bs == 1 and bs == 1:
                raise
    pbar.close()

    return total / (2.0 * sigma * N_eggroll)

"""MLP teacher–student fitness.

Single hidden weight matrix W ∈ ℝ^{d×d}. Fitness measures how well a one-layer
tanh net `tanh(X @ W)` matches a frozen teacher `tanh(X @ W_star)` on a fixed
batch of inputs.

This is the smallest setup with non-trivial higher-order derivatives — enough
for EGGROLL's O(σ²/r) bias to actually show up empirically.
"""
import torch


class MLPFitness:
    def __init__(self, d=128, n_data=32, device="cuda", seed=0):
        g = torch.Generator(device=device).manual_seed(seed)
        self.d = d
        self.n_data = n_data
        self.device = device
        # Inputs scaled so X @ W_star has O(1) entries (tanh isn't saturated).
        self.X = torch.randn(n_data, d, generator=g, device=device) / (d ** 0.5)
        self.W_star = torch.randn(d, d, generator=g, device=device)
        self.Y_target = torch.tanh(self.X @ self.W_star)

    def __call__(self, W):
        """f(W) = -mean ||tanh(X @ W) - Y_target||².

        Accepts W of shape (d, d) or (B, d, d). Returns scalar or (B,).
        """
        squeezed = False
        if W.dim() == 2:
            W = W.unsqueeze(0)
            squeezed = True
        # X @ W with broadcasting: (n_data, d) @ (B, d, d) -> (B, n_data, d)
        Y_pred = torch.tanh(torch.matmul(self.X, W))
        diff = Y_pred - self.Y_target
        f = -(diff * diff).mean(dim=(-1, -2))
        if squeezed:
            f = f.squeeze(0)
        return f

    def analytical_gradient(self, mu):
        """Autograd-computed ∇f(μ). Used as ground-truth sanity check."""
        mu_var = mu.detach().clone().requires_grad_(True)
        loss = self(mu_var)
        (g,) = torch.autograd.grad(loss, mu_var)
        return g.detach()

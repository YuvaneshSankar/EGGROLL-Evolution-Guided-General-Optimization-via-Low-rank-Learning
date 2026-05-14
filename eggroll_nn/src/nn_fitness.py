"""MNIST MLP fitness for ES.

Network: 784 → H → 10  (1 hidden layer, ReLU). Weights W ∈ ℝ^{out×in}, biases b ∈ ℝ^{out}.

Fitness f(θ) = -cross_entropy(net(X; θ), Y)  averaged over a fixed batch of MNIST examples.

The forward pass supports a leading "perturbation batch" dimension B, so we can
evaluate B perturbed networks in parallel on the GPU.
"""
import os

import torch
import torch.nn.functional as F


def load_mnist_batch(n_data, device, data_dir):
    """Load a fixed batch of `n_data` MNIST training examples, flattened.

    Falls back to deterministic synthetic data if torchvision can't fetch MNIST.
    Returns (X, Y) where X ∈ [0,1] flattened to (n_data, 784), Y ∈ {0..9}.
    """
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            root=data_dir, train=True, download=True,
            transform=transforms.ToTensor(),
        )
        # Take first n_data examples deterministically.
        Xs, Ys = [], []
        for i in range(n_data):
            x, y = ds[i]
            Xs.append(x)
            Ys.append(y)
        X = torch.stack(Xs).view(n_data, -1).to(device)
        Y = torch.tensor(Ys, dtype=torch.long, device=device)
        return X, Y
    except Exception as e:
        print(f"[warn] could not load MNIST ({e}); falling back to synthetic data")
        g = torch.Generator(device="cpu").manual_seed(0)
        X = torch.rand(n_data, 784, generator=g).to(device)
        Y = torch.randint(0, 10, (n_data,), generator=g).to(device)
        return X, Y


def init_params(d_in=784, d_hidden=128, d_out=10, device="cuda", seed=0):
    """Initialize MLP params (Kaiming for W1, scaled for W2)."""
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = torch.randn(d_hidden, d_in, generator=g, device=device) * (2.0 / d_in) ** 0.5
    b1 = torch.zeros(d_hidden, device=device)
    W2 = torch.randn(d_out, d_hidden, generator=g, device=device) * (1.0 / d_hidden) ** 0.5
    b2 = torch.zeros(d_out, device=device)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, X):
    """Single network forward pass. params: dict of tensors. X: (n_data, 784)."""
    h = F.relu(X @ params["W1"].T + params["b1"])
    logits = h @ params["W2"].T + params["b2"]
    return logits


def forward_batched(params_batched, X):
    """Batched forward pass. params_batched: dict with leading B dim.
    Returns logits of shape (B, n_data, 10).
    """
    W1, b1, W2, b2 = (params_batched[k] for k in ("W1", "b1", "W2", "b2"))
    # X (n, d_in)  @  W1.T (B, d_in, d_hidden)  ->  (B, n, d_hidden)
    h = torch.matmul(X, W1.transpose(-1, -2)) + b1.unsqueeze(-2)
    h = F.relu(h)
    logits = torch.matmul(h, W2.transpose(-1, -2)) + b2.unsqueeze(-2)
    return logits


class Fitness:
    def __init__(self, X, Y, d_hidden=128, device="cuda", seed=0):
        self.X = X
        self.Y = Y
        self.device = device
        self.d_hidden = d_hidden
        self.params = init_params(d_hidden=d_hidden, device=device, seed=seed)
        self.param_shapes = {k: v.shape for k, v in self.params.items()}

    def __call__(self, params):
        """f(params) = -mean CE loss. Returns scalar."""
        logits = forward(params, self.X)
        loss = F.cross_entropy(logits, self.Y, reduction="mean")
        return -loss

    def call_batched(self, params_batched):
        """f for B perturbed param sets in parallel. Returns shape (B,)."""
        logits = forward_batched(params_batched, self.X)  # (B, n, 10)
        B, n, _ = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * n, -1), self.Y.repeat(B), reduction="none"
        ).view(B, n).mean(dim=-1)
        return -loss

    def analytical_gradient(self):
        """Autograd ∇f at the current params. Returns dict of grads."""
        params_var = {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}
        logits = forward(params_var, self.X)
        loss = F.cross_entropy(logits, self.Y, reduction="mean")
        f = -loss
        grads = torch.autograd.grad(f, list(params_var.values()))
        return {k: g.detach() for k, g in zip(params_var.keys(), grads)}

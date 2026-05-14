"""Per-parameter perturbations for ES.

- `sample_gaussian(shapes, B, device)`  – full-rank N(0,I) per param.
- `sample_eggroll(shapes, B, r, device)` – low-rank E = (1/√r) A Bᵀ for matrix
  params, full-rank Gaussian for vector params (biases). Matches the EGGROLL
  paper's per-layer formulation.

Returns dicts {param_name: tensor of shape (B, *param_shape)}.
"""
import torch


def sample_gaussian(shapes, B, device, generator=None):
    out = {}
    for name, shape in shapes.items():
        out[name] = torch.randn(B, *shape, device=device, generator=generator)
    return out


def sample_eggroll(shapes, B, r, device, generator=None):
    """Low-rank EGGROLL perturbation per matrix param. Biases stay full Gaussian."""
    scale = 1.0 / (r ** 0.5)
    out = {}
    for name, shape in shapes.items():
        if len(shape) == 2:
            d_out, d_in = shape
            A = torch.randn(B, d_out, r, device=device, generator=generator)
            Bm = torch.randn(B, d_in, r, device=device, generator=generator)
            out[name] = scale * torch.matmul(A, Bm.transpose(-1, -2))  # (B, d_out, d_in)
        else:
            # Vector params (biases): just full Gaussian.
            out[name] = torch.randn(B, *shape, device=device, generator=generator)
    return out


def add_perturbation(params, perturb, sigma, sign=1.0):
    """Return params + sign·σ·perturb, broadcasting params over the B dim."""
    out = {}
    for k, p in params.items():
        out[k] = p.unsqueeze(0) + (sign * sigma) * perturb[k]
    return out

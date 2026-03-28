from __future__ import annotations

import torch


def _canonical_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _make_generator(seed: int | None, device: torch.device | str | None = None) -> torch.Generator:
    canonical_device = _canonical_device(device)
    generator = torch.Generator(device=canonical_device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def make_random_orthogonal_matrix(
    dim: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    canonical_device = _canonical_device(device)
    generator = _make_generator(seed, device=canonical_device)
    matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float32, device=canonical_device)
    q, r = torch.linalg.qr(matrix, mode="reduced")

    # Fix QR sign ambiguity so the result is deterministic under a fixed seed.
    diag = torch.diagonal(r)
    signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    return q * signs.unsqueeze(0)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norms

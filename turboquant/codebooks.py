from __future__ import annotations

import math
from functools import lru_cache

import torch


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(xp, x, right=False)
    idx = torch.clamp(idx, min=1, max=xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    denom = torch.clamp(x1 - x0, min=torch.finfo(xp.dtype).eps)
    weight = (x - x0) / denom
    return y0 + weight * (y1 - y0)


def turbo_coordinate_pdf(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 2:
        raise ValueError("dim must be >= 2")

    log_coeff = math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2.0)
    exponent = (dim - 3) / 2.0
    base = torch.clamp(1.0 - x.square(), min=0.0)
    return torch.exp(torch.tensor(log_coeff, dtype=x.dtype, device=x.device)) * base.pow(exponent)


def _weighted_lloyd_max(
    grid: torch.Tensor,
    pdf: torch.Tensor,
    k: int,
    steps: int,
) -> torch.Tensor:
    pdf = pdf / torch.trapezoid(pdf, grid)
    cdf = torch.cumsum(pdf, dim=0)
    cdf = cdf / cdf[-1]

    quantiles = (torch.arange(k, dtype=grid.dtype, device=grid.device) + 0.5) / k
    centroids = _interp1d(quantiles, cdf, grid)

    for _ in range(steps):
        boundaries = torch.empty(k + 1, dtype=grid.dtype, device=grid.device)
        boundaries[0] = grid[0]
        boundaries[-1] = grid[-1]
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        new_centroids = centroids.clone()
        for idx in range(k):
            left = boundaries[idx]
            right = boundaries[idx + 1]
            if idx == k - 1:
                mask = (grid >= left) & (grid <= right)
            else:
                mask = (grid >= left) & (grid < right)

            if not torch.any(mask):
                continue

            grid_slice = grid[mask]
            pdf_slice = pdf[mask]
            mass = torch.trapezoid(pdf_slice, grid_slice)
            if mass > 0:
                numerator = torch.trapezoid(grid_slice * pdf_slice, grid_slice)
                new_centroids[idx] = numerator / mass

        if torch.allclose(new_centroids, centroids, atol=1e-8, rtol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.to(dtype=torch.float32)


@lru_cache(maxsize=None)
def solve_beta_codebook(dim: int, bits: int, steps: int = 64, n_grid: int = 32769) -> torch.Tensor:
    if bits < 1:
        raise ValueError("bits must be >= 1")

    grid = torch.linspace(-1.0, 1.0, n_grid, dtype=torch.float64)
    pdf = turbo_coordinate_pdf(grid, dim)
    centroids = _weighted_lloyd_max(grid=grid, pdf=pdf, k=2**bits, steps=steps)
    return torch.sort(centroids).values


def centroid_boundaries(codebook: torch.Tensor) -> torch.Tensor:
    """Compute decision boundaries as midpoints between sorted centroids.

    Returns a 1D tensor of length ``len(codebook) - 1`` suitable for use
    with :func:`torch.searchsorted`.
    """
    return 0.5 * (codebook[:-1] + codebook[1:])


def searchsorted_quantize(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Quantize each element of *values* to the nearest centroid index.

    Uses :func:`torch.searchsorted` on pre-computed *boundaries* for
    ``O(d log k)`` lookup instead of the default ``O(d * k)`` broadcast.
    """
    return torch.searchsorted(boundaries, values)

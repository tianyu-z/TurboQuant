from __future__ import annotations

import math

import torch

from turboquant.math import _canonical_device, _make_generator


class QJLQuantizer:
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.dim = dim
        self.device = _canonical_device(device)
        generator = _make_generator(seed, device=self.device)
        self.projection = torch.randn(dim, dim, generator=generator, dtype=torch.float32, device=self.device)
        self.scale = math.sqrt(math.pi / 2.0) / float(dim)

    def to(self, device: torch.device | str) -> "QJLQuantizer":
        self.device = _canonical_device(device)
        self.projection = self.projection.to(self.device)
        return self

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.projection.device:
            self.to(x.device)
        projected = x @ self.projection.T
        return torch.where(projected >= 0, torch.ones_like(projected), -torch.ones_like(projected))

    def dequantize(self, z: torch.Tensor) -> torch.Tensor:
        if z.device != self.projection.device:
            self.to(z.device)
        return self.scale * (z.to(dtype=self.projection.dtype) @ self.projection)

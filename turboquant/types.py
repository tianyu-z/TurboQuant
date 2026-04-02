from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass
class PackedCodes:
    data: torch.Tensor
    n_rows: int
    dim: int
    bits: int

    def __post_init__(self) -> None:
        if self.data.dtype != torch.uint8:
            raise ValueError("PackedCodes data must use torch.uint8 storage")
        if self.data.ndim != 1 or not self.data.is_contiguous():
            raise ValueError("PackedCodes data must be a flat contiguous byte buffer")
        if self.n_rows < 0:
            raise ValueError("n_rows must be non-negative")
        if self.dim < 0:
            raise ValueError("dim must be non-negative")
        if not 1 <= self.bits <= 8:
            raise ValueError("bits must be in [1, 8]")
        expected_bytes = math.ceil(self.n_rows * self.dim * self.bits / 8) if self.n_rows * self.dim else 0
        if self.data.numel() != expected_bytes:
            raise ValueError("PackedCodes byte buffer size does not match the logical shape")

    def num_bytes(self) -> int:
        return int(self.data.numel() * self.data.element_size())

    def bytes_per_vector(self) -> float:
        if self.n_rows == 0:
            return 0.0
        return self.num_bytes() / float(self.n_rows)


@dataclass
class TurboQuantMSEPayload:
    codes: PackedCodes | None = None
    indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.codes is None) == (self.indices is None):
            raise ValueError("TurboQuantMSEPayload expects exactly one of codes or indices")

    def num_bytes(self) -> int:
        if self.codes is not None:
            return self.codes.num_bytes()
        assert self.indices is not None
        return int(self.indices.numel() * self.indices.element_size())

    def bytes_per_vector(self) -> float:
        if self.codes is not None:
            return self.codes.bytes_per_vector()
        assert self.indices is not None
        if self.indices.shape[0] == 0:
            return 0.0
        return self.num_bytes() / float(self.indices.shape[0])


@dataclass
class TurboQuantProdPayload:
    mse_codes: PackedCodes | None = None
    residual_signs: PackedCodes | torch.Tensor | None = None
    residual_norm: torch.Tensor | None = None
    mse_payload: TurboQuantMSEPayload | None = None

    def __post_init__(self) -> None:
        if self.residual_signs is None or self.residual_norm is None:
            raise ValueError("TurboQuantProdPayload requires residual signs and residual norm")
        if (self.mse_codes is None) == (self.mse_payload is None):
            raise ValueError("TurboQuantProdPayload expects exactly one of mse_codes or mse_payload")

        n_rows, dim = self._logical_shape()
        if self.residual_norm.ndim != 2 or self.residual_norm.shape != (n_rows, 1):
            raise ValueError("TurboQuantProd residual norm must be a column vector with one row per payload entry")

        if isinstance(self.residual_signs, PackedCodes):
            if self.residual_signs.bits != 1:
                raise ValueError("TurboQuantProd residual signs must be packed as 1-bit codes")
            if self.residual_signs.n_rows != n_rows or self.residual_signs.dim != dim:
                raise ValueError("TurboQuantProd residual signs must align with the payload logical shape")
        else:
            if self.residual_signs.ndim != 2 or self.residual_signs.shape != (n_rows, dim):
                raise ValueError("TurboQuantProd residual signs tensor must match the payload logical shape")
            if not torch.all((self.residual_signs == -1) | (self.residual_signs == 1)):
                raise ValueError("TurboQuantProd legacy residual signs tensor must be binary {-1, 1}")

    def _logical_shape(self) -> tuple[int, int]:
        if self.mse_codes is not None:
            return self.mse_codes.n_rows, self.mse_codes.dim
        assert self.mse_payload is not None
        if self.mse_payload.codes is not None:
            return self.mse_payload.codes.n_rows, self.mse_payload.codes.dim
        assert self.mse_payload.indices is not None
        if self.mse_payload.indices.ndim != 2:
            raise ValueError("legacy mse_payload indices must be a 2D tensor")
        return int(self.mse_payload.indices.shape[0]), int(self.mse_payload.indices.shape[1])

    def num_bytes(self) -> int:
        mse_bytes = self.mse_codes.num_bytes() if self.mse_codes is not None else self.mse_payload.num_bytes()
        if isinstance(self.residual_signs, PackedCodes):
            residual_sign_bytes = self.residual_signs.num_bytes()
        else:
            residual_sign_bytes = int(self.residual_signs.numel() * self.residual_signs.element_size())
        assert self.residual_norm is not None
        return (
            mse_bytes
            + residual_sign_bytes
            + int(self.residual_norm.numel() * self.residual_norm.element_size())
        )

    def bytes_per_vector(self) -> float:
        n_rows, _ = self._logical_shape()
        if n_rows == 0:
            return 0.0
        return self.num_bytes() / float(n_rows)


@dataclass
class QuantizerState:
    kind: str
    dim: int
    bits: int
    codebook: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    scale: float | None = None
    norm_correction: bool = False
    fast_lookup: bool = False
    format_version: int = 1


@dataclass
class IndexArtifact:
    state: QuantizerState
    payload: TurboQuantMSEPayload | TurboQuantProdPayload
    normalization: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)
    quantizer: object | None = None
    format_version: int = 1


@dataclass
class SearchResult:
    indices: torch.Tensor
    scores: torch.Tensor

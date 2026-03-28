from __future__ import annotations

import torch

from turboquant.codebooks import solve_beta_codebook
from turboquant.math import _canonical_device, make_random_orthogonal_matrix
from turboquant.packing import pack_codes, unpack_codes
from turboquant.types import QuantizerState, TurboQuantMSEPayload


class TurboQuantMSE:
    _SCORE_CHUNK_ROWS = 1024

    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.device = _canonical_device(device)
        self.rotation = make_random_orthogonal_matrix(dim=dim, seed=seed, device=self.device)
        self.codebook = solve_beta_codebook(dim=dim, bits=bits).to(self.device)

    def to(self, device: torch.device | str) -> "TurboQuantMSE":
        self.device = _canonical_device(device)
        self.rotation = self.rotation.to(self.device)
        self.codebook = self.codebook.to(self.device)
        return self

    def prepare(self, x: torch.Tensor | None = None) -> "TurboQuantMSE":
        """Data-oblivious setup. Validates shapes and moves tensors to the correct device.
        Does NOT learn codebooks, projections, or rotations from x."""
        if x is not None and x.shape[-1] != self.dim:
            raise ValueError(f"expected input dim {self.dim}, got {x.shape[-1]}")
        if x is not None and x.device != self.device:
            self.to(x.device)
        return self

    def fit(self, x: torch.Tensor | None = None) -> "TurboQuantMSE":
        """Non-learning alias for prepare(). Retained for backward compatibility."""
        return self.prepare(x)

    def quantize(self, x: torch.Tensor) -> TurboQuantMSEPayload:
        if x.device != self.device:
            self.to(x.device)
        y = x @ self.rotation.T
        distances = torch.abs(y.unsqueeze(-1) - self.codebook.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1)
        return TurboQuantMSEPayload(codes=pack_codes(indices, bits=self.bits))

    def _decode_indices(self, payload: TurboQuantMSEPayload) -> torch.Tensor:
        if payload.codes is not None:
            if payload.codes.dim != self.dim or payload.codes.bits != self.bits:
                raise ValueError(f"expected packed codes for dim={self.dim}, bits={self.bits}")
            packed_data = payload.codes.data
            if packed_data.device != self.device:
                packed_data = packed_data.to(self.device)
            indices = unpack_codes(
                packed_data,
                n_rows=payload.codes.n_rows,
                dim=payload.codes.dim,
                bits=payload.codes.bits,
            )
            return indices

        assert payload.indices is not None
        if payload.indices.ndim != 2 or payload.indices.shape[1] != self.dim:
            raise ValueError(f"expected legacy indices as a 2D tensor with dim={self.dim}")
        if payload.indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("expected legacy indices to use an integer dtype")
        indices = payload.indices.to(device=self.device, dtype=torch.int64)
        if torch.any(indices < 0) or torch.any(indices >= self.codebook.numel()):
            raise ValueError("legacy indices must lie within the codebook range")
        return indices

    def _decode_packed_code_rows(
        self,
        payload: TurboQuantMSEPayload,
        *,
        row_start: int,
        row_count: int,
    ) -> torch.Tensor:
        assert payload.codes is not None
        packed = payload.codes
        if packed.dim != self.dim or packed.bits != self.bits:
            raise ValueError(f"expected packed codes for dim={self.dim}, bits={self.bits}")
        if row_start < 0 or row_count < 0 or row_start + row_count > packed.n_rows:
            raise ValueError("requested packed-code rows are out of bounds")

        packed_data = packed.data
        if packed_data.device != self.device:
            packed_data = packed_data.to(self.device)

        total_values = row_count * packed.dim
        if total_values == 0:
            return torch.empty((row_count, packed.dim), dtype=torch.int64, device=self.device)

        start_bit = row_start * packed.dim * packed.bits
        total_bits = total_values * packed.bits
        bit_positions = start_bit + torch.arange(total_bits, dtype=torch.int64, device=self.device)
        byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
        bit_indices = bit_positions % 8
        bit_values = ((packed_data[byte_indices].to(torch.int64) >> bit_indices) & 1).reshape(total_values, packed.bits)
        bit_weights = (1 << torch.arange(packed.bits, dtype=torch.int64, device=self.device)).view(1, packed.bits)
        return (bit_values * bit_weights).sum(dim=-1).reshape(row_count, packed.dim)

    def dequantize(self, payload: TurboQuantMSEPayload) -> torch.Tensor:
        indices = self._decode_indices(payload)
        y_hat = self.codebook[indices]
        return y_hat @ self.rotation

    def score(self, query: torch.Tensor, payload: TurboQuantMSEPayload) -> torch.Tensor:
        if query.device != self.device:
            self.to(query.device)
        q_rot = query @ self.rotation.T
        legacy_indices = None
        if payload.codes is not None:
            n_rows = payload.codes.n_rows
        else:
            assert payload.indices is not None
            n_rows = payload.indices.shape[0]
            legacy_indices = self._decode_indices(payload)
        scores = torch.empty((query.shape[0], n_rows), dtype=q_rot.dtype, device=self.device)
        for start in range(0, n_rows, self._SCORE_CHUNK_ROWS):
            end = min(start + self._SCORE_CHUNK_ROWS, n_rows)
            if payload.codes is not None:
                chunk_indices = self._decode_packed_code_rows(payload, row_start=start, row_count=end - start)
            else:
                assert legacy_indices is not None
                chunk_indices = legacy_indices[start:end]
            chunk_y_hat = self.codebook[chunk_indices]
            scores[:, start:end] = q_rot @ chunk_y_hat.T
        return scores

    def export_state(self) -> QuantizerState:
        return QuantizerState(
            kind="turboquant_mse",
            dim=self.dim,
            bits=self.bits,
            codebook=self.codebook.detach().clone(),
            rotation=self.rotation.detach().clone(),
            format_version=1,
        )

    @classmethod
    def from_state(cls, state: QuantizerState) -> "TurboQuantMSE":
        if state.kind != "turboquant_mse":
            raise ValueError(f"expected turboquant_mse state, got {state.kind}")
        if state.codebook is None or state.rotation is None:
            raise ValueError("TurboQuantMSE state must include codebook and rotation tensors")
        quantizer = cls(dim=state.dim, bits=state.bits, device=state.codebook.device)
        quantizer.codebook = state.codebook.detach().clone()
        quantizer.rotation = state.rotation.detach().clone()
        quantizer.device = quantizer.codebook.device
        return quantizer

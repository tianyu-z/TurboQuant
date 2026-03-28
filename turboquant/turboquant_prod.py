from __future__ import annotations

import torch

from turboquant.packing import pack_signs, unpack_signs
from turboquant.qjl import QJLQuantizer
from turboquant.turboquant_mse import TurboQuantMSE, TurboQuantMSEPayload
from turboquant.types import QuantizerState, TurboQuantProdPayload


class TurboQuantProd:
    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if bits < 2:
            raise ValueError("bits must be >= 2")

        self.dim = dim
        self.bits = bits
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mse_quantizer = TurboQuantMSE(dim=dim, bits=bits - 1, seed=seed, device=self.device)
        self.qjl_quantizer = QJLQuantizer(dim=dim, seed=seed, device=self.device)

    def to(self, device: torch.device | str) -> "TurboQuantProd":
        self.device = torch.device(device)
        self.mse_quantizer.to(self.device)
        self.qjl_quantizer.to(self.device)
        return self

    def prepare(self, x: torch.Tensor | None = None) -> "TurboQuantProd":
        """Data-oblivious setup. Validates shapes and moves tensors to the correct device.
        Does NOT learn codebooks, projections, or rotations from x."""
        if x is not None and x.shape[-1] != self.dim:
            raise ValueError(f"expected input dim {self.dim}, got {x.shape[-1]}")
        if x is not None and x.device != self.device:
            self.to(x.device)
        self.mse_quantizer.prepare(x)
        return self

    def fit(self, x: torch.Tensor | None = None) -> "TurboQuantProd":
        """Non-learning alias for prepare(). Retained for backward compatibility."""
        return self.prepare(x)

    def quantize(self, x: torch.Tensor) -> TurboQuantProdPayload:
        if x.device != self.device:
            self.to(x.device)
        mse_payload = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(mse_payload)
        residual = x - x_hat_mse
        residual_signs = pack_signs(self.qjl_quantizer.quantize(residual))
        residual_norm = torch.linalg.norm(residual, dim=-1, keepdim=True)
        return TurboQuantProdPayload(
            mse_codes=mse_payload.codes,
            residual_signs=residual_signs,
            residual_norm=residual_norm,
        )

    def _mse_payload(self, payload: TurboQuantProdPayload) -> TurboQuantMSEPayload:
        if payload.mse_codes is not None:
            if payload.mse_codes.dim != self.dim or payload.mse_codes.bits != self.bits - 1:
                raise ValueError(f"expected packed MSE codes for dim={self.dim}, bits={self.bits - 1}")
            return TurboQuantMSEPayload(codes=payload.mse_codes)

        assert payload.mse_payload is not None
        return payload.mse_payload

    def _residual_signs(self, payload: TurboQuantProdPayload) -> torch.Tensor:
        if isinstance(payload.residual_signs, torch.Tensor):
            return payload.residual_signs.to(device=self.device)

        if payload.residual_signs.dim != self.dim or payload.residual_signs.bits != 1:
            raise ValueError(f"expected packed residual signs for dim={self.dim}, bits=1")
        packed_data = payload.residual_signs.data
        if packed_data.device != self.device:
            packed_data = packed_data.to(self.device)
        return unpack_signs(
            packed_data,
            n_rows=payload.residual_signs.n_rows,
            dim=payload.residual_signs.dim,
        )

    def dequantize(self, payload: TurboQuantProdPayload) -> torch.Tensor:
        x_hat_mse = self.mse_quantizer.dequantize(self._mse_payload(payload))
        residual_signs = self._residual_signs(payload)
        assert payload.residual_norm is not None
        residual_norm = payload.residual_norm.to(device=self.device)
        x_hat_qjl = self.qjl_quantizer.dequantize(residual_signs)
        return x_hat_mse + residual_norm * x_hat_qjl

    def score(self, query: torch.Tensor, payload: TurboQuantProdPayload) -> torch.Tensor:
        if query.device != self.device:
            self.to(query.device)
        mse_scores = self.mse_quantizer.score(query, self._mse_payload(payload))
        residual_signs = self._residual_signs(payload).to(dtype=query.dtype)
        projected_query = query @ self.qjl_quantizer.projection.T
        qjl_scores = self.qjl_quantizer.scale * (projected_query @ residual_signs.T)
        assert payload.residual_norm is not None
        residual_norm = payload.residual_norm.to(device=self.device, dtype=query.dtype).T
        return mse_scores + qjl_scores * residual_norm

    def export_state(self) -> QuantizerState:
        return QuantizerState(
            kind="turboquant_prod",
            dim=self.dim,
            bits=self.bits,
            codebook=self.mse_quantizer.codebook.detach().clone(),
            rotation=self.mse_quantizer.rotation.detach().clone(),
            projection=self.qjl_quantizer.projection.detach().clone(),
            scale=float(self.qjl_quantizer.scale),
            format_version=1,
        )

    @classmethod
    def from_state(cls, state: QuantizerState) -> "TurboQuantProd":
        if state.kind != "turboquant_prod":
            raise ValueError(f"expected turboquant_prod state, got {state.kind}")
        if state.codebook is None or state.rotation is None or state.projection is None or state.scale is None:
            raise ValueError("TurboQuantProd state must include codebook, rotation, projection, and scale")
        quantizer = cls(dim=state.dim, bits=state.bits, device=state.codebook.device)
        quantizer.mse_quantizer.codebook = state.codebook.detach().clone()
        quantizer.mse_quantizer.rotation = state.rotation.detach().clone()
        quantizer.qjl_quantizer.projection = state.projection.detach().clone()
        quantizer.qjl_quantizer.scale = float(state.scale)
        quantizer.device = quantizer.mse_quantizer.codebook.device
        return quantizer

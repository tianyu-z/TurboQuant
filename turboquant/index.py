from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from turboquant.io import load_index_artifact, save_index_artifact
from turboquant.math import normalize_rows
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import SearchResult, TurboQuantMSEPayload, TurboQuantProdPayload


Quantizer = TurboQuantMSE | TurboQuantProd
Payload = TurboQuantMSEPayload | TurboQuantProdPayload


def _payload_n_rows(payload: Payload) -> int:
    if isinstance(payload, TurboQuantMSEPayload):
        if payload.codes is not None:
            return payload.codes.n_rows
        assert payload.indices is not None
        return int(payload.indices.shape[0])

    if payload.mse_codes is not None:
        return payload.mse_codes.n_rows
    assert payload.mse_payload is not None
    if payload.mse_payload.codes is not None:
        return payload.mse_payload.codes.n_rows
    assert payload.mse_payload.indices is not None
    return int(payload.mse_payload.indices.shape[0])


def _require_unit_normalization(normalization: str) -> None:
    if normalization != "unit":
        raise ValueError("normalization must be 'unit' in this plan")


@dataclass
class TurboQuantIndex:
    quantizer: Quantizer
    payload: Payload
    normalization: str = "unit"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def algorithm(self) -> str:
        if isinstance(self.quantizer, TurboQuantProd):
            return "prod"
        return "mse"

    @property
    def dim(self) -> int:
        return int(self.quantizer.dim)

    @property
    def bits(self) -> int:
        return int(self.quantizer.bits)

    @property
    def n_rows(self) -> int:
        return _payload_n_rows(self.payload)

    @classmethod
    def build(
        cls,
        data: torch.Tensor,
        algorithm: str = "prod",
        bits: int = 3,
        seed: int = 0,
        device: str = "cpu",
        normalization: str = "unit",
    ) -> "TurboQuantIndex":
        if data.ndim != 2:
            raise ValueError("data must be a 2D tensor")
        _require_unit_normalization(normalization)

        normalized_data = normalize_rows(data).to(device)
        dim = int(normalized_data.shape[1])

        if algorithm == "prod":
            quantizer: Quantizer = TurboQuantProd(dim=dim, bits=bits, seed=seed, device=device)
        elif algorithm == "mse":
            quantizer = TurboQuantMSE(dim=dim, bits=bits, seed=seed, device=device)
        else:
            raise ValueError("algorithm must be one of {'mse', 'prod'}")

        quantizer.prepare(normalized_data)
        payload = quantizer.quantize(normalized_data)
        metadata = {
            "algorithm": algorithm,
            "bits": bits,
            "dim": dim,
            "n_rows": int(normalized_data.shape[0]),
            "normalization": normalization,
            "seed": seed,
        }
        return cls(
            quantizer=quantizer,
            payload=payload,
            normalization=normalization,
            metadata=metadata,
        )

    def search(self, query: torch.Tensor, k: int) -> SearchResult:
        if query.ndim != 2:
            raise ValueError("query must be a 2D tensor")
        if query.shape[1] != self.dim:
            raise ValueError(f"expected query dim {self.dim}, got {query.shape[1]}")
        if k <= 0:
            raise ValueError("k must be positive")
        _require_unit_normalization(self.normalization)

        query_device = torch.device(self.quantizer.device)
        normalized_query = normalize_rows(query.to(query_device))
        scores = self.quantizer.score(normalized_query, self.payload)
        topk = torch.topk(scores, k=min(k, self.n_rows), dim=-1)
        return SearchResult(indices=topk.indices, scores=topk.values)

    def save(self, path: str | Path) -> None:
        _require_unit_normalization(self.normalization)
        save_index_artifact(
            path,
            quantizer=self.quantizer,
            payload=self.payload,
            metadata={**self.metadata, "normalization": self.normalization},
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TurboQuantIndex":
        artifact = load_index_artifact(path)
        if artifact.quantizer is None:
            raise ValueError("saved artifact did not include a quantizer")
        _require_unit_normalization(artifact.normalization)
        artifact.quantizer.to(device)
        metadata = dict(artifact.metadata)
        metadata.setdefault("algorithm", artifact.state.kind.removeprefix("turboquant_"))
        metadata.setdefault("bits", artifact.state.bits)
        metadata.setdefault("dim", artifact.state.dim)
        metadata.setdefault("n_rows", _payload_n_rows(artifact.payload))
        metadata.setdefault("normalization", artifact.normalization)
        return cls(
            quantizer=artifact.quantizer,
            payload=artifact.payload,
            normalization=artifact.normalization,
            metadata=metadata,
        )

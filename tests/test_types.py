from __future__ import annotations

import pytest
import torch

from turboquant.math import normalize_rows
from turboquant.packing import unpack_codes, unpack_signs
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import (
    IndexArtifact,
    PackedCodes,
    QuantizerState,
    SearchResult,
    TurboQuantMSEPayload,
    TurboQuantProdPayload,
)


def test_packed_codes_report_actual_storage_bytes() -> None:
    packed = PackedCodes(data=torch.zeros(3, dtype=torch.uint8), n_rows=2, dim=6, bits=2)
    assert packed.num_bytes() == 3
    assert packed.bytes_per_vector() == pytest.approx(1.5)


def test_packed_codes_require_uint8_storage() -> None:
    with pytest.raises(ValueError, match="torch.uint8"):
        PackedCodes(data=torch.zeros(3, dtype=torch.int64), n_rows=2, dim=6, bits=2)


def test_packed_codes_require_contiguous_storage() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        PackedCodes(data=torch.arange(4, dtype=torch.uint8)[::2], n_rows=1, dim=16, bits=1)


def test_packed_codes_require_exact_byte_count() -> None:
    with pytest.raises(ValueError, match="byte"):
        PackedCodes(data=torch.zeros(1, dtype=torch.uint8), n_rows=1, dim=8, bits=8)


def test_payload_and_artifact_types_preserve_typed_fields() -> None:
    codes = PackedCodes(data=torch.tensor([0, 1, 2, 3], dtype=torch.uint8), n_rows=2, dim=8, bits=2)
    signs = PackedCodes(data=torch.tensor([7, 8], dtype=torch.uint8), n_rows=2, dim=8, bits=1)
    mse_payload = TurboQuantMSEPayload(codes=codes)
    prod_payload = TurboQuantProdPayload(
        mse_codes=codes,
        residual_signs=signs,
        residual_norm=torch.ones(2, 1),
    )
    state = QuantizerState(
        kind="turboquant_prod",
        dim=8,
        bits=3,
        codebook=torch.linspace(-1.0, 1.0, steps=4),
        rotation=torch.eye(8),
        projection=torch.eye(8),
        scale=0.25,
    )
    artifact = IndexArtifact(state=state, payload=prod_payload, normalization="unit_sphere")
    result = SearchResult(
        indices=torch.tensor([[1, 0]], dtype=torch.int64),
        scores=torch.tensor([[0.5, 0.25]], dtype=torch.float32),
    )

    assert mse_payload.codes.bits == 2
    assert prod_payload.residual_signs.data.dtype == torch.uint8
    assert prod_payload.mse_codes.bits == 2
    assert artifact.state.scale == pytest.approx(0.25)
    assert artifact.normalization == "unit_sphere"
    assert result.indices.dtype == torch.int64


def test_prod_payload_requires_column_vector_residual_norm() -> None:
    codes = PackedCodes(data=torch.tensor([0, 1, 2, 3], dtype=torch.uint8), n_rows=2, dim=8, bits=2)
    signs = PackedCodes(data=torch.tensor([7, 8], dtype=torch.uint8), n_rows=2, dim=8, bits=1)
    with pytest.raises(ValueError, match="residual norm"):
        TurboQuantProdPayload(mse_codes=codes, residual_signs=signs, residual_norm=torch.ones(2))


def test_runtime_quantizers_use_stable_payload_types() -> None:
    x = normalize_rows(torch.randn(8, 16))

    mse_payload = TurboQuantMSE(dim=16, bits=2, seed=0).quantize(x)
    prod_payload = TurboQuantProd(dim=16, bits=3, seed=0).quantize(x)

    assert isinstance(mse_payload, TurboQuantMSEPayload)
    assert mse_payload.codes.data.dtype == torch.uint8
    assert isinstance(prod_payload, TurboQuantProdPayload)
    assert prod_payload.mse_codes.data.dtype == torch.uint8
    assert prod_payload.residual_signs.data.dtype == torch.uint8


def test_legacy_payload_shapes_remain_supported_for_compatibility() -> None:
    x = normalize_rows(torch.randn(8, 16))

    mse_quantizer = TurboQuantMSE(dim=16, bits=2, seed=0)
    mse_payload = mse_quantizer.quantize(x)
    legacy_indices = unpack_codes(
        mse_payload.codes.data,
        n_rows=mse_payload.codes.n_rows,
        dim=mse_payload.codes.dim,
        bits=mse_payload.codes.bits,
    )
    legacy_mse_payload = TurboQuantMSEPayload(indices=legacy_indices)
    assert mse_quantizer.dequantize(legacy_mse_payload).shape == x.shape

    prod_quantizer = TurboQuantProd(dim=16, bits=3, seed=0)
    prod_payload = prod_quantizer.quantize(x)
    legacy_prod_payload = TurboQuantProdPayload(
        mse_payload=legacy_mse_payload,
        residual_signs=unpack_signs(
            prod_payload.residual_signs.data,
            n_rows=prod_payload.residual_signs.n_rows,
            dim=prod_payload.residual_signs.dim,
        ),
        residual_norm=prod_payload.residual_norm,
    )
    assert prod_quantizer.dequantize(legacy_prod_payload).shape == x.shape


def test_legacy_prod_payload_rejects_non_matrix_indices() -> None:
    with pytest.raises(ValueError, match="2D"):
        TurboQuantProdPayload(
            mse_payload=TurboQuantMSEPayload(indices=torch.zeros(16, dtype=torch.int64)),
            residual_signs=torch.ones(1, 16),
            residual_norm=torch.ones(1, 1),
        )


def test_legacy_prod_payload_rejects_non_binary_residual_signs() -> None:
    with pytest.raises(ValueError, match="binary"):
        TurboQuantProdPayload(
            mse_payload=TurboQuantMSEPayload(indices=torch.zeros((1, 16), dtype=torch.int64)),
            residual_signs=torch.zeros(1, 16),
            residual_norm=torch.ones(1, 1),
        )

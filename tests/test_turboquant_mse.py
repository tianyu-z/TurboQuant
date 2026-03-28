from __future__ import annotations

import pytest
import torch

from turboquant.math import normalize_rows
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.types import PackedCodes, TurboQuantMSEPayload


def test_turboquant_mse_roundtrip_preserves_shape() -> None:
    x = normalize_rows(torch.randn(32, 128))
    quantizer = TurboQuantMSE(dim=128, bits=2, seed=0)
    payload = quantizer.quantize(x)
    x_hat = quantizer.dequantize(payload)
    assert x_hat.shape == x.shape


def test_turboquant_mse_quantize_returns_packed_payload() -> None:
    x = normalize_rows(torch.randn(32, 128))
    payload = TurboQuantMSE(dim=128, bits=3, seed=0).quantize(x)
    assert payload.codes.data.dtype == torch.uint8
    assert payload.codes.bits == 3


def test_prepare_does_not_change_theoretical_codebook() -> None:
    x = torch.randn(32, 128)
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    before = quantizer.codebook.clone()
    quantizer.prepare(x)
    assert torch.allclose(quantizer.codebook, before)


def test_turboquant_mse_error_improves_with_more_bits() -> None:
    x = normalize_rows(torch.randn(64, 128))
    q2 = TurboQuantMSE(dim=128, bits=2, seed=0)
    q4 = TurboQuantMSE(dim=128, bits=4, seed=0)
    mse2 = torch.mean((x - q2.dequantize(q2.quantize(x))) ** 2)
    mse4 = torch.mean((x - q4.dequantize(q4.quantize(x))) ** 2)
    assert mse4 < mse2


def test_turboquant_mse_score_matches_dequantized_reference() -> None:
    x = normalize_rows(torch.randn(64, 64))
    q = normalize_rows(torch.randn(8, 64))
    quantizer = TurboQuantMSE(dim=64, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    scores = quantizer.score(q, payload)
    expected = q @ quantizer.dequantize(payload).T
    assert scores.shape == (8, 64)
    assert torch.allclose(scores, expected, atol=1e-5)


def test_mse_score_does_not_call_dequantize(monkeypatch) -> None:
    x = normalize_rows(torch.randn(64, 64))
    q = normalize_rows(torch.randn(8, 64))
    quantizer = TurboQuantMSE(dim=64, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    def _boom(*args, **kwargs):
        raise AssertionError("score() should not call dequantize()")

    monkeypatch.setattr(quantizer, "dequantize", _boom)
    scores = quantizer.score(q, payload)
    assert scores.shape == (8, 64)


def test_turboquant_mse_rejects_mismatched_packed_metadata() -> None:
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    payload = TurboQuantMSEPayload(
        codes=PackedCodes(data=torch.zeros(32, dtype=torch.uint8), n_rows=1, dim=128, bits=2)
    )
    with pytest.raises(ValueError, match="packed codes"):
        quantizer.dequantize(payload)


def test_turboquant_mse_rejects_non_matrix_legacy_indices() -> None:
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    payload = TurboQuantMSEPayload(indices=torch.zeros(128, dtype=torch.int64))
    with pytest.raises(ValueError, match="2D"):
        quantizer.dequantize(payload)


def test_turboquant_mse_rejects_non_integer_legacy_indices() -> None:
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    payload = TurboQuantMSEPayload(indices=torch.zeros((1, 128), dtype=torch.float32))
    with pytest.raises(ValueError, match="integer"):
        quantizer.dequantize(payload)


def test_turboquant_mse_rejects_negative_legacy_indices() -> None:
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    payload = TurboQuantMSEPayload(indices=-torch.ones((1, 128), dtype=torch.int64))
    with pytest.raises(ValueError, match="range"):
        quantizer.dequantize(payload)


def test_turboquant_mse_rejects_out_of_range_legacy_indices() -> None:
    quantizer = TurboQuantMSE(dim=128, bits=3, seed=0)
    payload = TurboQuantMSEPayload(
        indices=torch.full((1, 128), fill_value=quantizer.codebook.numel(), dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="range"):
        quantizer.dequantize(payload)

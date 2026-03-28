from __future__ import annotations

import pytest
import torch

from turboquant.math import normalize_rows
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import PackedCodes, TurboQuantProdPayload


def test_turboquant_prod_roundtrip_preserves_shape() -> None:
    x = normalize_rows(torch.randn(16, 128))
    quantizer = TurboQuantProd(dim=128, bits=3, seed=0)
    payload = quantizer.quantize(x)
    x_hat = quantizer.dequantize(payload)
    assert x_hat.shape == x.shape


def test_turboquant_prod_quantize_returns_packed_payload() -> None:
    x = normalize_rows(torch.randn(32, 128))
    payload = TurboQuantProd(dim=128, bits=3, seed=0).quantize(x)
    assert payload.mse_codes.data.dtype == torch.uint8
    assert payload.residual_signs.data.dtype == torch.uint8
    assert payload.residual_signs.bits == 1


def test_turboquant_prod_inner_product_bias_is_small() -> None:
    x = normalize_rows(torch.randn(128, 128))
    q = normalize_rows(torch.randn(128, 128))
    quantizer = TurboQuantProd(dim=128, bits=3, seed=0)
    x_hat = quantizer.dequantize(quantizer.quantize(x))
    error = torch.sum(q * (x_hat - x), dim=-1).mean().abs()
    assert error < 0.05


def test_turboquant_prod_public_api_supports_prepare_and_score() -> None:
    x = normalize_rows(torch.randn(32, 64))
    q = normalize_rows(torch.randn(8, 64))
    quantizer = TurboQuantProd(dim=64, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)
    scores = quantizer.score(q, payload)
    expected = q @ quantizer.dequantize(payload).T
    assert scores.shape == (8, 32)
    assert torch.allclose(scores, expected, atol=1e-5)


def test_turboquant_prod_score_does_not_call_dequantize(monkeypatch) -> None:
    x = normalize_rows(torch.randn(64, 64))
    q = normalize_rows(torch.randn(8, 64))
    quantizer = TurboQuantProd(dim=64, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    def _boom(*args, **kwargs):
        raise AssertionError("score() should not call dequantize()")

    monkeypatch.setattr(quantizer, "dequantize", _boom)
    scores = quantizer.score(q, payload)
    assert scores.shape == (8, 64)


def test_turboquant_prod_uses_b_minus_one_bits_for_mse_stage() -> None:
    quantizer = TurboQuantProd(dim=128, bits=3, seed=0)
    assert quantizer.mse_quantizer.bits == 2


def test_turboquant_prod_requires_at_least_two_bits() -> None:
    with pytest.raises(ValueError, match="bits must be >= 2"):
        TurboQuantProd(dim=128, bits=1, seed=0)


def test_turboquant_prod_prepare_is_non_learning() -> None:
    x = normalize_rows(torch.randn(32, 64))
    quantizer = TurboQuantProd(dim=64, bits=3, seed=0)
    cb_before = quantizer.mse_quantizer.codebook.clone()
    quantizer.prepare(x)
    assert torch.allclose(quantizer.mse_quantizer.codebook, cb_before)


def test_turboquant_prod_fit_is_backward_compatible_alias() -> None:
    x = normalize_rows(torch.randn(32, 64))
    q = normalize_rows(torch.randn(8, 64))
    quantizer = TurboQuantProd(dim=64, bits=3, seed=0)
    assert quantizer.fit(x) is quantizer
    payload = quantizer.quantize(x)
    scores = quantizer.score(q, payload)
    expected = q @ quantizer.dequantize(payload).T
    assert torch.allclose(scores, expected, atol=1e-5)


def test_prod_payload_is_smaller_than_fp32_input() -> None:
    x = normalize_rows(torch.randn(64, 128))
    payload = TurboQuantProd(dim=128, bits=3, seed=0).quantize(x)
    raw_bytes = x.numel() * x.element_size()
    assert payload.num_bytes() < raw_bytes / 4


def test_turboquant_prod_rejects_mismatched_packed_mse_codes() -> None:
    quantizer = TurboQuantProd(dim=128, bits=4, seed=0)
    payload = TurboQuantProdPayload(
        mse_codes=PackedCodes(data=torch.zeros(32, dtype=torch.uint8), n_rows=1, dim=128, bits=2),
        residual_signs=PackedCodes(data=torch.zeros(16, dtype=torch.uint8), n_rows=1, dim=128, bits=1),
        residual_norm=torch.ones(1, 1),
    )
    with pytest.raises(ValueError, match="packed MSE codes"):
        quantizer.dequantize(payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_turboquant_prod_runs_on_cuda() -> None:
    device = torch.device("cuda")
    x = normalize_rows(torch.randn(64, 64, device=device))
    q = normalize_rows(torch.randn(8, 64, device=device))
    quantizer = TurboQuantProd(dim=64, bits=3, seed=0, device=device).prepare(x)
    payload = quantizer.quantize(x)
    x_hat = quantizer.dequantize(payload)
    scores = quantizer.score(q, payload)
    assert x_hat.device.type == "cuda"
    assert payload.residual_signs.data.device.type == "cuda"
    assert payload.mse_codes.data.device.type == "cuda"
    assert scores.device.type == "cuda"

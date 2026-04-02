from __future__ import annotations

import torch
import pytest

from turboquant import TurboQuantMSE, TurboQuantProd


DIM = 64
BITS = 3
SEED = 42
N = 128


def _make_data(n: int = N, dim: int = DIM) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(n, dim, generator=gen)
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


class TestNormCorrectionMSE:
    def test_default_unchanged(self) -> None:
        """Default (norm_correction=False) produces identical output to baseline."""
        x = _make_data()
        q_base = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED)
        q_new = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=False)
        p_base = q_base.quantize(x)
        p_new = q_new.quantize(x)
        torch.testing.assert_close(q_base.dequantize(p_base), q_new.dequantize(p_new))

    def test_norm_correction_changes_output(self) -> None:
        """norm_correction=True produces different (improved) reconstruction."""
        x = _make_data()
        q_off = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=False)
        q_on = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        p = q_off.quantize(x)
        x_hat_off = q_off.dequantize(p)
        x_hat_on = q_on.dequantize(p)
        assert not torch.allclose(x_hat_off, x_hat_on, atol=1e-7)

    def test_norm_correction_reduces_mse(self) -> None:
        """norm_correction should reduce or maintain MSE for unit-normalized data."""
        x = _make_data()
        q_off = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=False)
        q_on = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        p = q_off.quantize(x)
        mse_off = (x - q_off.dequantize(p)).pow(2).mean().item()
        mse_on = (x - q_on.dequantize(p)).pow(2).mean().item()
        assert mse_on <= mse_off * 1.01, f"norm_correction MSE {mse_on} > baseline MSE {mse_off}"

    def test_score_consistent_with_dequantize(self) -> None:
        """score() must match dequantize()-based inner product with norm_correction on."""
        x = _make_data()
        query = _make_data(n=8)
        q = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        q.prepare(x)
        p = q.quantize(x)
        scores = q.score(query, p)
        x_hat = q.dequantize(p)
        expected = query @ x_hat.T
        torch.testing.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_export_state_roundtrip(self) -> None:
        """norm_correction flag persists through export/import."""
        q = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        state = q.export_state()
        assert state.norm_correction is True
        q2 = TurboQuantMSE.from_state(state)
        assert q2.norm_correction is True

        x = _make_data()
        p = q.quantize(x)
        torch.testing.assert_close(q.dequantize(p), q2.dequantize(p))


class TestNormCorrectionProd:
    def test_default_unchanged(self) -> None:
        x = _make_data()
        q_base = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED)
        q_new = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, norm_correction=False)
        p_base = q_base.quantize(x)
        p_new = q_new.quantize(x)
        torch.testing.assert_close(q_base.dequantize(p_base), q_new.dequantize(p_new))

    def test_norm_correction_changes_output(self) -> None:
        x = _make_data()
        q_off = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, norm_correction=False)
        q_on = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        p_off = q_off.quantize(x)
        p_on = q_on.quantize(x)
        x_hat_off = q_off.dequantize(p_off)
        x_hat_on = q_on.dequantize(p_on)
        assert not torch.allclose(x_hat_off, x_hat_on, atol=1e-7)

    def test_score_consistent_with_dequantize(self) -> None:
        x = _make_data()
        query = _make_data(n=8)
        q = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, norm_correction=True)
        q.prepare(x)
        p = q.quantize(x)
        scores = q.score(query, p)
        x_hat = q.dequantize(p)
        expected = query @ x_hat.T
        torch.testing.assert_close(scores, expected, atol=1e-4, rtol=1e-4)

    def test_export_state_roundtrip(self) -> None:
        q = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, norm_correction=True, fast_lookup=True)
        state = q.export_state()
        assert state.norm_correction is True
        assert state.fast_lookup is True
        q2 = TurboQuantProd.from_state(state)
        assert q2.norm_correction is True
        assert q2.fast_lookup is True

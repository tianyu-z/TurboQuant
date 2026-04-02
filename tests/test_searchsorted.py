from __future__ import annotations

import torch
import pytest

from turboquant import TurboQuantMSE, TurboQuantProd
from turboquant.codebooks import centroid_boundaries, searchsorted_quantize, solve_beta_codebook


DIM = 64
BITS = 3
SEED = 42
N = 128


def _make_data(n: int = N, dim: int = DIM) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(n, dim, generator=gen)
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


class TestCentroidBoundaries:
    def test_boundaries_shape(self) -> None:
        codebook = solve_beta_codebook(dim=DIM, bits=BITS)
        boundaries = centroid_boundaries(codebook)
        assert boundaries.shape == (codebook.numel() - 1,)

    def test_boundaries_sorted(self) -> None:
        codebook = solve_beta_codebook(dim=DIM, bits=BITS)
        boundaries = centroid_boundaries(codebook)
        assert torch.all(boundaries[:-1] <= boundaries[1:])


class TestSearchsortedQuantize:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_matches_argmin(self, bits: int) -> None:
        """searchsorted lookup must produce identical indices to broadcast argmin."""
        codebook = solve_beta_codebook(dim=DIM, bits=bits)
        boundaries = centroid_boundaries(codebook)

        gen = torch.Generator().manual_seed(SEED)
        values = torch.randn(32, DIM, generator=gen) * 0.2

        indices_searchsorted = searchsorted_quantize(values, boundaries)
        distances = torch.abs(values.unsqueeze(-1) - codebook.view(1, 1, -1))
        indices_argmin = torch.argmin(distances, dim=-1)

        torch.testing.assert_close(indices_searchsorted, indices_argmin)


class TestFastLookupMSE:
    def test_fast_lookup_matches_default(self) -> None:
        """fast_lookup=True must produce identical quantization to default."""
        x = _make_data()
        q_default = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, fast_lookup=False)
        q_fast = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, fast_lookup=True)

        p_default = q_default.quantize(x)
        p_fast = q_fast.quantize(x)

        x_hat_default = q_default.dequantize(p_default)
        x_hat_fast = q_fast.dequantize(p_fast)
        torch.testing.assert_close(x_hat_default, x_hat_fast)

    def test_fast_lookup_roundtrip(self) -> None:
        x = _make_data()
        q = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, fast_lookup=True)
        q.prepare(x)
        p = q.quantize(x)
        x_hat = q.dequantize(p)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 0.1, f"MSE too high: {mse}"

    def test_fast_lookup_export_state(self) -> None:
        q = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, fast_lookup=True)
        state = q.export_state()
        assert state.fast_lookup is True
        q2 = TurboQuantMSE.from_state(state)
        assert q2.fast_lookup is True
        assert q2._boundaries is not None

    def test_combined_options(self) -> None:
        """Both norm_correction and fast_lookup can be used together."""
        x = _make_data()
        q = TurboQuantMSE(dim=DIM, bits=BITS, seed=SEED, norm_correction=True, fast_lookup=True)
        q.prepare(x)
        p = q.quantize(x)
        x_hat = q.dequantize(p)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 0.1, f"MSE too high: {mse}"


class TestFastLookupProd:
    def test_fast_lookup_matches_default(self) -> None:
        x = _make_data()
        q_default = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, fast_lookup=False)
        q_fast = TurboQuantProd(dim=DIM, bits=BITS, seed=SEED, fast_lookup=True)

        p_default = q_default.quantize(x)
        p_fast = q_fast.quantize(x)

        x_hat_default = q_default.dequantize(p_default)
        x_hat_fast = q_fast.dequantize(p_fast)
        torch.testing.assert_close(x_hat_default, x_hat_fast)

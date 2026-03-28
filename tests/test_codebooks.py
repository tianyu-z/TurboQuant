from __future__ import annotations

import torch

from turboquant.codebooks import solve_beta_codebook


def test_codebook_has_expected_size_and_order() -> None:
    codebook = solve_beta_codebook(dim=128, bits=2, steps=8)
    assert codebook.shape == (4,)
    assert torch.all(codebook[1:] >= codebook[:-1])


def test_codebook_is_symmetric_for_symmetric_density() -> None:
    codebook = solve_beta_codebook(dim=128, bits=2, steps=8)
    assert torch.allclose(codebook, -torch.flip(codebook, dims=[0]), atol=1e-3)

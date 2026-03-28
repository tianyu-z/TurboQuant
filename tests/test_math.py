from __future__ import annotations

import torch

from turboquant import __all__
from turboquant.math import make_random_orthogonal_matrix, normalize_rows


def test_public_exports_exist() -> None:
    assert "TurboQuantMSE" in __all__
    assert "TurboQuantProd" in __all__


def test_random_orthogonal_matrix_is_orthogonal() -> None:
    q = make_random_orthogonal_matrix(dim=16, seed=0)
    eye = q.T @ q
    assert torch.allclose(eye, torch.eye(16), atol=1e-5)


def test_normalize_rows_outputs_unit_vectors() -> None:
    x = torch.randn(8, 16)
    y = normalize_rows(x)
    norms = torch.linalg.norm(y, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

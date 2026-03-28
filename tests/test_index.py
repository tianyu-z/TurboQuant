from __future__ import annotations

import pytest
import torch

from turboquant.index import TurboQuantIndex
from turboquant.math import normalize_rows


def test_index_build_and_search_returns_topk() -> None:
    data = normalize_rows(torch.randn(256, 64))
    query = normalize_rows(torch.randn(8, 64))

    index = TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, normalization="unit")
    result = index.search(query, k=5)

    assert result.indices.shape == (8, 5)
    assert result.scores.shape == (8, 5)


def test_index_search_matches_quantizer_topk() -> None:
    data = normalize_rows(torch.randn(128, 64))
    query = normalize_rows(torch.randn(4, 64))

    index = TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, normalization="unit")
    full_scores = index.quantizer.score(query, index.payload)
    expected = torch.topk(full_scores, k=3, dim=-1)

    result = index.search(query, k=3)
    assert torch.equal(result.indices, expected.indices)
    assert torch.allclose(result.scores, expected.values, atol=1e-5)


def test_index_supports_mse_algorithm() -> None:
    data = normalize_rows(torch.randn(64, 32))
    query = normalize_rows(torch.randn(2, 32))

    index = TurboQuantIndex.build(data, algorithm="mse", bits=3, seed=0, normalization="unit")
    result = index.search(query, k=2)

    assert index.algorithm == "mse"
    assert result.indices.shape == (2, 2)
    assert result.scores.shape == (2, 2)


def test_index_rejects_non_unit_normalization() -> None:
    data = normalize_rows(torch.randn(64, 32))

    with pytest.raises(ValueError, match="unit"):
        TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, normalization="raw")


def test_index_save_load_preserves_search_results(tmp_path) -> None:
    data = normalize_rows(torch.randn(128, 64))
    query = normalize_rows(torch.randn(4, 64))

    index = TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, normalization="unit")
    path = tmp_path / "toy_index.pt"
    index.save(path)
    restored = TurboQuantIndex.load(path)

    expected = index.search(query, k=4)
    actual = restored.search(query, k=4)
    assert torch.equal(actual.indices, expected.indices)
    assert torch.allclose(actual.scores, expected.scores, atol=1e-5)
    assert restored.normalization == "unit"

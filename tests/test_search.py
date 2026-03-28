from __future__ import annotations

import pytest
import torch

from turboquant.search import exact_topk_inner_product, one_at_k_recall


def test_exact_topk_inner_product_returns_indices() -> None:
    data = torch.randn(10, 8)
    query = torch.randn(2, 8)
    idx = exact_topk_inner_product(query, data, k=3)
    assert idx.shape == (2, 3)


def test_one_at_k_recall_returns_one_for_exact_match() -> None:
    exact = torch.tensor([[7], [3]])
    approx = torch.tensor([[7, 2, 1], [1, 3, 4]])
    assert one_at_k_recall(exact, approx) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_one_at_k_recall_supports_mixed_devices() -> None:
    exact = torch.tensor([[7], [3]], device="cpu")
    approx = torch.tensor([[7, 2, 1], [1, 3, 4]], device="cuda")
    assert one_at_k_recall(exact, approx) == 1.0

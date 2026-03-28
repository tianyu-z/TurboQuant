from __future__ import annotations

import torch


def exact_topk_inner_product(query: torch.Tensor, data: torch.Tensor, k: int) -> torch.Tensor:
    scores = query @ data.T
    return torch.topk(scores, k=k, dim=-1).indices


def one_at_k_recall(exact_indices: torch.Tensor, approx_indices: torch.Tensor) -> float:
    if exact_indices.ndim != 2 or approx_indices.ndim != 2:
        raise ValueError("exact_indices and approx_indices must be rank-2 tensors")
    if exact_indices.shape[0] != approx_indices.shape[0]:
        raise ValueError("exact_indices and approx_indices must share the same batch size")
    if exact_indices.shape[1] < 1 or approx_indices.shape[1] < 1:
        raise ValueError("exact_indices and approx_indices must contain at least one candidate per query")

    exact_top1 = exact_indices[:, :1].to(device=approx_indices.device)
    hits = (approx_indices == exact_top1).any(dim=-1)
    return float(hits.to(dtype=torch.float32).mean().item())

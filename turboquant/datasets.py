from __future__ import annotations

from pathlib import Path

import torch


def load_embeddings_pt(path: str | Path, key: str = "embeddings") -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and key in payload and isinstance(payload[key], torch.Tensor):
        return payload[key]
    raise ValueError(f"expected a tensor payload or a dict containing tensor key '{key}'")


def make_train_query_split(
    embeddings: torch.Tensor,
    n_query: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if n_query <= 0 or n_query >= embeddings.shape[0]:
        raise ValueError("n_query must be between 1 and len(embeddings) - 1")

    generator = torch.Generator(device=embeddings.device.type if embeddings.device.type != "mps" else "cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(embeddings.shape[0], generator=generator, device=embeddings.device)
    query_idx = permutation[:n_query]
    train_idx = permutation[n_query:]
    return embeddings[train_idx], embeddings[query_idx]

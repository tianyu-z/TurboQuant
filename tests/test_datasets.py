from __future__ import annotations

import torch

from turboquant.datasets import load_embeddings_pt, make_train_query_split


def test_load_embeddings_pt_roundtrip(tmp_path) -> None:
    embeddings = torch.randn(32, 16)
    path = tmp_path / "embeddings.pt"
    torch.save({"embeddings": embeddings}, path)

    loaded = load_embeddings_pt(path)
    assert torch.allclose(loaded, embeddings)


def test_make_train_query_split_is_disjoint() -> None:
    embeddings = torch.arange(128 * 16, dtype=torch.float32).reshape(128, 16)
    train, query = make_train_query_split(embeddings, n_query=16, seed=0)

    assert train.shape[0] == 112
    assert query.shape[0] == 16

    train_rows = {tuple(row.tolist()) for row in train}
    query_rows = {tuple(row.tolist()) for row in query}
    assert train_rows.isdisjoint(query_rows)

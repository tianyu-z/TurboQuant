from __future__ import annotations

import pytest
import torch

from turboquant.io import load_index_artifact, save_index_artifact
from turboquant.math import normalize_rows
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd


def test_save_load_artifact_preserves_scores(tmp_path) -> None:
    x = normalize_rows(torch.randn(64, 128))
    q = normalize_rows(torch.randn(8, 128))

    quantizer = TurboQuantProd(dim=128, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    path = tmp_path / "toy_index.pt"
    save_index_artifact(path, quantizer=quantizer, payload=payload, metadata={"dataset": "toy"})
    restored = load_index_artifact(path)

    expected = quantizer.score(q, payload)
    actual = restored.quantizer.score(q, restored.payload)
    assert torch.allclose(actual, expected, atol=1e-5)
    assert restored.metadata["dataset"] == "toy"
    assert restored.normalization == "unit_sphere"
    assert restored.state.format_version == 1


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_restored_artifact_scores_on_cuda_queries(tmp_path) -> None:
    x = normalize_rows(torch.randn(64, 128))
    q = normalize_rows(torch.randn(8, 128, device="cuda"))

    quantizer = TurboQuantProd(dim=128, bits=3, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    path = tmp_path / "toy_index_cuda.pt"
    save_index_artifact(path, quantizer=quantizer, payload=payload, metadata={"dataset": "toy"})
    restored = load_index_artifact(path)

    scores = restored.quantizer.score(q, restored.payload)
    assert scores.device.type == "cuda"


def test_save_load_artifact_preserves_norm_correction_and_fast_lookup(tmp_path) -> None:
    x = normalize_rows(torch.randn(64, 128))
    q = normalize_rows(torch.randn(8, 128))

    quantizer = TurboQuantProd(dim=128, bits=3, seed=0, norm_correction=True, fast_lookup=True).prepare(x)
    payload = quantizer.quantize(x)

    path = tmp_path / "enhanced_index.pt"
    save_index_artifact(path, quantizer=quantizer, payload=payload)
    restored = load_index_artifact(path)

    assert restored.state.norm_correction is True
    assert restored.state.fast_lookup is True
    assert restored.quantizer.norm_correction is True
    assert restored.quantizer.fast_lookup is True

    expected = quantizer.score(q, payload)
    actual = restored.quantizer.score(q, restored.payload)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_load_legacy_artifact_without_new_flags(tmp_path) -> None:
    """Old saved artifacts without norm_correction/fast_lookup should load with defaults."""
    x = normalize_rows(torch.randn(32, 64))
    quantizer = TurboQuantMSE(dim=64, bits=2, seed=0).prepare(x)
    payload = quantizer.quantize(x)

    path = tmp_path / "legacy_index.pt"
    save_index_artifact(path, quantizer=quantizer, payload=payload)

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    del artifact["state"]["norm_correction"]
    del artifact["state"]["fast_lookup"]
    torch.save(artifact, path)

    restored = load_index_artifact(path)
    assert restored.state.norm_correction is False
    assert restored.state.fast_lookup is False


def test_save_index_artifact_rejects_mismatched_quantizer_and_payload(tmp_path) -> None:
    x = normalize_rows(torch.randn(32, 64))
    mse_quantizer = TurboQuantMSE(dim=64, bits=2, seed=0).prepare(x)
    prod_payload = TurboQuantProd(dim=64, bits=3, seed=0).prepare(x).quantize(x)

    with pytest.raises(ValueError, match="match"):
        save_index_artifact(tmp_path / "bad_index.pt", quantizer=mse_quantizer, payload=prod_payload)

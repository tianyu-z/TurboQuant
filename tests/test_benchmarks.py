from __future__ import annotations

import pytest
import torch

from turboquant.benchmarks import run_reference_benchmark


def test_reference_benchmark_reports_storage_and_timing() -> None:
    metrics = run_reference_benchmark(dim=64, n_data=256, n_query=32, bits=[2], seed=0)
    assert "mse" in metrics
    assert "inner_product_bias" in metrics
    assert "inner_product_mse" in metrics
    assert "one_at_k_recall" in metrics
    assert "payload_bytes_per_vector" in metrics
    assert "amortized_total_bytes_per_vector" in metrics
    assert "payload_compression_ratio_vs_fp32" in metrics
    assert "amortized_total_compression_ratio_vs_fp32" in metrics
    assert "build_time_s" in metrics
    assert "query_time_s" in metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reference_benchmark_runs_on_cuda() -> None:
    metrics = run_reference_benchmark(
        dim=32,
        n_data=128,
        n_query=16,
        bits=[2],
        seed=0,
        device=torch.device("cuda"),
    )
    assert "mse" in metrics
    assert "build_time_s" in metrics

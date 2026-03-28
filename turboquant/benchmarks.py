from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import torch

from turboquant.index import TurboQuantIndex
from turboquant.math import _make_generator, normalize_rows
from turboquant.search import exact_topk_inner_product, one_at_k_recall
from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd


def _make_synthetic_data(
    dim: int,
    n_data: int,
    n_query: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(seed, device=device)
    data = normalize_rows(torch.randn(n_data, dim, generator=generator, device=device))
    query = normalize_rows(torch.randn(n_query, dim, generator=generator, device=device))
    return data, query


def _sync_if_cuda(device: torch.device | str) -> None:
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _quantizer_state_nbytes(quantizer: TurboQuantProd) -> int:
    state = quantizer.export_state()
    return (
        _tensor_nbytes(state.codebook)
        + _tensor_nbytes(state.rotation)
        + _tensor_nbytes(state.projection)
    )


def run_reference_benchmark(
    dim: int,
    n_data: int,
    n_query: int,
    bits: Sequence[int],
    seed: int = 0,
    k: int = 10,
    device: torch.device | str = "cpu",
) -> dict[str, dict[int, float]]:
    data, query = _make_synthetic_data(dim=dim, n_data=n_data, n_query=n_query, seed=seed, device=device)
    exact_idx = exact_topk_inner_product(query, data, k=k)

    metrics: dict[str, dict[int, float]] = {
        "mse": {},
        "inner_product_bias": {},
        "inner_product_mse": {},
        "one_at_k_recall": {},
        # Temporary compatibility alias. Prefer one_at_k_recall in new code.
        "recall_at_k": {},
        "payload_bytes_per_vector": {},
        "amortized_total_bytes_per_vector": {},
        "payload_compression_ratio_vs_fp32": {},
        "amortized_total_compression_ratio_vs_fp32": {},
        "build_time_s": {},
        "query_time_s": {},
    }
    fp32_bytes_per_vector = float(dim * data.element_size())

    for bit_width in bits:
        mse_quantizer = TurboQuantMSE(dim=dim, bits=bit_width, seed=seed, device=device)
        _sync_if_cuda(device)
        build_start = perf_counter()
        index = TurboQuantIndex.build(
            data,
            algorithm="prod",
            bits=bit_width,
            seed=seed,
            device=str(device),
            normalization="unit",
        )
        _sync_if_cuda(device)
        metrics["build_time_s"][bit_width] = perf_counter() - build_start
        prod_quantizer = index.quantizer

        data_mse = mse_quantizer.dequantize(mse_quantizer.quantize(data))
        data_prod = prod_quantizer.dequantize(index.payload)

        metrics["mse"][bit_width] = float(torch.mean((data - data_mse) ** 2).item())

        paired_data = data[:n_query]
        paired_prod = data_prod[:n_query]
        exact_scores = torch.sum(query * paired_data, dim=-1)
        approx_scores = torch.sum(query * paired_prod, dim=-1)
        ip_error = approx_scores - exact_scores
        metrics["inner_product_bias"][bit_width] = float(ip_error.mean().abs().item())
        metrics["inner_product_mse"][bit_width] = float(torch.mean(ip_error**2).item())

        _sync_if_cuda(device)
        query_start = perf_counter()
        approx = index.search(query, k=k)
        _sync_if_cuda(device)
        metrics["query_time_s"][bit_width] = perf_counter() - query_start

        recall = one_at_k_recall(exact_idx, approx.indices)
        metrics["one_at_k_recall"][bit_width] = recall
        metrics["recall_at_k"][bit_width] = recall

        payload_bytes_per_vector = float(index.payload.bytes_per_vector())
        amortized_total_bytes_per_vector = payload_bytes_per_vector + (
            _quantizer_state_nbytes(prod_quantizer) / float(index.n_rows)
        )
        metrics["payload_bytes_per_vector"][bit_width] = payload_bytes_per_vector
        metrics["amortized_total_bytes_per_vector"][bit_width] = amortized_total_bytes_per_vector
        metrics["payload_compression_ratio_vs_fp32"][bit_width] = fp32_bytes_per_vector / payload_bytes_per_vector
        metrics["amortized_total_compression_ratio_vs_fp32"][bit_width] = (
            fp32_bytes_per_vector / amortized_total_bytes_per_vector
        )

    return metrics

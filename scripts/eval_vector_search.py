from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.datasets import load_embeddings_pt, make_train_query_split
from turboquant.index import TurboQuantIndex
from turboquant.math import normalize_rows
from turboquant.search import exact_topk_inner_product, one_at_k_recall


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _state_nbytes(index: TurboQuantIndex) -> int:
    state = index.quantizer.export_state()
    return (
        _tensor_nbytes(state.codebook)
        + _tensor_nbytes(state.rotation)
        + _tensor_nbytes(state.projection)
    )


def _compression_ratio(raw_bytes_per_vector: float, compressed_bytes_per_vector: float) -> float:
    if compressed_bytes_per_vector == 0.0:
        return math.inf
    return raw_bytes_per_vector / compressed_bytes_per_vector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TurboQuant vector-search recall on local embedding tensors.")
    parser.add_argument("--train-embeddings", required=True, type=Path)
    parser.add_argument("--query-embeddings", type=Path, default=None)
    parser.add_argument("--algorithm", choices=("mse", "prod"), default="prod")
    parser.add_argument("--bits", nargs="+", required=True, type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-query", type=int, default=None)
    return parser.parse_args()


def _prepare_embeddings(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    train_embeddings = load_embeddings_pt(args.train_embeddings)
    if args.limit_train is not None:
        train_embeddings = train_embeddings[: args.limit_train]

    if args.query_embeddings is not None:
        query_embeddings = load_embeddings_pt(args.query_embeddings)
        if args.limit_query is not None:
            query_embeddings = query_embeddings[: args.limit_query]
        return train_embeddings, query_embeddings

    n_query = args.limit_query
    if n_query is None:
        n_query = min(1000, max(1, train_embeddings.shape[0] // 10))
    train_embeddings, query_embeddings = make_train_query_split(train_embeddings, n_query=n_query, seed=args.seed)
    return train_embeddings, query_embeddings


def main() -> None:
    args = _parse_args()
    train_embeddings, query_embeddings = _prepare_embeddings(args)
    normalized_train = normalize_rows(train_embeddings)
    normalized_query = normalize_rows(query_embeddings)
    exact_topk = exact_topk_inner_product(normalized_query, normalized_train, k=args.k)
    fp32_bytes_per_vector = float(normalized_train.shape[1] * normalized_train.element_size())

    results: list[dict[str, object]] = []
    for bit_width in args.bits:
        build_start = perf_counter()
        index = TurboQuantIndex.build(
            normalized_train,
            algorithm=args.algorithm,
            bits=bit_width,
            seed=args.seed,
            device=args.device,
            normalization="unit",
        )
        build_time_s = perf_counter() - build_start

        query_start = perf_counter()
        approx = index.search(normalized_query, k=args.k)
        query_time_s = perf_counter() - query_start

        payload_bytes_per_vector = float(index.payload.bytes_per_vector())
        amortized_total_bytes_per_vector = payload_bytes_per_vector + (_state_nbytes(index) / float(index.n_rows))
        results.append(
            {
                "algorithm": args.algorithm,
                "bits": bit_width,
                "k": args.k,
                "normalization": "unit",
                "train_size": int(normalized_train.shape[0]),
                "query_size": int(normalized_query.shape[0]),
                "dim": int(normalized_train.shape[1]),
                "paper_faithful_core": {
                    "one_at_k_recall": one_at_k_recall(exact_topk, approx.indices),
                },
                "engineering_extensions": {
                    "build_time_s": build_time_s,
                    "query_time_s": query_time_s,
                    "payload_bytes_per_vector": payload_bytes_per_vector,
                    "amortized_total_bytes_per_vector": amortized_total_bytes_per_vector,
                    "payload_compression_ratio_vs_fp32": _compression_ratio(
                        fp32_bytes_per_vector,
                        payload_bytes_per_vector,
                    ),
                    "amortized_total_compression_ratio_vs_fp32": _compression_ratio(
                        fp32_bytes_per_vector,
                        amortized_total_bytes_per_vector,
                    ),
                },
            }
        )

    print(json.dumps({"algorithm": args.algorithm, "normalization": "unit", "results": results}, indent=2))


if __name__ == "__main__":
    main()

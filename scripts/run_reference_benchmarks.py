from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.benchmarks import run_reference_benchmark


CORE_KEYS = (
    "mse",
    "inner_product_bias",
    "inner_product_mse",
    "one_at_k_recall",
)
SYSTEM_KEYS = (
    "payload_bytes_per_vector",
    "amortized_total_bytes_per_vector",
    "payload_compression_ratio_vs_fp32",
    "amortized_total_compression_ratio_vs_fp32",
    "build_time_s",
    "query_time_s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TurboQuant reference benchmarks")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-data", type=int, default=4096)
    parser.add_argument("--n-query", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    metrics = run_reference_benchmark(
        dim=args.dim,
        n_data=args.n_data,
        n_query=args.n_query,
        bits=args.bits,
        seed=args.seed,
        k=args.k,
        device=args.device,
    )
    output = {
        "paper_faithful_core": {key: metrics[key] for key in CORE_KEYS},
        "engineering_extensions": {key: metrics[key] for key in SYSTEM_KEYS},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

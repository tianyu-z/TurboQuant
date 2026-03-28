# TurboQuant

PyTorch reference implementation of the TurboQuant components that are fully specified in `arXiv-2504.19874v1`, plus a C-stage engineering layer for packed payloads, persistence, vector search, and evaluation.

## Paper-Faithful Core

The paper-backed math stays fixed here:

- `QJLQuantizer`
  - Gaussian sketch matrix
  - 1-bit sign quantization
  - dequantization scale fixed to `sqrt(pi / 2) / d`
- `TurboQuantMSE`
  - random orthogonal rotation
  - theoretical scalar codebook
  - data-oblivious `prepare(...)`
  - quantize / dequantize / compressed-domain score
- `TurboQuantProd`
  - exact paper split `TurboQuantMSE(bits - 1) + residual QJL`
  - requires `bits >= 2`
  - quantize / dequantize / compressed-domain score

`prepare(...)` is the public setup API because it does not learn from data. It validates shapes, moves state to the target device, and preserves the paper's fixed codebook / rotation / projection setup. `fit(...)` remains only as a backward-compatible non-learning alias.

## C-Stage Engineering Extensions

The engineering layer adds reusable search and persistence on top of the frozen core:

- packed payload dataclasses backed by `torch.uint8`
- quantizer state export / import without regenerating random matrices
- artifact save / load helpers
- compressed-domain scoring for both `TurboQuantMSE` and `TurboQuantProd`
- `TurboQuantIndex` build / search / save / load API
- local `.pt` embedding loading and disjoint train/query splitting
- synthetic benchmark and real-data evaluation CLIs

Packed payloads stay packed in memory and on disk. Reconstruction may unpack internally, but the resident payload object reports real storage via `num_bytes()` and `bytes_per_vector()`.

## Normalization Contract

The default vector-search path is intentionally unit-normalized.

- Index build normalizes database vectors with `normalize_rows(...)`.
- Index search normalizes query vectors with `normalize_rows(...)`.
- Saved index artifacts persist `normalization="unit"`.
- Support for arbitrary non-normalized vectors is explicitly out of scope for this plan.

## Basic Usage

```python
import torch

from turboquant import TurboQuantIndex

data = torch.randn(1024, 128)
query = torch.randn(16, 128)

index = TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, device="cuda", normalization="unit")
result = index.search(query, k=10)

index.save("toy_index.pt")
restored = TurboQuantIndex.load("toy_index.pt", device="cuda")
restored_result = restored.search(query, k=10)
```

## Save and Load Artifacts

If you want direct access to the lower-level persistence helpers, use `save_index_artifact(...)` / `load_index_artifact(...)` from `turboquant.io`. They store:

- algorithm name
- quantizer state tensors
- packed payload tensors
- scalar metadata such as `dim`, `bits`, `n_rows`, `format_version`
- persisted normalization mode

## Run Tests

Use the `py311` Conda environment:

```bash
conda run -n py311 python -m pytest -v
```

## Run Synthetic Benchmarks

This CLI prints paper-faithful quality metrics separately from engineering-extension systems metrics:

```bash
conda run -n py311 python scripts/run_reference_benchmarks.py \
  --dim 128 \
  --n-data 4096 \
  --n-query 256 \
  --bits 2 3 4 \
  --device cuda
```

## Run Real-Data Vector Search Evaluation

For local `.pt` embeddings:

```bash
conda run -n py311 python scripts/eval_vector_search.py \
  --train-embeddings /absolute/path/to/embeddings.pt \
  --algorithm prod \
  --bits 3 4 \
  --k 10 \
  --device cuda \
  --limit-train 100000 \
  --limit-query 1000
```

The JSON output reports `paper_faithful_core.one_at_k_recall` and the engineering metrics for build/query time and storage.

## Paper Reproduction Status

Nearest-neighbor validation was run against the paper-aligned DBpedia OpenAI3 datasets:

- `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K`
- `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-100K`

The local check used a `99000` train + `1000` query split because the published `100K` repos do not ship a separate query set. Full notes are recorded in `docs/experiments/results/2026-03-27-paper-nn-validation.md`.

What reproduced:

- the qualitative nearest-neighbor trend from Figure 5
- higher recall with more bits
- near-perfect `1@k` once `k` is moderately large

What did not reproduce:

- the paper's top-1 numbers closely enough to claim an exact reproduction
- any KV-cache / `PolarQuant` "no precision loss" claim

Observed `TurboQuant_prod` recall:

| dataset | bits | 1@1 | 1@2 | 1@4 | 1@8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 large `d=1536` | 2 | 0.7360 | 0.8940 | 0.9690 | 0.9940 |
| OpenAI3 large `d=1536` | 4 | 0.9110 | 0.9870 | 1.0000 | 1.0000 |
| OpenAI3 large `d=3072` | 2 | 0.8230 | 0.9480 | 0.9920 | 1.0000 |
| OpenAI3 large `d=3072` | 4 | 0.9340 | 0.9960 | 1.0000 | 1.0000 |

Paper Figure 5 is a curve rather than a table, but visually the paper appears closer to about `0.90` / `0.97` at `1@1` for the `2-bit` / `4-bit` `prod` curves. This implementation therefore matches the paper directionally, but currently undershoots its reported top-1 quality.

## Explicitly Deferred

The repository still does not claim:

- `PolarQuant`
- faithful KV-cache reproduction from the supplied tarball
- the paper's undocumented `2.5` / `3.5` bit outlier-channel recipe
- custom kernels, Triton kernels, or ANN structures

## Layout

- `turboquant/codebooks.py`: scalar codebook solver
- `turboquant/datasets.py`: local embedding loading and train/query splitting
- `turboquant/index.py`: reusable vector-search index API
- `turboquant/io.py`: packed artifact persistence
- `turboquant/math.py`: orthogonal rotations and normalization
- `turboquant/qjl.py`: QJL primitive
- `turboquant/search.py`: exact search and `1@k` recall helpers
- `turboquant/turboquant_mse.py`: paper-faithful MSE quantizer
- `turboquant/turboquant_prod.py`: paper-faithful residual-corrected quantizer
- `turboquant/benchmarks.py`: synthetic benchmark harness
- `scripts/run_reference_benchmarks.py`: synthetic benchmark CLI
- `scripts/eval_vector_search.py`: local embedding evaluation CLI
- `tests/`: unit and smoke tests

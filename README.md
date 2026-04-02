# TurboQuant

PyTorch reference implementation of the TurboQuant vector quantization algorithm from `arXiv-2504.19874v1`. Includes packed payloads, compressed-domain scoring, persistence, vector search, and optional enhancements for KV-cache compression.

## Algorithm Overview

TurboQuant compresses high-dimensional vectors via two stages:

1. **MSE stage** (`TurboQuantMSE`): Random orthogonal rotation + per-coordinate scalar quantization using a theoretically optimal codebook derived from the spherical-coordinate Beta distribution.
2. **Inner-product stage** (`TurboQuantProd`): Combines `TurboQuantMSE(bits - 1)` with a 1-bit QJL (Quantized Johnson-Lindenstrauss) residual correction to better preserve inner products.

Both stages are **data-oblivious** — codebooks, rotations, and projections are fixed at construction time and do not learn from data.

### Core Classes

| Class | Purpose | Use case |
|-------|---------|----------|
| `TurboQuantMSE` | MSE-optimal quantization | V cache, embedding storage |
| `TurboQuantProd` | Inner-product-preserving quantization | K cache, similarity search |
| `TurboQuantIndex` | Build / search / save / load API | Vector search pipelines |

`QJLQuantizer` is used internally by `TurboQuantProd` but is not exported from the package root.

## Quick Start

```python
import torch
from turboquant import TurboQuantIndex

data = torch.randn(1024, 128)
query = torch.randn(16, 128)

# Build, search, save, reload
index = TurboQuantIndex.build(data, algorithm="prod", bits=3, seed=0, device="cuda")
result = index.search(query, k=10)

index.save("index.pt")
restored = TurboQuantIndex.load("index.pt", device="cuda")
```

### Low-Level Quantizer API

```python
import torch
from turboquant import TurboQuantProd

data = torch.randn(1024, 128)
data = data / data.norm(dim=-1, keepdim=True)   # unit-normalize (required for quality guarantees)
query = torch.randn(16, 128)
query = query / query.norm(dim=-1, keepdim=True)

q = TurboQuantProd(dim=128, bits=3, seed=0, norm_correction=True, fast_lookup=True)
q.prepare(data)

payload = q.quantize(data)            # -> TurboQuantProdPayload (packed codes + residual norm)
x_hat = q.dequantize(payload)         # -> reconstructed torch.Tensor
scores = q.score(query, payload)      # -> compressed-domain inner products

state = q.export_state()              # -> QuantizerState dataclass
q2 = TurboQuantProd.from_state(state) # -> reconstructed quantizer
```

## Optional Enhancements

Two opt-in flags (`norm_correction`, `fast_lookup`) and one usage recommendation (asymmetric K/V). Both flags default to `False`, preserving the paper-faithful baseline.

### Norm Correction

Re-normalizes the reconstructed rotated vector before inverse rotation. Scalar quantization distorts the norm of the rotated representation; this correction is mathematically free but has dramatic impact on LLM perplexity.

```python
q = TurboQuantProd(dim=128, bits=3, seed=0, norm_correction=True)
```

**Experimentally validated** on Qwen2.5-3B (wikitext-2, 4096 tokens, CPU):

| Method | PPL | K-cache MSE | vs fp16 |
|--------|----:|------------:|--------:|
| fp16 baseline | 7.77 | — | — |
| TQ 3-bit | 1355.09 | 0.3542 | catastrophic |
| **TQ+NC 3-bit** | **68.43** | **0.3507** | +60.66 |
| TQ 4-bit | 19.63 | 0.0963 | +11.86 |
| **TQ+NC 4-bit** | **8.64** | **0.0949** | **+0.87** |

Key insight: MSE improves only 1%, but PPL improves **19.8x** at 3-bit. Softmax amplifies norm distortion in attention scores.

The flag propagates through `export_state()` / `from_state()` and is preserved across save/load. When enabled, `score()` stays consistent with `dequantize()`.

### Fast Lookup

Replaces the default `O(d * k)` broadcast-argmin with `O(d log k)` `torch.searchsorted` on precomputed centroid boundaries. Produces identical indices.

```python
q = TurboQuantMSE(dim=128, bits=4, seed=0, fast_lookup=True)
```

Both options can be combined:

```python
q = TurboQuantProd(dim=128, bits=3, seed=0, norm_correction=True, fast_lookup=True)
```

### Asymmetric K/V Strategy

When compressing KV caches, K and V have different optimization objectives:

- **K cache** -> `TurboQuantProd` (inner-product preservation for `Q @ K^T` attention scores)
- **V cache** -> `TurboQuantMSE` (MSE preservation for `attn_weights @ V` output)

```python
from turboquant import TurboQuantProd, TurboQuantMSE

head_dim = 128
k_quantizer = TurboQuantProd(dim=head_dim, bits=3, seed=0, norm_correction=True)
v_quantizer = TurboQuantMSE(dim=head_dim, bits=3, seed=1, norm_correction=True)
```

**Validated via llama.cpp** on Qwen2.5-3B Q4_K_M (A100, wikitext-2, 512 ctx):

| K cache | V cache | PPL | vs q8_0 baseline |
|---------|---------|----:|------:|
| q8_0 | q8_0 | 10.00 | baseline |
| q8_0 | turbo4 | 10.05 | **+0.5%** |
| q8_0 | turbo3 | 10.09 | **+0.9%** |
| turbo3 | turbo3 | 174.40 | catastrophic |

Asymmetric `q8_0-K + turbo-V` is safe; symmetric turbo on small sensitive models is not. K precision dominates quality through softmax amplification.

Note: this repo's normalization contract assumes unit vectors. A full KV-cache pipeline additionally needs per-vector norm extraction and rescaling.

## Experimental Results

Results below were measured on an NVIDIA A100-SXM4-80GB. Quantization quality (MSE, recall, inner-product error) can be reproduced using the in-repo benchmark scripts. Norm-correction PPL and llama.cpp integration results require external tooling (HuggingFace `transformers` and the TurboQuant llama.cpp fork respectively) and are reported here for reference.

### Quantization Quality (unit-normalized vectors)

MSE and cosine similarity on 500 random vectors (`TurboQuantMSE`):

| dim | bits | MSE | Cosine Sim |
|----:|-----:|---------:|----------:|
| 64 | 2 | 0.001790 | 0.9417 |
| 64 | 3 | 0.000532 | 0.9832 |
| 64 | 4 | 0.000148 | 0.9954 |
| 128 | 2 | 0.000905 | 0.9407 |
| 128 | 3 | 0.000266 | 0.9830 |
| 128 | 4 | 0.000074 | 0.9954 |
| 256 | 3 | 0.000133 | 0.9830 |
| 256 | 4 | 0.000037 | 0.9953 |

### llama.cpp Integration (A100, Qwen2.5-3B Q8_0)

Perplexity (wikitext-2, 512 context, 8 chunks):

| KV cache type | PPL | vs q8_0 | KV size |
|---------------|----:|--------:|--------:|
| q8_0 | 9.82 | baseline | 38.25 MiB |
| turbo4 | 12.42 | +26.4% | 19.12 MiB |
| turbo3 | 145.47 | +1381% | 14.06 MiB |

Note: Qwen2.5-3B is exceptionally KV-sensitive — even standard `q4_0` KV cache gives +92.5% PPL on this model. Reference benchmarks on larger models (Qwen3.5-35B) show turbo3 at +1.06% and turbo4 at +0.23%.

Decode speed:

| KV cache | Decode tok/s | vs q8_0 | Prefill tok/s (4K) | vs q8_0 |
|----------|------------:|--------:|-------------------:|--------:|
| q8_0 | 200.7 | baseline | 9473 | baseline |
| turbo4 | 172.5 | 0.86x | — | — |
| turbo3 | 178.6 | 0.89x | 9046 | 0.96x |

Context scaling (turbo3 vs q8_0):

| Context depth | Prefill ratio | Decode ratio |
|--------------:|--------------:|-------------:|
| 0 | — | 0.89x |
| 2K | 0.96x | 0.89x |
| 4K | 0.96x | 0.89x |
| 8K | 0.95x | 0.89x |

Decode overhead is flat across context depths. Prefill near parity (~96%).

### Vector Search Recall

Evaluated on DBpedia OpenAI3 embeddings (99K train, 1K query) using `scripts/eval_vector_search.py`:

| dataset | bits | 1@1 | 1@2 | 1@4 | 1@8 |
|---------|-----:|----:|----:|----:|----:|
| `d=1536` | 2 | 0.736 | 0.894 | 0.969 | 0.994 |
| `d=1536` | 4 | 0.911 | 0.987 | 1.000 | 1.000 |
| `d=3072` | 2 | 0.823 | 0.948 | 0.992 | 1.000 |
| `d=3072` | 4 | 0.934 | 0.996 | 1.000 | 1.000 |

These match the paper's Figure 5 directionally but undershoot its top-1 numbers.

## Engineering Features

- **Packed payloads**: Bit-level `torch.uint8` encoding with exact byte accounting via `num_bytes()` / `bytes_per_vector()`
- **Compressed-domain scoring**: `score()` computes approximate inner products without dequantization
- **State persistence**: `export_state()` / `from_state()` + artifact save/load via `turboquant.io`
- **Device management**: `.to(device)` for CPU/CUDA migration
- **Legacy compatibility**: Supports both packed `PackedCodes` and legacy 2D tensor indices

## Compression Ratios

`TurboQuantMSE` stores only packed codebook indices. Exact byte count: `ceil(n * d * bits / 8)` per batch; approximate bytes/vector: `d * bits / 8`.

| bits | Approx bytes/vector | vs fp32 | vs fp16 |
|-----:|--------------------:|--------:|--------:|
| 2 | `d / 4` | 16x | 8x |
| 3 | `3d / 8` | 10.7x | 5.3x |
| 4 | `d / 2` | 8x | 4x |

`TurboQuantProd` stores MSE codes at `(bits-1)` per coordinate, 1-bit packed residual signs, and a 4-byte float32 `residual_norm` per vector. Approximate bytes/vector: `d * bits / 8 + 4`. Exact byte counts use per-buffer `ceil()` rounding (see `TurboQuantProdPayload.num_bytes()`).

| bits | Approx bytes/vector (d=128) | vs fp32 | vs fp16 |
|-----:|----------------------------:|--------:|--------:|
| 3 | `128*3/8 + 4` = 52 | 9.8x | 4.9x |
| 4 | `128*4/8 + 4` = 68 | 7.5x | 3.8x |

## Normalization Contract

The default vector-search path is intentionally unit-normalized:

- `TurboQuantIndex.build()` normalizes database vectors.
- `TurboQuantIndex.search()` normalizes query vectors.
- The low-level quantizer API (`TurboQuantMSE`, `TurboQuantProd`) does **not** normalize automatically. Quality guarantees assume unit-normalized input.

## Run Tests

```bash
conda run -n py311 python -m pytest tests/ -v
```

86 tests (including norm_correction and searchsorted tests). Expected: all pass.

## Run Benchmarks

Synthetic:
```bash
conda run -n py311 python scripts/run_reference_benchmarks.py \
  --dim 128 --n-data 4096 --n-query 256 --bits 2 3 4 --device cuda
```

Real embeddings:
```bash
conda run -n py311 python scripts/eval_vector_search.py \
  --train-embeddings /path/to/embeddings.pt \
  --algorithm prod --bits 3 4 --k 10 --device cuda
```

## Layout

```
turboquant/
  codebooks.py          Scalar codebook solver (Beta PDF + searchsorted helpers)
  turboquant_mse.py     MSE quantizer (norm_correction, fast_lookup options)
  turboquant_prod.py    Inner-product quantizer (MSE + QJL residual)
  qjl.py                1-bit QJL primitive
  types.py              PackedCodes, payloads, QuantizerState
  packing.py            Bit-level pack/unpack
  index.py              TurboQuantIndex build/search/save/load
  io.py                 Artifact persistence
  math.py               Orthogonal rotations, normalization
  search.py             Exact search, 1@k recall
  datasets.py           Embedding loading, train/query splitting
  benchmarks.py         Synthetic benchmark harness
scripts/
  run_reference_benchmarks.py    Synthetic benchmark CLI
  eval_vector_search.py          Real embedding evaluation CLI
tests/                           86 unit and integration tests
```

## Explicitly Deferred

- PolarQuant (norm-aware variant for non-unit vectors)
- Faithful KV-cache pipeline with per-vector norm storage
- The paper's undocumented 2.5 / 3.5 bit outlier-channel recipe
- Custom kernels, Triton kernels, or ANN structures

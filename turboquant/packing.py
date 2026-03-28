from __future__ import annotations

import math

import torch

from turboquant.types import PackedCodes


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_row_major_tensor(values: torch.Tensor) -> None:
    if values.ndim != 2 or not values.is_contiguous():
        raise ValueError("packing expects a row-major contiguous 2D tensor")


def _validate_bits(bits: int) -> None:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in [1, 8]")


def _validate_packed_data(data: torch.Tensor, n_rows: int, dim: int, bits: int) -> None:
    if data.dtype != torch.uint8:
        raise ValueError("packed data must use torch.uint8 storage")
    if data.ndim != 1 or not data.is_contiguous():
        raise ValueError("packed data must be a flat contiguous byte buffer")
    if n_rows < 0 or dim < 0:
        raise ValueError("logical shape must be non-negative")
    _validate_bits(bits)


def pack_codes(codes: torch.Tensor, bits: int) -> PackedCodes:
    _validate_row_major_tensor(codes)
    _validate_bits(bits)
    if codes.dtype not in _INTEGER_DTYPES:
        raise ValueError("codes must be a real integer tensor")

    max_code = 1 << bits
    codes_i64 = codes.to(torch.int64)
    if torch.any(codes_i64 < 0) or torch.any(codes_i64 >= max_code):
        raise ValueError("code values must lie within the representable range")

    flat = codes_i64.reshape(-1)
    total_bits = flat.numel() * bits
    total_bytes = math.ceil(total_bits / 8) if total_bits else 0
    packed_words = torch.zeros(total_bytes, dtype=torch.int64, device=flat.device)
    if total_bits:
        bit_offsets = torch.arange(bits, dtype=torch.int64, device=flat.device)
        bit_values = ((flat.unsqueeze(-1) >> bit_offsets) & 1).reshape(-1)
        bit_positions = torch.arange(total_bits, dtype=torch.int64, device=flat.device)
        byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
        bit_indices = bit_positions % 8
        packed_words.scatter_add_(0, byte_indices, bit_values << bit_indices)

    return PackedCodes(
        data=packed_words.to(torch.uint8),
        n_rows=int(codes.shape[0]),
        dim=int(codes.shape[1]),
        bits=bits,
    )


def unpack_codes(data: torch.Tensor, n_rows: int, dim: int, bits: int) -> torch.Tensor:
    _validate_packed_data(data, n_rows=n_rows, dim=dim, bits=bits)

    total_values = n_rows * dim
    total_bits = total_values * bits
    expected_bytes = math.ceil(total_bits / 8) if total_bits else 0
    if data.numel() != expected_bytes:
        raise ValueError("packed data size does not match the requested logical shape")
    if total_values == 0:
        return torch.empty((n_rows, dim), dtype=torch.int64, device=data.device)

    bit_positions = torch.arange(total_bits, dtype=torch.int64, device=data.device)
    byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
    bit_indices = bit_positions % 8
    bit_values = ((data[byte_indices].to(torch.int64) >> bit_indices) & 1).reshape(total_values, bits)
    bit_weights = (1 << torch.arange(bits, dtype=torch.int64, device=data.device)).view(1, bits)
    return (bit_values * bit_weights).sum(dim=-1).reshape(n_rows, dim)


def pack_signs(signs: torch.Tensor) -> PackedCodes:
    _validate_row_major_tensor(signs)
    if signs.is_complex():
        raise ValueError("sign values must be real-valued")
    if not torch.all((signs == -1) | (signs == 1)):
        raise ValueError("sign values must be either -1 or 1")
    signs_i64 = signs.to(torch.int64)
    return pack_codes((signs_i64 > 0).to(torch.int64), bits=1)


def unpack_signs(data: torch.Tensor, n_rows: int, dim: int) -> torch.Tensor:
    bits = unpack_codes(data, n_rows=n_rows, dim=dim, bits=1)
    return torch.where(bits > 0, torch.ones_like(bits, dtype=torch.int8), -torch.ones_like(bits, dtype=torch.int8))

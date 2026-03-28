from __future__ import annotations

import pytest
import torch

from turboquant.packing import pack_codes, pack_signs, unpack_codes, unpack_signs
from turboquant.qjl import QJLQuantizer


def test_pack_unpack_roundtrip_for_two_bit_codes() -> None:
    codes = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64)
    packed = pack_codes(codes, bits=2)
    restored = unpack_codes(packed.data, n_rows=2, dim=4, bits=2)
    assert torch.equal(restored, codes)


def test_pack_unpack_roundtrip_for_three_bit_codes() -> None:
    codes = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 0, 1]], dtype=torch.int64)
    packed = pack_codes(codes, bits=3)
    restored = unpack_codes(packed.data, n_rows=2, dim=5, bits=3)
    assert torch.equal(restored, codes)


def test_pack_unpack_roundtrip_for_sign_bits() -> None:
    signs = torch.tensor([[1, -1, 1, -1, -1, 1, 1, -1]], dtype=torch.int8)
    packed = pack_signs(signs)
    restored = unpack_signs(packed.data, n_rows=1, dim=8)
    assert torch.equal(restored, signs)


def test_pack_codes_rejects_values_out_of_range() -> None:
    codes = torch.tensor([[0, 1, 4, 3]], dtype=torch.int64)
    with pytest.raises(ValueError, match="range"):
        pack_codes(codes, bits=2)


def test_pack_codes_rejects_complex_values() -> None:
    codes = torch.tensor([[1 + 2j, 2 + 0j]])
    with pytest.raises(ValueError, match="integer"):
        pack_codes(codes, bits=2)


def test_pack_signs_rejects_complex_values() -> None:
    signs = torch.tensor([[1 + 2j, -1 + 3j]])
    with pytest.raises(ValueError, match="real-valued"):
        pack_signs(signs)


def test_pack_signs_rejects_fractional_values() -> None:
    signs = torch.tensor([[1.9, -1.1]], dtype=torch.float32)
    with pytest.raises(ValueError, match="either -1 or 1"):
        pack_signs(signs)


def test_unpacked_signs_compose_with_qjl_dequantize() -> None:
    qjl = QJLQuantizer(dim=8, seed=0)
    signs = torch.tensor([[1, -1, 1, -1, -1, 1, 1, -1]], dtype=torch.int8)
    restored = unpack_signs(pack_signs(signs).data, n_rows=1, dim=8)
    decoded = qjl.dequantize(restored)
    assert decoded.shape == (1, 8)


def test_pack_codes_requires_row_major_input() -> None:
    codes = torch.arange(8, dtype=torch.int64).reshape(2, 4).transpose(0, 1)
    with pytest.raises(ValueError, match="row-major"):
        pack_codes(codes, bits=3)

from __future__ import annotations

import math

import pytest
import torch

from turboquant.qjl import QJLQuantizer


def test_qjl_quantize_outputs_binary_signs() -> None:
    qjl = QJLQuantizer(dim=32, seed=0)
    x = torch.randn(4, 32)
    z = qjl.quantize(x)
    assert set(torch.unique(z).tolist()) <= {-1.0, 1.0}


def test_qjl_decode_preserves_shape() -> None:
    qjl = QJLQuantizer(dim=32, seed=0)
    x = torch.randn(4, 32)
    z = qjl.quantize(x)
    y = qjl.dequantize(z)
    assert y.shape == x.shape


def test_qjl_uses_paper_scale_factor() -> None:
    qjl = QJLQuantizer(dim=64, seed=0)
    assert qjl.scale == pytest.approx(math.sqrt(math.pi / 2.0) / 64)

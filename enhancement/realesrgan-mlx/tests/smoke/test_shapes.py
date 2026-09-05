"""Smoke tests — no torch, no weights. Verify shapes + spatial-op invariants."""

import mlx.core as mx
import pytest

from realesrgan_mlx.archs import RRDBNet, SRVGGNetCompact
from realesrgan_mlx.config import VARIANTS
from realesrgan_mlx.utils.spatial import (
    interpolate_nearest_nhwc,
    pixel_shuffle_nhwc,
    pixel_unshuffle_nhwc,
)


def test_pixel_shuffle_unshuffle_roundtrip():
    x = mx.arange(1 * 8 * 8 * 3, dtype=mx.float32).reshape((1, 8, 8, 3))
    y = pixel_unshuffle_nhwc(x, 2)
    assert y.shape == (1, 4, 4, 12)
    back = pixel_shuffle_nhwc(y, 2)
    assert back.shape == x.shape
    assert mx.max(mx.abs(back - x)).item() == 0.0


def test_interpolate_nearest():
    x = mx.arange(1 * 2 * 2 * 1, dtype=mx.float32).reshape((1, 2, 2, 1))
    y = interpolate_nearest_nhwc(x, 2)
    assert y.shape == (1, 4, 4, 1)


@pytest.mark.parametrize("scale,block", [(4, 23), (4, 6), (2, 23)])
def test_rrdbnet_forward_shape(scale, block):
    m = RRDBNet(3, 3, scale=scale, num_block=block)
    x = mx.zeros((1, 16, 16, 3))
    y = m(x)
    assert y.shape == (1, 16 * scale, 16 * scale, 3)


@pytest.mark.parametrize("num_conv", [16, 32])
def test_srvgg_forward_shape(num_conv):
    m = SRVGGNetCompact(3, 3, num_conv=num_conv, upscale=4)
    x = mx.zeros((1, 16, 16, 3))
    y = m(x)
    assert y.shape == (1, 64, 64, 3)


def test_variant_registry_complete():
    assert len(VARIANTS) == 5
    assert {v.arch for v in VARIANTS.values()} == {"RRDBNet", "SRVGGNetCompact"}

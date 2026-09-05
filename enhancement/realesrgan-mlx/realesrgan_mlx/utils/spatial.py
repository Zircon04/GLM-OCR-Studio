"""NHWC spatial ops used by both architectures.

These mirror the verified helpers from the Forge ecosystem
(``ForgeTraining/Python/models/efrlfn_mlx.py``) — in particular the pixel-shuffle
channel ordering, which is mlx-porting pitfall #7 ("the recurring one"): NHWC
pixel-shuffle MUST split channels as ``(C, r, r)`` to match ``torch.nn.PixelShuffle``.
The ``(r, r, C)`` ordering produces correct *shapes* but wrong *content* — the
divergence first shows up at the upsampler as max_abs ~2.27 in parity tests.
"""

from __future__ import annotations

import mlx.core as mx


def pixel_shuffle_nhwc(x: mx.array, r: int) -> mx.array:
    """``[N, H, W, C*r^2] -> [N, H*r, W*r, C]`` matching torch.nn.PixelShuffle.

    Output channel ``c`` at sub-pixel ``(i, j)`` reads input channel
    ``c*r*r + i*r + j`` — the verified ``(C, r, r)`` ordering.
    """
    n, h, w, c_in = x.shape
    if c_in % (r * r) != 0:
        raise ValueError(f"pixel_shuffle: channels {c_in} not divisible by r*r={r * r}")
    c = c_in // (r * r)
    x = x.reshape((n, h, w, c, r, r))
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))  # (N, H, r_i, W, r_j, C)
    return x.reshape((n, h * r, w * r, c))


def pixel_unshuffle_nhwc(x: mx.array, r: int) -> mx.array:
    """``[N, H, W, C] -> [N, H/r, W/r, C*r^2]`` — inverse of pixel_shuffle.

    Matches ``basicsr.archs.arch_util.pixel_unshuffle`` (NCHW) under NHWC.
    Channel grouping is the exact inverse of pixel_shuffle: new channel
    ``c*r*r + i*r + j`` gathers spatial sub-position ``(i, j)``.
    """
    n, h, w, c = x.shape
    if h % r != 0 or w % r != 0:
        raise ValueError(f"pixel_unshuffle: H={h}, W={w} not divisible by r={r}")
    oh, ow = h // r, w // r
    x = x.reshape((n, oh, r, ow, r, c))      # (N, H/r, r_i, W/r, r_j, C)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))  # (N, H/r, W/r, C, r_i, r_j)
    return x.reshape((n, oh, ow, c * r * r))


def interpolate_nearest_nhwc(x: mx.array, scale: int) -> mx.array:
    """Nearest-neighbour upsample by an integer factor, NHWC.

    Equivalent to ``F.interpolate(x, scale_factor=scale, mode='nearest')``.
    """
    n, h, w, c = x.shape
    x = x.reshape((n, h, 1, w, 1, c))
    x = mx.broadcast_to(x, (n, h, scale, w, scale, c))
    return x.reshape((n, h * scale, w * scale, c))

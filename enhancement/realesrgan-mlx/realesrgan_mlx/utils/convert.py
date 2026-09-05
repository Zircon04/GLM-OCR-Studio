"""PyTorch state_dict -> MLX weight conversion.

Used by both the parity tests (P2/P3) and the publish converter (P4). Operates on
plain numpy so it never imports torch.

The only layout change is conv weight transpose: PyTorch ``(O, I, kH, kW)`` ->
MLX ``(O, kH, kW, I)``. Everything else (bias, per-channel PReLU weight) is 1-D
and carries over unchanged — in NHWC the channel axis is last, so a ``(C,)``
PReLU weight broadcasts directly.

Key names already match the MLX module tree 1:1 (upstream ``body.N...`` ==
our list-attribute flatten), so no renaming is needed.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def convert_state_dict(sd: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """Map a numpy state_dict to MLX arrays, transposing 4-D conv weights.

    Heuristic: every 4-D tensor is a conv weight (the only 4-D params in
    RRDBNet / SRVGGNetCompact). 1-D tensors (bias, PReLU weight) pass through.
    """
    out: dict[str, mx.array] = {}
    for k, v in sd.items():
        v = np.asarray(v)
        if v.ndim == 4:  # (O, I, kH, kW) -> (O, kH, kW, I)
            v = np.transpose(v, (0, 2, 3, 1))
        out[k] = mx.array(v)
    return out

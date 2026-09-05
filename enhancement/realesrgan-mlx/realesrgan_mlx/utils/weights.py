"""Weight loading — local dist / env override / HF auto-download.

Entry point ``build_model(variant)`` returns a parity-ready MLX module with
weights loaded. Resolution order for a variant's weight dir:
  1. explicit ``weights_dir`` arg
  2. ``$REALESRGAN_MLX_WEIGHTS_DIR/<hf_name>``
  3. local ``dist/<hf_name>``  (produced by scripts/convert_to_mlx.py)
  4. HF hub ``mlx-community/<hf_name>``  (downloaded + cached)

Each dir holds ``model.safetensors`` (+ optional ``model_wdn.safetensors`` for
realesr-general-x4v3) and ``config.json``.
"""

from __future__ import annotations

import os

import mlx.core as mx

from ..archs import RRDBNet, SRVGGNetCompact
from ..config import VARIANTS, VariantConfig

WEIGHTS_DIR_ENV = "REALESRGAN_MLX_WEIGHTS_DIR"
HF_ORG = "mlx-community"


def resolve_repo_id(variant: str) -> str:
    return f"{HF_ORG}/{VARIANTS[variant].hf_name}"


def _build_arch(cfg: VariantConfig):
    if cfg.arch == "RRDBNet":
        return RRDBNet(3, 3, scale=cfg.netscale, num_feat=cfg.num_feat,
                       num_block=cfg.num_block, num_grow_ch=cfg.num_grow_ch)
    return SRVGGNetCompact(3, 3, num_feat=cfg.num_feat, num_conv=cfg.num_conv,
                           upscale=cfg.netscale, act_type=cfg.act_type)


def _resolve_dir(cfg: VariantConfig, weights_dir: str | None) -> str:
    if weights_dir:
        return weights_dir
    env = os.environ.get(WEIGHTS_DIR_ENV)
    if env and os.path.isdir(os.path.join(env, cfg.hf_name)):
        return os.path.join(env, cfg.hf_name)
    local = os.path.join("dist", cfg.hf_name)
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=f"{HF_ORG}/{cfg.hf_name}")


def _dni_blend(general: dict, wdn: dict, denoise_strength: float) -> dict:
    """net = s*general + (1-s)*wdn  (s=1 -> pure general; s=0 -> max denoise)."""
    s = float(denoise_strength)
    return {k: general[k] * s + wdn[k] * (1.0 - s) for k in general}


def build_model(
    variant: str = "RealESRGAN_x4plus",
    weights_dir: str | None = None,
    denoise_strength: float = 1.0,
):
    """Construct the arch for ``variant`` and load its converted MLX weights."""
    cfg = VARIANTS[variant]
    wdir = _resolve_dir(cfg, weights_dir)

    weights = dict(mx.load(os.path.join(wdir, "model.safetensors")).items())

    if cfg.wdn_url and denoise_strength < 1.0:
        wdn_path = os.path.join(wdir, "model_wdn.safetensors")
        if not os.path.exists(wdn_path):
            raise FileNotFoundError(
                f"denoise_strength<1 needs model_wdn.safetensors in {wdir}")
        wdn = dict(mx.load(wdn_path).items())
        weights = _dni_blend(weights, wdn, denoise_strength)

    model = _build_arch(cfg)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model

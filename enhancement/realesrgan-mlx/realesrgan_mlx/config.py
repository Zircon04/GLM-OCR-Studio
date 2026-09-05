"""Variant registry — the pinned config truth for every shipped Real-ESRGAN model.

Source of truth: upstream ``inference_realesrgan.py`` (xinntao/Real-ESRGAN, BSD-3),
cross-checked against ``basicsr/archs/{rrdbnet,srvgg}_arch.py``. Defaults here MUST
match the *trained* checkpoint config, never a framework default (mlx-porting
CONFIRM gate #3). Pin: xinntao/Real-ESRGAN @ master (v0.3.0 release weights).

Two architectures only — every variant is just a config over one of them:
  * RRDBNet           — heavy, photo/anime stills
  * SRVGGNetCompact   — compact, video / realtime

``netscale`` is the network's intrinsic scale (what the weights produce). The
pipeline may resize to a different ``--outscale`` afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantConfig:
    name: str
    arch: str  # "RRDBNet" | "SRVGGNetCompact"
    netscale: int
    # RRDBNet fields
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    # SRVGGNetCompact fields
    num_conv: int = 16
    act_type: str = "prelu"
    # provenance
    pth_url: str = ""
    # realesr-general only: the paired denoise-strength sibling (dni blend)
    wdn_url: str = ""
    # published HF repo name (under mlx-community/)
    hf_name: str = ""


_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download"

VARIANTS: dict[str, VariantConfig] = {
    # ---- RRDBNet family -------------------------------------------------
    "RealESRGAN_x4plus": VariantConfig(
        name="RealESRGAN_x4plus", arch="RRDBNet", netscale=4,
        num_feat=64, num_block=23, num_grow_ch=32,
        pth_url=f"{_REL}/v0.1.0/RealESRGAN_x4plus.pth",
        hf_name="Real-ESRGAN-x4plus",
    ),
    "RealESRGAN_x4plus_anime_6B": VariantConfig(
        name="RealESRGAN_x4plus_anime_6B", arch="RRDBNet", netscale=4,
        num_feat=64, num_block=6, num_grow_ch=32,
        pth_url=f"{_REL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        hf_name="Real-ESRGAN-x4plus-anime-6B",
    ),
    "RealESRGAN_x2plus": VariantConfig(
        name="RealESRGAN_x2plus", arch="RRDBNet", netscale=2,
        num_feat=64, num_block=23, num_grow_ch=32,
        pth_url=f"{_REL}/v0.2.1/RealESRGAN_x2plus.pth",
        hf_name="Real-ESRGAN-x2plus",
    ),
    # ---- SRVGGNetCompact family ----------------------------------------
    "realesr-animevideov3": VariantConfig(
        name="realesr-animevideov3", arch="SRVGGNetCompact", netscale=4,
        num_feat=64, num_conv=16, act_type="prelu",
        pth_url=f"{_REL}/v0.2.5.0/realesr-animevideov3.pth",
        hf_name="Real-ESRGAN-animevideov3",
    ),
    "realesr-general-x4v3": VariantConfig(
        name="realesr-general-x4v3", arch="SRVGGNetCompact", netscale=4,
        num_feat=64, num_conv=32, act_type="prelu",
        pth_url=f"{_REL}/v0.2.5.0/realesr-general-x4v3.pth",
        # denoise sibling — blended into the general weights via dni_weight
        wdn_url=f"{_REL}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        hf_name="Real-ESRGAN-general-x4v3",
    ),
}

DEFAULT_VARIANT = "RealESRGAN_x4plus"

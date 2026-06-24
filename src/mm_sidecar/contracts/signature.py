from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProcessorConfig:
    model_name: str
    revision: str
    processor_name: str
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 28 * 28 * 1280
    do_resize: bool = True
    resize_strategy: str = "smart_resize"
    crop_strategy: str = "none"
    pad_strategy: str = "repeat_last_frame"
    resample: str = "bicubic"
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    do_normalize: bool = True
    image_mean: tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    image_std: tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )
    do_convert_rgb: bool = True
    data_format: str = "channels_first"
    output_dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class ProcessorSignature:
    value: str

    @classmethod
    def from_config(cls, config: ProcessorConfig) -> "ProcessorSignature":
        cfg = asdict(config)
        mean = ",".join(repr(value) for value in cfg["image_mean"])
        std = ",".join(repr(value) for value in cfg["image_std"])
        return cls(
            value=(
                f"model={cfg['model_name']}"
                f"|rev={cfg['revision']}"
                f"|processor={cfg['processor_name']}"
                f"|patch={cfg['patch_size']}"
                f"|merge={cfg['merge_size']}"
                f"|temporal={cfg['temporal_patch_size']}"
                f"|min_pixels={cfg['min_pixels']}"
                f"|max_pixels={cfg['max_pixels']}"
                f"|do_resize={int(cfg['do_resize'])}"
                f"|resize_strategy={cfg['resize_strategy']}"
                f"|crop_strategy={cfg['crop_strategy']}"
                f"|pad_strategy={cfg['pad_strategy']}"
                f"|resample={cfg['resample']}"
                f"|do_rescale={int(cfg['do_rescale'])}"
                f"|rescale_factor={repr(cfg['rescale_factor'])}"
                f"|do_normalize={int(cfg['do_normalize'])}"
                f"|image_mean={mean}"
                f"|image_std={std}"
                f"|do_convert_rgb={int(cfg['do_convert_rgb'])}"
                f"|data_format={cfg['data_format']}"
                f"|output_dtype={cfg['output_dtype']}"
            )
        )

    @classmethod
    def parse(cls, value: str) -> dict[str, str]:
        parts: dict[str, str] = {}
        for chunk in value.split("|"):
            if "=" not in chunk:
                continue
            key, raw = chunk.split("=", 1)
            parts[key] = raw
        return parts

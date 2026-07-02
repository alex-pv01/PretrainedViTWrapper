from torch import Device
from timm import create_model
from timm.models import VisionTransformer

from dataclasses import dataclass
from typing import Literal
from functools import partial

from .types import TimmArchitectures, ARCHS_TO_IDS, ViTBackbone
from .modifications import ModificationFn, change_stride, replace_pos_embed

Dtype = Literal["float32", "float16"]


@dataclass
class WrapperConfig:
    model_name: str
    model_arch: TimmArchitectures
    pretrained: bool = True
    checkpoint_path: str | None = None

    device: Device = "cuda"
    dtype: Dtype = "float32"

    stride: tuple[int, int] | None = None
    remove_pos_embed: bool = False
    add_flash_attn: bool = False

    modifications: list[ModificationFn] | None = None


def _populate_modifications(
    prev_modifications: list[ModificationFn] | None, cfg: WrapperConfig
) -> list[ModificationFn]:
    if prev_modifications is None:
        prev_modifications = []

    if cfg.stride:
        prev_modifications.append(partial(change_stride, new_stride=cfg.stride))

    if cfg.remove_pos_embed:
        prev_modifications.append(partial(replace_pos_embed, new_pos_embed=None, requires_grad=False))

    return prev_modifications


def create_timm_backbone(cfg: WrapperConfig) -> ViTBackbone:
    model_identifier = ARCHS_TO_IDS[cfg.model_arch]
    vit = create_model(model_identifier, pretrained=cfg.pretrained, checkpoint_path=cfg.checkpoint_path)

    assert isinstance(vit, VisionTransformer)

    if cfg.modifications is not None:
        for mod in cfg.modifications:
            vit = mod(vit)

    return vit


class WrapperFactory:
    _registry: dict[str, WrapperConfig] = {}

    @classmethod
    def register(cls, name: str, cfg: WrapperConfig) -> None:
        cls._registry[name] = cfg

    @classmethod
    def build(cls, name: str) -> ViTBackbone:
        cfg = cls._registry[name]
        cfg.modifications = _populate_modifications(cfg.modifications, cfg)
        return create_timm_backbone(cfg)

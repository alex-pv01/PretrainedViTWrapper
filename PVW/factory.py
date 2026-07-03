import torch
from timm import create_model

from dataclasses import dataclass, field
from typing import Literal, Type, Callable, Any, Dict
from functools import partial

from .types import (
    Architectures,
    ARCHS_TO_TIMM_IDS,
    ViTBackbone,
    ImageTransform,
    ModificationFn,
    satisfies_protocol,
    ARCHS_TO_LOCAL_IDS,
    TimmViT,
    Dv3ViT,
)
from .modifications import change_stride, replace_pos_embed
from .utils import closest_resize, logger
from .wrapper import PretrainedViTWrapper

Dtype = Literal["float32", "float16"]


@dataclass
class BackboneConfig:
    backbone_type: str = "timm"  # e.g., "timm", "dinov3", or a custom registered type
    model_arch: Architectures = "dinov2_s"
    pretrained: bool = True
    checkpoint_path: str | None = None
    model_conf_path: str | None = None

    stride: tuple[int, int] | None = None
    remove_pos_embed: bool = False
    add_flash_attn: bool = False

    dynamic_img_size: bool = True
    dynamic_img_pad: bool = False

    modifications: list[ModificationFn] = field(default_factory=list)
    dtype: torch.dtype = torch.float32


@dataclass
class WrapperConfig:
    backbone_cfg: BackboneConfig
    wrapper_class: Type[PretrainedViTWrapper] = PretrainedViTWrapper
    wrapper_kwargs: Dict[str, Any] = field(default_factory=dict)
    transform: ImageTransform = closest_resize
    dtype: torch.dtype = torch.float32


def _populate_modifications(
    prev_modifications: list[ModificationFn] | None, cfg: BackboneConfig
) -> list[ModificationFn]:
    if prev_modifications is None:
        prev_modifications = []
    else:
        # copy list to avoid mutating class default
        prev_modifications = list(prev_modifications)

    if cfg.stride:
        prev_modifications.append(partial(change_stride, new_stride=cfg.stride))

    if cfg.remove_pos_embed:
        prev_modifications.append(partial(replace_pos_embed, new_pos_embed=None, requires_grad=False))

    return prev_modifications


def create_timm_backbone(cfg: BackboneConfig) -> TimmViT:
    model_identifier = ARCHS_TO_TIMM_IDS[cfg.model_arch]

    if "sam" in model_identifier or "conv" in model_identifier:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
            checkpoint_path=cfg.checkpoint_path,
        )
    else:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
            checkpoint_path=cfg.checkpoint_path,
            dynamic_img_size=cfg.dynamic_img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
            pretrained_strict=False,
        )
    assert satisfies_protocol(vit, TimmViT)

    mods = _populate_modifications(cfg.modifications, cfg)
    for mod in mods:
        vit = mod(vit)

    vit = vit.to(dtype=cfg.dtype)

    return vit


def create_local_backbone(cfg: BackboneConfig) -> Dv3ViT:
    model_identifier = ARCHS_TO_LOCAL_IDS[cfg.model_arch]

    kwargs = {}
    if cfg.checkpoint_path is not None:
        kwargs["weights"] = cfg.checkpoint_path
        kwargs["pretrained"] = True
    else:
        kwargs["pretrained"] = False

    vit = torch.hub.load(
        cfg.model_conf_path,
        model_identifier,
        source="local",
        **kwargs,
    )
    assert satisfies_protocol(vit, Dv3ViT)

    mods = _populate_modifications(cfg.modifications, cfg)
    for mod in mods:
        vit = mod(vit)

    vit = vit.to(dtype=cfg.dtype)

    return vit


class BackboneRegistry:
    _registry: Dict[str, Callable[[BackboneConfig], ViTBackbone]] = {}

    @classmethod
    def register(cls, backbone_type: str, builder_fn: Callable[[BackboneConfig], ViTBackbone]) -> None:
        cls._registry[backbone_type] = builder_fn

    @classmethod
    def build(cls, cfg: BackboneConfig) -> ViTBackbone:
        logger.info("Building backbone with config: %s", cfg)
        builder = cls._registry.get(cfg.backbone_type)
        if not builder:
            raise ValueError(f"Backbone type '{cfg.backbone_type}' not registered in BackboneRegistry.")
        return builder(cfg)


# Register the default timm backbone builder
BackboneRegistry.register("timm", create_timm_backbone)
BackboneRegistry.register("torch_hub", create_local_backbone)


class WrapperRegistry:
    _registry: Dict[str, WrapperConfig] = {}

    @classmethod
    def register(cls, name: str, cfg: WrapperConfig) -> None:
        cls._registry[name] = cfg

    @classmethod
    def build(cls, name: str, device: torch.device | str = "cuda", **override_kwargs) -> PretrainedViTWrapper:
        logger.info("Building wrapper '%s' on device %s", name, device)
        # Auto-register standard architectures if they are not in the registry but are valid timm architectures
        if name not in cls._registry:
            if name in ARCHS_TO_TIMM_IDS:
                cls.register(
                    name,
                    WrapperConfig(
                        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch=name)  # type: ignore
                    ),
                )
            else:
                raise KeyError(f"Model variant '{name}' not found in registry and is not a default TimmArchitecture.")

        cfg = cls._registry[name]

        # Build the backbone using BackboneRegistry
        vit = BackboneRegistry.build(cfg.backbone_cfg)

        # Prepare wrapper parameters
        wrapper_cls = cfg.wrapper_class
        kwargs = {
            **cfg.wrapper_kwargs,
            "device": device,
            **override_kwargs,
        }
        import inspect
        sig = inspect.signature(wrapper_cls.__init__)
        if "dtype" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            kwargs["dtype"] = cfg.dtype

        transform = kwargs.pop("transform", cfg.transform)

        # Construct wrapper
        wrapper = wrapper_cls(vit=vit, **kwargs)
        wrapper.name = name
        wrapper.arch_name = cfg.backbone_cfg.model_arch
        wrapper.transform = transform
        return wrapper

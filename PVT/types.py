from torch import Tensor
from torchvision.transforms import Compose
from torch.nn import Parameter, Module, Sequential, Linear, Dropout, Conv2d

from typing import Any, Union, Protocol, Literal, ParamSpec, Concatenate, Callable


P = ParamSpec("P")
ImageTransform = Callable[Concatenate[int, int, P], Compose]


"""
These are interfaces defining relevant ViT properties we want for type safety in our wrapper & modification code.
Should be respected by `timm` and DINO style models.  
"""


class HasAttn(Protocol):
    num_heads: int
    head_dim: int
    attn_dim: int
    scale: float
    fused_attn: bool

    qkv: Linear
    q_norm: Module
    k_norm: Module
    attn_drop: Dropout
    norm: Module
    proj: Linear
    proj_drop: Dropout

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...


class HasBlock(Protocol):
    norm1: Module
    attn: HasAttn
    ls1: Module
    drop_path1: Module

    norm2: Module
    mlp: Module
    ls2: Module
    drop_path2: Module

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...


class HasPatchEmbed(Protocol):
    patch_size: tuple[int, int]
    proj: Conv2d

    def dynamic_feat_size(self, img_size: tuple[int, int]) -> tuple[int, int]: ...


class ViTBackbone(Protocol):
    # Model constants
    num_features: int

    num_prefix_tokens: int

    # Layers
    pos_embed: Parameter | None

    patch_embed: HasPatchEmbed

    blocks: Sequential | Module

    # Methods
    def forward_features(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, list[int], tuple[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        attn_mask: Tensor | None = None,
    ) -> list[Tensor]: ...

    def __call__(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor: ...


J = ParamSpec("J")
ModificationFn = Callable[Concatenate[ViTBackbone, J], ViTBackbone]

TimmArchitectures = Literal[
    "dino_s",
    "dino_b",
    "dinov2_s",
    "dinov2_b",
    "dinov3_s",
    "dinov3_s+",
    "dinov3_b",
    "EUPE_t",
    "EUPE_s",
    "mae_b",
    "in1k_b",
    "deit_s",
    "sam_b",
    "clip_b",
    "eva02_b",
]
ARCHS_TO_IDS: dict[TimmArchitectures, str] = {
    "dino_s": "vit_small_patch16_224.dino",
    "dino_b": "vit_base_patch16_224.dino",
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov3_s": "vit_small_patch16_dinov3_qkvb.lvd1689m",
    "dinov3_s+": "vit_small_plus_patch16_dinov3_qkvb.lvd1689m",
    "dinov3_b": "vit_base_patch16_dinov3_qkvb.lvd1689m",
    "EUPE_t": "hf_hub:timm/vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m",
    "EUPE_s": "hf_hub:timm/vit_small_patch16_dinov3_qkvb.eupe_lvd1689m",
    "mae_b": "vit_base_patch16_224.mae",
    "in1k_b": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "deit_s": "deit3_small_patch16_224.fb_in1k",
    "sam_b": "samvit_base_patch16.sa1b",
    "clip_b": "vit_base_patch16_clip_224.openai",
    "eva02_b": "eva02_base_patch14_224.mim_in22k",
}

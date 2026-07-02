from torch import Tensor
from torch.nn import Parameter, Sequential
from types import MethodType

from .types import ViTBackbone, HasPatchEmbed, HasBlock


def change_stride(vit: ViTBackbone, new_stride: tuple[int, int]) -> ViTBackbone:
    """Change the stride of the patch embedding layer of a ViT backbone.

    Args:
        vit (ViTBackbone): The ViT backbone to modify.
        new_stride (tuple[int, int]): The new stride to set for the patch embedding layer.
    """
    old_sx, old_sy = vit.patch_embed.proj.stride
    new_sx, new_sy = new_stride
    if (old_sx, old_sy) == (new_sx, new_sy):
        return vit

    vit.patch_embed.proj.stride = new_stride

    def dynamic_feat_size(self: HasPatchEmbed, img_size: tuple[int, int]) -> tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
            img_size[1] - self.patch_size[1]
        ) // self.proj.stride[1] + 1

    vit.patch_embed.dynamic_feat_size = MethodType(dynamic_feat_size, vit.patch_embed)  # type: ignore

    return vit


def replace_pos_embed(vit: ViTBackbone, new_pos_embed: Tensor | None, requires_grad: bool) -> ViTBackbone:
    """Replace the positional embedding of a ViT backbone.

    Args:
        vit (ViTBackbone): The ViT backbone to modify.
        new_pos_embed (Tensor): The new positional embedding to set.
    """
    # TODO: consider dinov3 and .rope_embed attr
    if isinstance(new_pos_embed, Tensor):
        vit.pos_embed = Parameter(new_pos_embed, requires_grad=requires_grad)
    else:
        vit.pos_embed = None
    return vit


def replace_blocks(vit: ViTBackbone, new_blocks: list[HasBlock]) -> ViTBackbone:
    """Replace the blocks of a ViT backbone.

    Args:
        vit (ViTBackbone): The ViT backbone to modify.
        new_blocks (Tensor): The new blocks to set.
    """
    vit.blocks = Sequential(*new_blocks)  # type: ignore

    return vit

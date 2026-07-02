from torch import Tensor
from torch.nn import Module

from .types import ViTBackbone, TimmArchitectures


class PretrainedViTWrapper(Module):
    def __init__(self, vit: ViTBackbone, device):
        super().__init__()
        self.vit: ViTBackbone = vit

        self.name: str = ""
        self.arch_name: TimmArchitectures

        self.stride = self.vit.patch_embed.proj.stride
        self.patch_size = self.vit.patch_embed.patch_size

        self.embed_dim = self.vit.num_features
        self.num_blocks = len(self.vit.blocks)

    def _get_tokens_per_dim(self, x: Tensor) -> tuple[int, int]:
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p[0]) // s[0] + 1, (w - p[1]) // s[1] + 1
        return n_patch_h, n_patch_w

    def forward_intermediates(
        self, x: Tensor, n: int | list[int] = 1, make_2D: bool = True, norm: bool = True
    ) -> list[Tensor]:
        b, _, _, _ = x.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x)

        feats = self.vit.get_intermediate_layers(x, n=n, reshape=False, norm=norm)
        feats = [f.permute(0, 2, 1) for f in feats]

        if make_2D:
            feats = [f.view(b, self.embed_dim, n_patch_h, n_patch_w) for f in feats]
        return feats

    def forward_features(self, x: Tensor, make_2D: bool = True) -> Tensor:
        b, _, _, _ = x.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x)

        feats = self.vit.forward_features(x)
        feats = feats.permute(0, 2, 1)

        if make_2D:
            feats = feats.view(b, self.embed_dim, n_patch_h, n_patch_w)
        return feats

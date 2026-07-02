import torch
from torch import Tensor
from torch.nn import Module

from .types import ViTBackbone, TimmArchitectures, ImageTransform, PILImage


class PretrainedViTWrapper(Module):
    def __init__(
        self,
        vit: ViTBackbone,
        device,
        name: str = "",
        arch_name: TimmArchitectures | None = None,
        transform: ImageTransform | None = None,
    ):
        super().__init__()
        self.vit: ViTBackbone = vit

        self.name: str = name
        self.arch_name: TimmArchitectures | None = arch_name
        self.device = device
        self.transform = transform

        self.stride = self.vit.patch_embed.proj.stride
        self.patch_size = self.vit.patch_embed.patch_size

        self.embed_dim = self.vit.num_features
        self.num_prefix_tokens = self.vit.num_prefix_tokens
        self.num_blocks = len(self.vit.blocks)

        self.to(device)

    def preprocess_input(self, x: Tensor | PILImage | list[PILImage]) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(self.device)

        if self.transform is None:
            raise ValueError("An image transform must be provided to process PIL Images.")

        from PIL import Image

        if isinstance(x, Image.Image):
            return self.transform(x).unsqueeze(0).to(self.device)
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], Image.Image):
            return torch.stack([self.transform(img) for img in x]).to(self.device)

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _get_tokens_per_dim(self, x: Tensor) -> tuple[int, int]:
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p[0]) // s[0] + 1, (w - p[1]) // s[1] + 1
        return n_patch_h, n_patch_w

    def forward(self, x: Tensor | PILImage | list[PILImage]) -> Tensor:
        x_tensor = self.preprocess_input(x)
        return self.vit(x_tensor)

    def forward_intermediates(
        self,
        x: Tensor | PILImage | list[PILImage],
        n: int | list[int] = 1,
        make_2D: bool = True,
        norm: bool = True,
    ) -> list[Tensor]:
        x_tensor = self.preprocess_input(x)
        b, _, _, _ = x_tensor.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x_tensor)

        feats = self.vit.get_intermediate_layers(x_tensor, n=n, reshape=False, norm=norm)
        feats = [f.permute(0, 2, 1) for f in feats]

        if make_2D:
            feats = [f.view(b, self.embed_dim, n_patch_h, n_patch_w) for f in feats]
        return feats

    def forward_features(
        self,
        x: Tensor | PILImage | list[PILImage],
        make_2D: bool = True,
    ) -> Tensor:
        x_tensor = self.preprocess_input(x)
        b, _, _, _ = x_tensor.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x_tensor)

        feats = self.vit.forward_features(x_tensor)
        feats = feats[:, self.num_prefix :]
        feats = feats.permute(0, 2, 1)

        if make_2D:
            feats = feats.view(b, self.embed_dim, n_patch_h, n_patch_w)
        return feats

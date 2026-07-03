import torch
from torch import Tensor
from torch.nn import Module
from PIL import Image

from .utils import closest_resize
from .types import ViTBackbone, Architectures, ImageTransform, TimmViT, satisfies_protocol, Dv3ViT


class PretrainedViTWrapper(Module):
    def __init__(
        self,
        vit: ViTBackbone,
        device,
        name: str = "",
        arch_name: Architectures | None = None,
        transform: ImageTransform = closest_resize,
    ):
        super().__init__()
        self.vit: ViTBackbone = vit

        self.name: str = name
        self.arch_name: Architectures | None = arch_name
        self.device = device
        self.transform = transform

        self.stride = self.vit.patch_embed.proj.stride
        self.patch_size = self.vit.patch_embed.patch_size

        self.embed_dim = self.vit.num_features
        self.num_blocks = len(self.vit.blocks)  # type: ignore

        self.to(device)

    @property
    def num_prefix_tokens(self) -> int:
        if satisfies_protocol(self.vit, TimmViT):
            return self.vit.num_prefix_tokens
        elif satisfies_protocol(self.vit, Dv3ViT):
            return self.vit.n_storage_tokens
        else:
            return 0

    def preprocess_input(self, x: Tensor | Image.Image | list[Image.Image]) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(self.device)

        p_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size

        if isinstance(x, Image.Image):
            w, h = x.size
            resolved_transform = self.transform(h, w, patch_size=p_size)
            img_tensor: Tensor = resolved_transform(x).unsqueeze(0).to(self.device)  # type: ignore
            return img_tensor.to(self.device)
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], Image.Image):
            w, h = x[0].size
            resolved_transform = self.transform(h, w, patch_size=p_size)
            return torch.stack([resolved_transform(img) for img in x]).to(self.device)  # type: ignore

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _get_tokens_per_dim(self, x: Tensor) -> tuple[int, int]:
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p[0]) // s[0] + 1, (w - p[1]) // s[1] + 1
        return n_patch_h, n_patch_w

    def forward(self, x: Tensor | Image.Image | list[Image.Image]) -> Tensor:
        x_tensor = self.preprocess_input(x)
        return self.vit(x_tensor)

    def forward_intermediates(
        self,
        x: Tensor | Image.Image | list[Image.Image],
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
        x: Tensor | Image.Image | list[Image.Image],
        make_2D: bool = True,
    ) -> Tensor:
        x_tensor = self.preprocess_input(x)
        b, _, _, _ = x_tensor.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x_tensor)

        if satisfies_protocol(self.vit, TimmViT):
            feats = self.vit.forward_features(x_tensor)
        elif satisfies_protocol(self.vit, Dv3ViT):
            feats = self.vit.forward_features(x_tensor)["x_norm_patchtokens"]
        else:
            raise TypeError(f"Unsupported ViT type: {type(self.vit)}")

        feats = feats[:, self.num_prefix_tokens :]
        feats = feats.permute(0, 2, 1)

        if make_2D:
            feats = feats.view(b, self.embed_dim, n_patch_h, n_patch_w)
        return feats

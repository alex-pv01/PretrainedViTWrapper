import numpy as np
import torch
from torchvision import transforms

from typing import Literal


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if len(arr.shape) == 4:
        arr = arr[0]
    return arr


MU, SIGMA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INV_MU = (-MU[0] / SIGMA[0], -MU[1] / SIGMA[1], -MU[2] / SIGMA[2])
INV_SIGMA = (1 / SIGMA[0], 1 / SIGMA[1], 1 / SIGMA[2])

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MU, std=SIGMA),
    ]
)

unnormalize = transforms.Normalize(
    mean=INV_MU,
    std=INV_SIGMA,
)


def closest_crop(h: int, w: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose([transforms.CenterCrop((new_h, new_w)), to_norm_tensor])
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
            ]
        )
    return transform


def closest_resize(h: int, w: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose([transforms.Resize((new_h, new_w)), to_norm_tensor])
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((new_h, new_w)),
            ]
        )
    return transform


def _std_scale(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x - mean) / std


def _minmax_scale(x: torch.Tensor) -> torch.Tensor:
    min_val = x.min(dim=0, keepdim=True).values
    max_val = x.max(dim=0, keepdim=True).values
    return (x - min_val) / (max_val - min_val + 1e-8)


NormType = Literal["minmax", "std", None]


def do_pca(
    features: torch.Tensor,
    n_components: int = 3,
    pre_norm: NormType = "std",
    post_norm: NormType = "minmax",
) -> torch.Tensor:
    orig_shape = features.shape
    if len(orig_shape) == 4:
        B, C, H, W = orig_shape
        assert B == 1, "Only single batch is supported for PCA"
        x = features.squeeze(0).permute(1, 2, 0).reshape(-1, C)
    elif len(orig_shape) == 3:
        C, H, W = orig_shape
        x = features.permute(1, 2, 0).reshape(-1, C)
    elif len(orig_shape) == 2:
        x = features
    else:
        raise ValueError(f"Unsupported features shape: {orig_shape}")

    if pre_norm == "std":
        x = _std_scale(x)
    elif pre_norm == "minmax":
        x = _minmax_scale(x)

    _, _, V = torch.pca_lowrank(x, q=n_components)
    projected = x @ V[:, :n_components]

    if post_norm == "std":
        projected = _std_scale(projected)
    elif post_norm == "minmax":
        projected = _minmax_scale(projected)

    if len(orig_shape) in (3, 4):
        H, W = orig_shape[-2], orig_shape[-1]
        if len(orig_shape) == 3:
            projected = projected.reshape(H, W, n_components).permute(2, 0, 1)
        else:
            projected = projected.reshape(1, H, W, n_components).permute(0, 3, 1, 2)

    return projected

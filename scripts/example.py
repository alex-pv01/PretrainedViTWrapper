from PIL import Image
from torch import no_grad

from PVW import BackboneConfig, BackboneRegistry, PretrainedViTWrapper
from PVW.utils import to_numpy, do_pca

import matplotlib.pyplot as plt

from timm.models import _registry

from timm.models import list_models

import timm

# print(timm.__version__)

# for m in list_models("*dinov3*"):
#     print(m)

# entry = _registry.model_entrypoint("vit_small_plus_patch16_dinov3_qkvb")
# print(entry)
# print(entry.__module__)

# for m in [
#     "vit_small_patch16_dinov3_qkvb",
#     "vit_small_plus_patch16_dinov3_qkvb",
#     "vit_tiny_patch16_dinov3_qkvb",
#     "vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m",
#     "vit_small_patch16_dinov3_qkvb.eupe_lvd1689m",
# ]:
#     fn = _registry.model_entrypoint(m)
#     print(m, "->", fn.__module__)

backbone_cfg = BackboneConfig(
    backbone_type="timm",
    model_arch="dinov2_s",
    # checkpoint_path="../dino-saw/trained_models/dinov3_vits_patch16_plus_reg4.pth",
)
backbone = BackboneRegistry.build(backbone_cfg)

pvt = PretrainedViTWrapper(backbone, device="cuda", name="dinov3_s", arch_name="dinov3_s")

img = Image.open("tests/data/plant_cells.jpg").convert("RGB")

_w, _h = img.size
img = img.resize((_w // 2, _h // 2), Image.Resampling.BICUBIC)

with no_grad():
    feats = pvt.forward_features(img, True)

feats_reduced = do_pca(feats, 9)
feats_reduced_np = to_numpy(feats_reduced)

plt.imsave("test.png", feats_reduced_np.transpose(1, 2, 0)[:, :, 0:3])

# print(feats_reduced.shape)

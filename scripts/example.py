from PIL import Image
from torch import no_grad

from PVT import BackboneConfig, BackboneRegistry, PretrainedViTWrapper
from PVT.utils import to_numpy, do_pca

import matplotlib.pyplot as plt

backbone_cfg = BackboneConfig(
    backbone_type="timm",
    model_arch="dinov3_s+",
)
backbone = BackboneRegistry.build(backbone_cfg)

pvt = PretrainedViTWrapper(backbone, device="cuda", name="dinov3_s", arch_name="dinov3_s")

img = Image.open("tests/data/bimodal.png").convert("RGB")

with no_grad():
    feats = pvt.forward_features(img, True)

feats_reduced = do_pca(feats, 15)
feats_reduced_np = to_numpy(feats_reduced)

plt.imsave("test.png", feats_reduced_np.transpose(1, 2, 0)[:, :, 12:15])

# print(feats_reduced.shape)

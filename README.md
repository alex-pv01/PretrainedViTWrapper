# PretrainedViTWrapper
Wrapper module for pretrained ViTs (DINO etc). Includes timm support.

Usage:
```python
from PIL import Image
from torch import no_grad

from PVW import BackboneConfig, BackboneRegistry, PretrainedViTWrapper

backbone_cfg = BackboneConfig(
    backbone_type="torch_hub",
    model_arch="dinov3_s+",
    model_conf_path="models/dinov3",
    checkpoint_path="models/checkpoints/dinov3_vits_patch16_plus_reg4.pth",
)
backbone = BackboneRegistry.build(backbone_cfg)
pvt = PretrainedViTWrapper(backbone, device="cuda", name="dinov3_s", arch_name="dinov3_s")

img = Image.open("tests/data/ni_superalloy.png").convert("RGB")
with no_grad():
    feats = pvt.forward_features(img, True)
```

## Installation

```bash
uv sync
```

To get local models, run the following:
```bash
mkdir models/
cd models
git clone https://github.com/facebookresearch/EUPE
git clone https://github.com/facebookresearch/dinov3/
```
from huggingface_hub import snapshot_download

snapshot_download(
    "timm/vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m",
    force_download=True,
)

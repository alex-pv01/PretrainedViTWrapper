import pytest
import torch
from torch import nn
import timm

from PVW import BackboneRegistry, BackboneConfig
from PVW.types import ViTBackbone, HasBlock, HasAttn, satisfies_protocol


def test_satisfies_protocol_non_vit():
    # 1. Test basic PyTorch modules that are not ViTs
    linear = nn.Linear(10, 10)
    conv = nn.Conv2d(3, 64, 3)
    basic_module = nn.Module()

    # None of these should satisfy the ViTBackbone protocol
    assert not satisfies_protocol(linear, ViTBackbone)
    assert not satisfies_protocol(conv, ViTBackbone)
    assert not satisfies_protocol(basic_module, ViTBackbone)


def test_satisfies_protocol_vit():
    # 2. Build a real ViT backbone using the library factory
    cfg = BackboneConfig(
        backbone_type="timm",
        model_arch="deit_s",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    # The created backbone should satisfy the ViTBackbone protocol
    assert satisfies_protocol(vit, ViTBackbone)


def test_satisfies_protocol_blocks_and_attn():
    # 3. Build a standard timm ViT to extract blocks and attention modules
    cfg = BackboneConfig(
        backbone_type="timm",
        model_arch="deit_s",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    # Extract block and attention modules
    block = vit.blocks[0]
    attn = block.attn

    # Test the structural protocols
    assert satisfies_protocol(block, HasBlock)
    assert not satisfies_protocol(block, HasAttn)
    assert satisfies_protocol(attn, HasAttn)


def test_satisfies_protocol_invalid_protocol_cls():
    # 4. Test that passing a non-protocol class raises TypeError
    linear = nn.Linear(10, 10)
    with pytest.raises(TypeError):
        satisfies_protocol(linear, nn.Linear)  # type: ignore

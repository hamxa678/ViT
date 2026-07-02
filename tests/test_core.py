import torch
from ViT import ViT


def test_vit_output_shape():
    model = ViT(img_size=32, patch_size=8, num_classes=10, embed_dim=64, depth=2, num_heads=4)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
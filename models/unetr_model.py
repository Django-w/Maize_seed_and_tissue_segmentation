import torch
import torch.nn as nn
from typing import Tuple
from monai.networks.nets import UNETR


def create_unetr_model(
    in_channels: int = 1,
    out_channels: int = 4,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    proj_type: str = "conv",        # ? Ìæ´ú pos_embed
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
    spatial_dims: int = 3,
    qkv_bias: bool = False,
    save_attn: bool = False,
) -> nn.Module:
    """
    UNETR (MONAI current version)
    """
    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        proj_type=proj_type,         # new model parameters
        norm_name=norm_name,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
        qkv_bias=qkv_bias,
        save_attn=save_attn,
    )
    return model


if __name__ == "__main__":
    model = create_unetr_model(in_channels=1, out_channels=4, img_size=(96, 96, 96))
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("UNETR模型创建成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

"""
Attention UNet 3D 模型定义
使用MONAI的AttentionUnet实现
"""
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet


def create_attention_unet3d_model(
    in_channels: int = 1,
    out_channels: int = 4,
    channels: list = [16, 32, 64, 128, 256],
    strides: list = [2, 2, 2, 2],
    spatial_dims: int = 3,
) -> AttentionUnet:
    """
    创建Attention UNet 3D模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（类别数）
        channels: 编码器通道数列表
        strides: 下采样步长列表
        spatial_dims: 空间维度，3D为3
        
    Returns:
        Attention UNet 3D模型
        
    Note:
        AttentionUnet 不支持 num_res_units、norm、dropout 参数
    """
    model = AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
    )
    
    return model


if __name__ == "__main__":
    # 测试模型创建
    model = create_attention_unet3d_model(
        in_channels=1,
        out_channels=4,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
    )
    
    # 测试前向传播
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("Attention UNet 3D模型创建成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

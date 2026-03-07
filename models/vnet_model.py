"""
VNet 模型定义
使用MONAI的VNet实现
"""
import torch
import torch.nn as nn
from monai.networks.nets import VNet


def create_vnet_model(
    in_channels: int = 1,
    out_channels: int = 4,
    act: str = "PRELU",
    dropout_prob: float = 0.0,
    dropout_prob_down: float = None,
    dropout_prob_up: tuple = None,
    dropout_dim: int = 3,
    bias: bool = False,
    spatial_dims: int = 3,
) -> VNet:
    """
    创建VNet模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（类别数）
        act: 激活函数类型 ("PRELU", "RELU", "ELU") 或元组 (act_name, act_params)
        dropout_prob: Dropout率（如果设置，会同时用于down和up）
        dropout_prob_down: 下采样路径Dropout率
        dropout_prob_up: 上采样路径Dropout率，应为元组 (prob1, prob2)
        dropout_dim: Dropout维度
        bias: 是否使用偏置
        spatial_dims: 空间维度，3D为3
        
    Returns:
        VNet模型
    """
    # 处理dropout参数
    if dropout_prob_down is None:
        dropout_prob_down = dropout_prob
    if dropout_prob_up is None:
        dropout_prob_up = (dropout_prob, dropout_prob)
    elif isinstance(dropout_prob_up, (int, float)):
        # 如果是单个值，转换为元组
        dropout_prob_up = (dropout_prob_up, dropout_prob_up)
    
    kwargs = {
        'spatial_dims': spatial_dims,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'act': act,
        'dropout_prob_down': dropout_prob_down,
        'dropout_prob_up': dropout_prob_up,
        'dropout_dim': dropout_dim,
        'bias': bias,
    }
    
    model = VNet(**kwargs)
    
    return model


if __name__ == "__main__":
    # 测试模型创建
    model = create_vnet_model(
        in_channels=1,
        out_channels=4,
    )
    
    # 测试前向传播
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("VNet模型创建成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

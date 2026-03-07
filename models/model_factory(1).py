"""
模型工厂：根据配置创建不同的模型
"""
import torch
from models.swin_unetr_model import create_swin_unetr_model
from models.swin_unetr_moe_model import create_swin_unetr_moe_model
from models.unet3d_model import create_unet3d_model
from models.attention_unet3d_model import create_attention_unet3d_model
from models.vnet_model import create_vnet_model
from models.unetr_model import create_unetr_model
from models.nnUnet_model import create_nnunet_model
from models.swin_unetr_gasa_model import create_swin_unetr_gasa_model
from models.swin_unetr_dilated_model import create_swin_unetr_dilated_model


def create_model(model_config: dict):
    """
    根据配置创建模型
    
    Args:
        model_config: 模型配置字典，必须包含'name'字段
        
    Returns:
        创建的模型
        
    Raises:
        ValueError: 如果模型名称不支持
    """
    model_name = model_config.get('name', '').lower()
    
    if model_name == 'swin_unetr':
        return create_swin_unetr_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            feature_size=model_config.get('feature_size', 48),
            num_heads=tuple(model_config.get('num_heads', [3, 6, 12, 24])),
            depths=tuple(model_config.get('depths', [2, 2, 2, 2])),
            window_size=model_config.get('window_size', 7),
            patch_size=model_config.get('patch_size', 2),
            qkv_bias=model_config.get('qkv_bias', True),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            norm_name=model_config.get('norm_name', 'instance'),
            drop_rate=model_config.get('dropout_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            dropout_path_rate=model_config.get('dropout_path_rate', 0.0),
            use_checkpoint=model_config.get('use_checkpoint', False),
        )
    elif model_name =='swin_unetr_moe':
        return create_swin_unetr_moe_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            feature_size=model_config.get('feature_size', 48),
            num_heads=tuple(model_config.get('num_heads', [3, 6, 12, 24])),
            depths=tuple(model_config.get('depths', [2, 2, 2, 2])),
            window_size=model_config.get('window_size', 7),
            patch_size=model_config.get('patch_size', 2),
            qkv_bias=model_config.get('qkv_bias', True),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            norm_name=model_config.get('norm_name', 'instance'),
            drop_rate=model_config.get('dropout_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            dropout_path_rate=model_config.get('dropout_path_rate', 0.0),
            use_checkpoint=model_config.get('use_checkpoint', False),
        )
    elif model_name =='swin_unetr_gasa':
        return create_swin_unetr_gasa_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            feature_size=model_config.get('feature_size', 48),
            num_heads=tuple(model_config.get('num_heads', [3, 6, 12, 24])),
            depths=tuple(model_config.get('depths', [2, 2, 2, 2])),
            window_size=model_config.get('window_size', 7),
            patch_size=model_config.get('patch_size', 2),
            qkv_bias=model_config.get('qkv_bias', True),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            norm_name=model_config.get('norm_name', 'instance'),
            drop_rate=model_config.get('dropout_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            dropout_path_rate=model_config.get('dropout_path_rate', 0.0),
            use_checkpoint=model_config.get('use_checkpoint', False),
        )
    elif model_name =='swin_unetr_dilated':
        return create_swin_unetr_dilated_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            feature_size=model_config.get('feature_size', 48),
            num_heads=tuple(model_config.get('num_heads', [3, 6, 12, 24])),
            depths=tuple(model_config.get('depths', [2, 2, 2, 2])),
            window_size=model_config.get('window_size', 7),
            patch_size=model_config.get('patch_size', 2),
            qkv_bias=model_config.get('qkv_bias', True),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            norm_name=model_config.get('norm_name', 'instance'),
            drop_rate=model_config.get('dropout_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            dropout_path_rate=model_config.get('dropout_path_rate', 0.0),
            use_checkpoint=model_config.get('use_checkpoint', False),
        )
    elif model_name == 'unet3d':
        return create_unet3d_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            channels=model_config.get('channels', [16, 32, 64, 128, 256]),
            strides=model_config.get('strides', [2, 2, 2, 2]),
            num_res_units=model_config.get('num_res_units', 2),
            norm=model_config.get('norm', 'INSTANCE'),
            dropout=model_config.get('dropout', model_config.get('dropout_rate', 0.0)),
        )
    
    elif model_name == 'attention_unet3d':
        return create_attention_unet3d_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            channels=model_config.get('channels', [16, 32, 64, 128, 256]),
            strides=model_config.get('strides', [2, 2, 2, 2]),
        )
    
    elif model_name == 'vnet':
        return create_vnet_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            act=model_config.get('act', 'PRELU'),
            dropout_prob=model_config.get('dropout', model_config.get('dropout_rate', 0.0)),
            dropout_prob_down=model_config.get('dropout_prob_down'),
            dropout_prob_up=model_config.get('dropout_prob_up'),
            dropout_dim=model_config.get('dropout_dim', 3),
            bias=model_config.get('bias', False),
        )

    elif model_name == 'unetr':
        return create_unetr_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            img_size=tuple(model_config.get('img_size', [96, 96, 96])),
            feature_size=model_config.get('feature_size', 16),
            hidden_size=model_config.get('hidden_size', 768),
            mlp_dim=model_config.get('mlp_dim', 3072),
            num_heads=model_config.get('num_heads', 12),
            proj_type=model_config.get('proj_type', 'conv'),
            norm_name=model_config.get('norm_name', 'instance'),
            dropout_rate=model_config.get('dropout_rate', 0.0),
            qkv_bias=model_config.get('qkv_bias', False),
        )
    elif model_name == 'nnunet':
        return create_nnunet_model(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 4),
            img_size=tuple(model_config.get('img_size', [96, 96, 96])),
            channels=tuple(model_config.get('channels', [32, 64, 128, 256, 320])),
            strides=tuple(model_config.get('strides', [[2, 2, 2]])),
            kernel_size=tuple(model_config.get('kernel_size', [3, 3, 3])),
            upsample_kernel_size=tuple(model_config.get('upsample_kernel_size', [3, 3, 3])),
            norm_name=model_config.get('norm_name', 'instance'),
            deep_supervision=model_config.get('deep_supervision', False),
        )
    else:
        raise ValueError(f"不支持的模型名称: {model_name}. "
                        f"支持的模型: swin_unetr, unet3d, attention_unet3d, vnet")


if __name__ == "__main__":
    # 测试所有模型创建
    test_configs = [
        {'name': 'swin_unetr', 'in_channels': 1, 'out_channels': 4, 'feature_size': 48},
        {'name': 'unet3d', 'in_channels': 1, 'out_channels': 4},
        {'name': 'attention_unet3d', 'in_channels': 1, 'out_channels': 4},
        {'name': 'vnet', 'in_channels': 1, 'out_channels': 4},
    ]
    
    x = torch.randn(1, 1, 96, 96, 96)
    
    for config in test_configs:
        try:
            model = create_model(config)
            with torch.no_grad():
                y = model(x)
            print(f"{config['name']}: 输入{x.shape} -> 输出{y.shape}, "
                  f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        except Exception as e:
            print(f"{config['name']}: 创建失败 - {e}")

"""
推理脚本：对新的CT数据进行分割预测
"""
import os
import yaml
import torch
import numpy as np
from pathlib import Path

from sympy import false
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.model_factory import create_model
from utils.data_utils import load_nii_gz, save_nii_gz, get_inference_transforms
from monai.transforms import Compose
from collections import OrderedDict

def inference_single_image(model, image_path: str, device: torch.device, 
                          transforms, original_shape: tuple = None):
    """
    对单张图像进行推理
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        device: 设备
        transforms: 数据变换
        original_shape: 原始图像形状（用于恢复）
    """
    # 加载图像
    data_dict = {'image': image_path}
    data_dict = transforms(data_dict)
    
    image = data_dict['image'].unsqueeze(0).to(device)  # 添加batch维度
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return pred


def inference(config_path: str = "configs/config.yaml", 
              input_dir: str = None, output_dir: str = None):
    """
    批量推理
    
    Args:
        config_path: 配置文件路径
        input_dir: 输入图像目录
        output_dir: 输出预测结果目录
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置输入输出目录
    if input_dir is None:
        input_dir = config['data']['raw_data_root']
    if output_dir is None:
        output_dir = config['inference']['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型（使用模型工厂）
    print(f"创建模型: {config['model'].get('name', 'swin_unetr')}")
    model = create_model(config['model'])
    model = model.to(device)
    
    # 加载权重
    checkpoint_path = config['inference']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=false)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

            # remove ddp's preº
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                new_state_dict[k] = v

            msg = model.load_state_dict(new_state_dict, strict=True)
            print("Loaded checkpoint. Missing/unexpected:", msg)
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功")
    else:
        print(f"错误: 找不到模型权重文件 {checkpoint_path}")
        return
    
    # 准备数据变换（推理时只需要image，不需要label）
    transforms = get_inference_transforms(config['model']['img_size'])
    
    # 获取所有输入图像
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    print(f"\n找到 {len(image_files)} 个图像文件")
    print("开始推理...")
    
    # 批量推理
    for image_file in tqdm(image_files, desc="推理中"):
        image_path = os.path.join(input_dir, image_file)
        
        try:
            # 加载原始图像信息（用于保存时保持相同的affine和header）
            original_data, info = load_nii_gz(image_path)
            original_shape = original_data.shape
            
            # 推理
            pred = inference_single_image(model, image_path, device, transforms, original_shape)
            
            # 保存预测结果
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            save_nii_gz(pred.astype(np.uint8), output_path, info['affine'], info['header'])
            
        except Exception as e:
            print(f"处理 {image_file} 时出错: {e}")
            continue
    
    print(f"\n推理完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用Swin UNETR进行种子分割推理")
    parser.add_argument("--config", type=str, default="/home/songanran/pyProject/Swin-unetr/project/configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--input", type=str, default=None,
                       help="输入图像目录")
    parser.add_argument("--output", type=str, default=None,
                       help="输出预测结果目录")
    args = parser.parse_args()
    
    inference(args.config, args.input, args.output)



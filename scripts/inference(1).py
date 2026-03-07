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
from monai.transforms import Compose, Resize
import torch.nn.functional as F
from collections import OrderedDict
import nibabel as nib
import traceback


def get_crop_bbox(image_data: np.ndarray, threshold: float = 0.0):
    """
    获取前景区域的边界框（bbox）
    
    Args:
        image_data: 图像数据 [H, W, D]
        threshold: 前景阈值
        
    Returns:
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # 找到非零区域
    coords = np.where(image_data > threshold)
    if len(coords[0]) == 0:
        # 如果没有前景，返回整个图像
        return (0, image_data.shape[0], 0, image_data.shape[1], 0, image_data.shape[2])
    
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def inference_single_image(model, image_path: str, device: torch.device, 
                          transforms, preprocess_transforms=None):
    """
    对单张图像进行推理
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        device: 设备
        transforms: 完整的数据变换（包含Resize）
        preprocess_transforms: 预处理变换（不包含Resize，用于获取processed_shape和bbox）
        
    Returns:
        pred: 预测结果（numpy数组，96×96×96）
        processed_shape: 预处理后的图像形状（在Resize之前）
        processed_bbox: 预处理后的bbox（在原始预处理图像中的位置）
        preprocessed_image: 预处理后的图像（用于获取bbox）
    """
    # 获取预处理后的数据（在Resize之前）
    if preprocess_transforms is not None:
        preprocessed_data = preprocess_transforms({'image': image_path})
        preprocessed_image = preprocessed_data['image'][0].numpy()  # 去掉channel维度
        processed_shape = preprocessed_image.shape
        # 获取裁剪bbox（在预处理图像中的位置，实际上CropForegroundd已经裁剪过了）
        # 所以processed_shape就是裁剪后的尺寸，bbox就是整个图像
        processed_bbox = (0, processed_shape[0], 0, processed_shape[1], 0, processed_shape[2])
        print(f"Preprocessed Image Shape: {processed_shape}")
        print(f"Preprocessed Image (numpy) Sample: {preprocessed_image.shape}")
        print(f"Processed Bbox: {processed_bbox}")
    else:
        processed_shape = None
        processed_bbox = None
        preprocessed_image = None
        print("No preprocessing applied.")
    
    # 加载图像并应用完整transforms进行推理
    data_dict = {'image': image_path}
    data_dict = transforms(data_dict)
    
    image = data_dict['image'].unsqueeze(0).to(device)  # 添加batch维度
    print(f"Transformed Image Shape: {image.shape}")
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    print(f"Model Output Shape: {output.shape}")
    print(f"Predicted Label Shape: {pred.shape}")
    return pred, processed_shape, processed_bbox, preprocessed_image


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
    # device = 'cpu'
    
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            # model.load_state_dict(checkpoint['model_state_dict'])
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

    
    # 创建预处理transforms（不包含Resize，用于获取processed_shape）
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
        ScaleIntensityRanged, CropForegroundd
    )

    preprocess_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
    ])
    
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
            
            # 推理（得到96×96×96的预测结果、processed_shape和bbox）
            pred, processed_shape, processed_bbox, preprocessed_image = inference_single_image(
                model, image_path, device, transforms, preprocess_transforms
            )
            
            # 步骤1: 将96×96×96的预测结果resize回CropForegroundd后的尺寸
            if pred.shape != processed_shape:
                pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W, D]
                pred_resized = F.interpolate(
                    pred_tensor,
                    size=processed_shape,
                    mode='nearest'  # 最近邻插值，保持标签值不变
                ).squeeze(0).squeeze(0).numpy().astype(np.uint8)
            else:
                pred_resized = pred.astype(np.uint8)


            # 步骤2: 将预测结果映射回原始图像空间
            # 需要处理：Spacingd + Orientationd + CropForegroundd 的逆变换
            # 由于这些变换比较复杂，我们采用更实用的方法：
            # 1. 获取预处理后的图像（经过Spacingd+Orientationd但未Crop）
            # 2. 找到原始图像中对应的前景区域
            # 3. 将预测结果resize并放回正确位置
            
            # 获取预处理后的完整图像（未Crop）用于计算bbox
            from monai.transforms import (
                LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
                ScaleIntensityRanged
            )
            preprocess_no_crop = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                ),
            ])
            preprocessed_full = preprocess_no_crop({'image': image_path})
            preprocessed_full_image = preprocessed_full['image'][0].numpy()
            preprocessed_full_shape = preprocessed_full_image.shape
            
            # 获取完整预处理图像中的bbox
            full_bbox = get_crop_bbox(preprocessed_full_image, threshold=0.0)
            z_min, z_max, y_min, y_max, x_min, x_max = full_bbox
            bbox_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
            
            # 创建一个与完整预处理图像相同尺寸的零数组
            pred_full_preprocessed = np.zeros(preprocessed_full_shape, dtype=np.uint8)
            
            # 将resize后的预测结果放回bbox位置
            # 如果尺寸不匹配，需要再次resize
            if pred_resized.shape != bbox_shape:
                pred_tensor = torch.from_numpy(pred_resized).unsqueeze(0).unsqueeze(0).float()
                pred_bbox = F.interpolate(
                    pred_tensor,
                    size=bbox_shape,
                    mode='nearest'
                ).squeeze(0).squeeze(0).numpy().astype(np.uint8)
            else:
                pred_bbox = pred_resized



            # 将预测结果放回bbox位置
            pred_full_preprocessed[z_min:z_max, y_min:y_max, x_min:x_max] = pred_bbox
            
            # 步骤3: 将完整预处理图像resize回原始尺寸
            # 注意：由于Spacingd和Orientationd改变了图像的空间信息，
            # 直接resize可能不够精确，但对于大多数情况已经足够
            if pred_full_preprocessed.shape != original_shape:
                pred_tensor = torch.from_numpy(pred_full_preprocessed).unsqueeze(0).unsqueeze(0).float()
                pred_final = F.interpolate(
                    pred_tensor,
                    size=original_shape,
                    mode='nearest'  # 最近邻插值
                ).squeeze(0).squeeze(0).numpy().astype(np.uint8)
            else:
                pred_final = pred_full_preprocessed
            # print("original_shape:", original_shape)
            # print("pred_shape:", pred.shape)
            # print("processed_shape:", processed_shape)
            # print("preprocessed_full_shape:", preprocessed_full_shape)
            # print("bbox_shape:", bbox_shape)
            # print("pred_resized_shape:", pred_resized.shape)
            # print("pred_full_preprocessed_shape:", pred_full_preprocessed.shape)
            # print("pred_final_shape:", pred_final.shape)
            # 保存预测结果（使用原始图像的affine和header，保持空间信息一致）
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            save_nii_gz(pred_final, output_path, info['affine'], info['header'])
            nii = nib.load(output_path)
            # print("SAVED file:", output_path)
            # print("SAVED shape:", nii.shape)
            # print("SAVED zooms:", nii.header.get_zooms())
            ct = nib.load(image_path)
            # print("ORIG zooms:", ct.header.get_zooms())
        except Exception as e:
            print(f"处理 {image_file} 时出错: {e}")
            traceback.print_exc()
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



"""
评估脚本：评估模型在测试集上的性能
"""
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.swin_unetr_model import create_swin_unetr_model
from utils.data_utils import get_val_transforms, SeedDataset, split_dataset
from utils.metrics import compute_metrics, MetricAccumulator
from collections import OrderedDict
import logging

logging.getLogger("monai").setLevel(logging.ERROR)


def evaluate(config_path: str = "configs/config.yaml", checkpoint_path: str = None):
    """
    评估模型
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径（如果为None，使用配置文件中的路径）
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    data_config = config['data']
    processed_dir = data_config['processed_data_root']
    
    _, _, test_list = split_dataset(
        processed_dir,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        test_ratio=config['training']['test_ratio'],
        random_seed=config['training']['random_seed']
    )
    
    if len(test_list) == 0:
        print("警告: 测试集为空，使用验证集进行评估")
        _, test_list, _ = split_dataset(
            processed_dir,
            train_ratio=config['training']['train_ratio'],
            val_ratio=config['training']['val_ratio'],
            test_ratio=config['training']['test_ratio'],
            random_seed=config['training']['random_seed']
        )
    
    # 创建数据集和数据加载器
    val_transforms = get_val_transforms(config['model']['img_size'])
    test_dataset = SeedDataset(test_list, transforms=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # 创建模型
    model = create_swin_unetr_model(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        feature_size=config['model'].get('feature_size', 48),
        num_heads=tuple(config['model'].get('num_heads', (3, 6, 12, 24))),
        depths=tuple(config['model'].get('depths', (2, 2, 2, 2))),
        window_size=config['model'].get('window_size', 7),
        patch_size=config['model'].get('patch_size', 2),
        qkv_bias=config['model'].get('qkv_bias', True),
        mlp_ratio=config['model'].get('mlp_ratio', 4.0),
        norm_name=config['model'].get('norm_name', 'instance'),
        drop_rate=config['model'].get('dropout_rate', 0.0),
        use_checkpoint=config['model'].get('use_checkpoint', False),
    )
    model = model.to(device)
    
    # 加载权重
    if checkpoint_path is None:
        checkpoint_path = config['inference']['checkpoint_path']
    
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        '''songar:20260120'''
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        '''songar:20260120'''
        if 'model_state_dict' in checkpoint:
            '''songar:20260120'''
            # model.load_state_dict(checkpoint['model_state_dict'])
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

            # 1) move DDP/DataParallel's pre-string "module"
            if isinstance(state, dict) and len(state) > 0:
                first_key = next(iter(state.keys()))
                if first_key.startswith("module."):
                    new_state = OrderedDict()
                    for k, v in state.items():
                        new_state[k[len("module."):]] = v
                    state = new_state

            # 2) load
            missing, unexpected = model.load_state_dict(state, strict=False)

            print("Missing keys (first 20):", missing[:20])
            print("Unexpected keys (first 20):", unexpected[:20])
            '''songar:20260120'''
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功")
    else:
        print(f"警告: 找不到模型权重文件 {checkpoint_path}")
        return
    
    # 评估（使用增量计算避免内存溢出）
    model.eval()
    metric_accumulator = MetricAccumulator(num_classes=config['model']['out_channels'], ignore_index=0)
    
    print("\n开始评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 增量计算指标（避免内存溢出）
            metric_accumulator.update(preds, labels)
    
    # 计算最终指标
    metrics = metric_accumulator.compute_all()
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print("\nDice系数:")
    for class_name, score in metrics['dice'].items():
        print(f"  {class_name}: {score:.4f}")
    
    print("\nIoU:")
    for class_name, score in metrics['iou'].items():
        print(f"  {class_name}: {score:.4f}")
    print("="*50)
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_path = 'results/evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存到: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估Swin UNETR种子分割模型")
    parser.add_argument("--config", type=str, default="/home/songanran/pyProject/Swin-unetr/project/configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型检查点路径")
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)




"""
训练脚本：使用Swin UNETR训练种子分割模型
"""
import os
from copy import deepcopy

import yaml
import torch
import torch.nn as nn
try:
    from monai.data import DataLoader  # 使用MONAI的DataLoader，更好地处理字典格式
except ImportError:
    from torch.utils.data import DataLoader  # 回退到PyTorch的DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import get_train_transforms, get_val_transforms, SeedDataset, build_train_transforms
from utils.losses import CombinedLoss
" songar:20260119"
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.model_factory import create_model
from models.swin_unetr_model import create_swin_unetr_model, load_pretrained_weights, load_pretrained_weights_for_pt
from torch.cuda.amp import autocast, GradScaler
" songar:20260119"


def split_dataset(data_dir: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15,
                  random_seed: int = 42):
    """
    划分数据集
    
    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    # 获取所有样本ID
    image_files = [f.replace('.nii.gz', '') for f in os.listdir(images_dir) 
                   if f.endswith('.nii.gz')]
    
    # 过滤出有对应标签的样本
    valid_samples = []
    for sample_id in image_files:
        image_path = os.path.join(images_dir, f'{sample_id}.nii.gz')
        label_path = os.path.join(labels_dir, f'{sample_id}.nii.gz')
        if os.path.exists(label_path):
            valid_samples.append({
                'image': image_path,
                'label': label_path
            })
    
    # 随机划分
    np.random.seed(random_seed)
    np.random.shuffle(valid_samples)
    
    n_total = len(valid_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_list = valid_samples[:n_train]
    val_list = valid_samples[n_train:n_train+n_val]
    test_list = valid_samples[n_train+n_val:]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_list)} 个样本")
    print(f"  验证集: {len(val_list)} 个样本")
    print(f"  测试集: {len(test_list)} 个样本")
    
    return train_list, val_list, test_list


" songar:20260119"
# For DDP
def is_distributed():
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

def setup_distributed():
    # if initialized, return
    if dist.is_available() and dist.is_initialized():
        # local_rank from: torchrun will LOCAL_RANK
        return int(os.environ.get("LOCAL_RANK", 0))
    # torchrun
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
" songar:20260119"


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    from utils.metrics import MetricAccumulator

    # initialize GradScaler
    scaler = GradScaler()
    model.train()
    total_loss = 0.0
    metric_accumulator = MetricAccumulator(num_classes=4, ignore_index=0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", 
                unit="batch", ncols=100, leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 确保标签是整数类型
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # 前向传播
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

            m = model.module if hasattr(model, "module") else model

            # swin_unetr moe
            if hasattr(m, "moe_aux_loss"):
                moe_loss = m.moe_aux_loss()
            else:
                moe_loss = 0.0

            loss = loss + 0.01 * moe_loss
        # 反向传播
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        
        total_loss += loss.item()
        avg_loss_so_far = total_loss / (batch_idx + 1)
        
        # 增量计算指标（避免内存溢出）
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # 只在第一个batch的第一次迭代打印调试信息
            if epoch == 1 and batch_idx == 0:
                unique_preds = torch.unique(preds).cpu().numpy()
                unique_labels = torch.unique(labels).cpu().numpy()
                print(f"\n[调试] 训练 Epoch {epoch}, Batch 0:")
                print(f"  预测唯一值: {unique_preds}")
                print(f"  标签唯一值: {unique_labels}")
                print(f"  预测形状: {preds.shape}, 标签形状: {labels.shape}")
                print(f"  预测dtype: {preds.dtype}, 标签dtype: {labels.dtype}")
            metric_accumulator.update(preds, labels, debug=(epoch == 1 and batch_idx == 0))
        
        # 更新进度条显示
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss_so_far:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)


    
    # 计算最终指标
    metrics = metric_accumulator.compute_all()
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """验证"""
    from utils.metrics import MetricAccumulator
    
    model.eval()

    loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    n_batches = torch.tensor(0.0, device=device, dtype=torch.float64)
    metric_accumulator = MetricAccumulator(num_classes=4, ignore_index=0)

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", 
                    unit="batch", ncols=100, leave=False) if rank == 0 else dataloader
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # 确保标签是整数类型
            if labels.dtype != torch.long:
                labels = labels.long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.detach().to(torch.float64)
            n_batches += 1.0
            
            # total_loss += loss.item()
            # avg_loss_so_far = total_loss / (batch_idx + 1)
            
            # 增量计算指标（避免内存溢出）
            preds = torch.argmax(outputs, dim=1)
            metric_accumulator.update(preds, labels)

            if rank == 0:
                avg_loss_so_far = (loss_sum / torch.clamp(n_batches, min=1.0)).item()
                # 更新进度条显示
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss_so_far:.4f}'
                })

    # DDP : loss + Statistic
    if is_dist:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        # dice/iou DDP
        metric_accumulator.all_reduce_(device=device)

    avg_loss = (loss_sum / torch.clamp(n_batches, min=1.0)).item()
    
    # 计算最终指标
    metrics = metric_accumulator.compute_all()
    return avg_loss, metrics


def train(config_path: str = "configs/config.yaml"):
    """主训练函数"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # =============== DDP initializaiton ===============
    distributed = is_distributed()
    if distributed:
        local_rank = setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    # device (Each process is bound to one card)
    if torch.cuda.is_available() and config.get("device", "cuda") == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"divce?: {device}")
        print(f"DDP: {distributed}, world_size={world_size}")

    # crate log/model contents ( only for rank0 key)
    if rank == 0:
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    # # 设置设备
    # device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    # print(f"使用设备: {device}")
    #
    # # 创建输出目录
    # os.makedirs('logs', exist_ok=True)
    # os.makedirs('models', exist_ok=True)
    
    # 准备数据
    data_config = config['data']
    processed_dir = data_config['processed_data_root']
    
    train_list, val_list, test_list = split_dataset(
        processed_dir,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        test_ratio=config['training']['test_ratio'],
        random_seed=config['training']['random_seed']
    )
    
    # 创建数据集和数据加载器
    # train_transforms = get_train_transforms(config['model']['img_size'])
    train_transforms = build_train_transforms(config)
    val_transforms = get_val_transforms(config['model']['img_size'])
    
    train_dataset = SeedDataset(train_list, transforms=train_transforms)
    val_dataset = SeedDataset(val_list, transforms=val_transforms)

    # =============== DistributedSampler (key) ===============
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(config['num_workers']),
        pin_memory=bool(config['pin_memory'])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(config['num_workers']),
        pin_memory=bool(config['pin_memory'])
    )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=int(config['training']['batch_size']),
    #     shuffle=True,
    #     num_workers=int(config['num_workers']),
    #     pin_memory=bool(config['pin_memory'])
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,  # 验证时使用batch_size=1
    #     shuffle=False,
    #     num_workers=int(config['num_workers']),
    #     pin_memory=bool(config['pin_memory'])
    # )
    
    # 创建模型
    # model = create_swin_unetr_model(
    #     in_channels=config['model']['in_channels'],
    #     out_channels=config['model']['out_channels'],
    #     feature_size=config['model'].get('feature_size', 48),
    #     num_heads=tuple(config['model'].get('num_heads', (3, 6, 12, 24))),
    #     depths=tuple(config['model'].get('depths', (2, 2, 2, 2))),
    #     window_size=config['model'].get('window_size', 7),
    #     patch_size=config['model'].get('patch_size', 2),
    #     qkv_bias=config['model'].get('qkv_bias', True),
    #     mlp_ratio=config['model'].get('mlp_ratio', 4.0),
    #     norm_name=config['model'].get('norm_name', 'instance'),
    #     drop_rate=config['model'].get('dropout_rate', 0.0),
    #     use_checkpoint=config['model'].get('use_checkpoint', False),
    # )
    # model = model.to(device)

    # creat model
    print(f"creat model: {config['model'].get('name', 'unet3d')}")
    model = create_model(config['model'])
    model = model.to(device)

    # print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters size: {total_params / 1e6:.2f}M (training: {trainable_params / 1e6:.2f}M)")

    pretrained_path = "/home/songanran/pyProject/Swin-unetr/project/model_swinvit.pt"  # load pretrained
    # pretrained_path = ""

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"used pretrained: {pretrained_path}")
        load_pretrained_weights_for_pt(model, pretrained_path,prefix_mode="manual",add_prefix="swinViT.")
    else:
        print("no used!!!!!¯")

    # =============== DDP bag ===============
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 损失函数（支持创新损失函数）
    loss_config = config['training']['loss']
    use_advanced_loss = loss_config.get('use_boundary_loss', False) or \
                       loss_config.get('use_shape_loss', False) or \
                       loss_config.get('use_class_weight', False)
    
    if use_advanced_loss:
        from utils.losses import AdvancedCombinedLoss
        criterion = AdvancedCombinedLoss(
            dice_weight=float(loss_config.get('dice_weight', 0.3)),
            ce_weight=float(loss_config.get('ce_weight', 0.3)),
            boundary_weight=float(loss_config.get('boundary_weight', 0.2)),
            shape_weight=float(loss_config.get('shape_weight', 0.1)),
            class_weight_weight=float(loss_config.get('class_weight_weight', 0.1)),
            num_classes=int(config['model']['out_channels']),
            boundary_loss_weight=float(loss_config.get('boundary_loss_weight', 2.0)),
            boundary_width=int(loss_config.get('boundary_width', 2)),
            use_boundary_loss=bool(loss_config.get('use_boundary_loss', True)),
            use_shape_loss=bool(loss_config.get('use_shape_loss', True)),
            use_class_weight=bool(loss_config.get('use_class_weight', True))
        )
        print("使用高级组合损失函数（包含创新损失）")
    else:
        criterion = CombinedLoss(
            dice_weight=float(loss_config.get('dice_weight', 0.5)),
            ce_weight=float(loss_config.get('ce_weight', 0.5)),
            num_classes=int(config['model']['out_channels'])
        )
        print("使用基础组合损失函数")
    
    # 获取训练轮数（用于学习率调度器）
    num_epochs = int(config['training']['num_epochs'])
    
    # 优化器（确保学习率和权重衰减是浮点数）
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器（支持warmup）
    warmup_epochs = int(config['training'].get('warmup_epochs', 0))
    if config['training']['scheduler'] == 'cosine':
        if warmup_epochs > 0:
            # 使用带warmup的余弦退火
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    # 余弦退火
                    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = LambdaLR(optimizer, lr_lambda)
            print(f"使用带warmup的余弦退火调度器 (warmup: {warmup_epochs} epochs)")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
    else:
        scheduler = None
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_dice = 0.0
    best_model_state = None
    best_epoch = 0
    
    # 早停机制
    early_stopping_config = config['training'].get('early_stopping', {})
    use_early_stopping = early_stopping_config.get('enabled', False)
    patience = int(early_stopping_config.get('patience', 10))
    min_delta = float(early_stopping_config.get('min_delta', 0.001))
    restore_best_weights = early_stopping_config.get('restore_best_weights', True)
    target_dice = float(early_stopping_config.get('target_dice', 0.0))  # 目标Dice值
    
    if use_early_stopping:
        print(f"启用早停机制: patience={patience}, min_delta={min_delta}")
        if target_dice > 0:
            print(f"目标Dice: {target_dice:.4f}")
    
    no_improve_count = 0

    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    # 总体进度条
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="总体进度", 
                     unit="epoch", ncols=100)
    target_reached = False
    try:
        for epoch in epoch_pbar:
            '''songar 20260120'''

            # DDP
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            '''songar 20260120'''
            # 训练
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # 验证
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            # 更新学习率
            if scheduler:
                scheduler.step()


            if rank == 0:
                # 记录历史
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_metrics'].append(train_metrics)
                history['val_metrics'].append(val_metrics)

                # 保存训练历史
                with open('logs/training_history.json', 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                # 更新总体进度条
                val_dice_avg = val_metrics['dice']['平均']
                val_iou_avg = val_metrics['iou']['平均']
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_dice': f'{val_dice_avg:.4f}',
                    'val_iou': f'{val_iou_avg:.4f}',
                    'best_dice': f'{best_val_dice:.4f}'
                })

                # 打印详细结果
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"{'='*60}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"\n验证指标 (Dice):")
                for class_name, score in val_metrics['dice'].items():
                    print(f"  {class_name}: {score:.4f}")
                print(f"\n验证指标 (IoU):")
                for class_name, score in val_metrics['iou'].items():
                    print(f"  {class_name}: {score:.4f}")

                # 保存最佳模型和早停判断
                improved = False
                if val_dice_avg > best_val_dice + min_delta:
                    best_val_dice = val_dice_avg
                    best_epoch = epoch
                    no_improve_count = 0
                    improved = True

                    # 保存最佳模型
                    if restore_best_weights:
                        best_model_state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict().copy(),
                            'optimizer_state_dict': optimizer.state_dict().copy(),
                            'val_dice': val_dice_avg,
                            'config': config
                        }

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': val_dice_avg,
                        'config': config
                    }, config['model']['save_path'])
                    print(f"\n[保存] 最佳模型 (Val Dice: {val_dice_avg:.4f}, Epoch {epoch})")

                    # 检查是否达到目标Dice
                    if target_dice > 0 and val_dice_avg >= target_dice and not target_reached:
                        target_reached = True
                        print(f"\n[目标达成] 达到目标Dice {target_dice:.4f}！(当前: {val_dice_avg:.4f})")
                else:
                    no_improve_count += 1
                    if use_early_stopping and no_improve_count >= patience:
                        print(f"\n{'='*60}")
                        print(f"早停触发！连续 {patience} 个epoch未提升")
                        print(f"最佳验证集Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
                        print(f"{'='*60}")
                        if restore_best_weights and best_model_state is not None:
                            model.load_state_dict(best_model_state['model_state_dict'])
                            optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                            torch.save(best_model_state, config['model']['save_path'])
                            print(f"已恢复最佳模型权重 (Epoch {best_epoch})")

                        if target_reached:  # 如果已达到目标，可以提前停止；否则继续训练
                            print(f"已达到目标Dice，提前停止训练")
                            break
                        else:
                            print(f"继续训练，尝试达到目标Dice {target_dice:.4f}")
                            no_improve_count = 0  # 重置计数器，给更多机会

            # epoch_pbar.close()
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"最佳模型保存在: {config['model']['save_path']}")
        print(f"最佳验证 Dice: {best_val_dice:.4f}")
        '''songar 20260120'''
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练Swin UNETR种子分割模型")
    parser.add_argument("--config", type=str, default="/home/songanran/pyProject/Swin-unetr/project/configs/config.yaml",
                       help="配置文件路径")
    args = parser.parse_args()
    
    train(args.config)



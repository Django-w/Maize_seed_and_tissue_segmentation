"""
分析训练日志中的指标变化
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_history(json_path: str):
    """分析训练历史"""
    with open(json_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # 提取验证集指标
    val_metrics = history['val_metrics']
    train_metrics = history['train_metrics']
    
    # 提取平均Dice和IoU
    val_dice_avg = [m['dice']['平均'] for m in val_metrics]
    val_iou_avg = [m['iou']['平均'] for m in val_metrics]
    train_dice_avg = [m['dice']['平均'] for m in train_metrics]
    train_iou_avg = [m['iou']['平均'] for m in train_metrics]
    
    # 提取各类别指标
    val_dice_embryo = [m['dice']['种胚'] for m in val_metrics]
    val_dice_endosperm = [m['dice']['胚乳'] for m in val_metrics]
    val_dice_cavity = [m['dice']['空腔'] for m in val_metrics]
    
    print("="*60)
    print("验证集指标分析")
    print("="*60)
    print(f"\n总epoch数: {len(val_metrics)}")
    print(f"\n平均Dice系数:")
    print(f"  初始: {val_dice_avg[0]:.4f}")
    print(f"  最终: {val_dice_avg[-1]:.4f}")
    print(f"  变化: {val_dice_avg[-1] - val_dice_avg[0]:.4f}")
    print(f"  最大值: {max(val_dice_avg):.4f} (epoch {val_dice_avg.index(max(val_dice_avg))+1})")
    print(f"  最小值: {min(val_dice_avg):.4f} (epoch {val_dice_avg.index(min(val_dice_avg))+1})")
    
    print(f"\n平均IoU:")
    print(f"  初始: {val_iou_avg[0]:.4f}")
    print(f"  最终: {val_iou_avg[-1]:.4f}")
    print(f"  变化: {val_iou_avg[-1] - val_iou_avg[0]:.4f}")
    print(f"  最大值: {max(val_iou_avg):.4f} (epoch {val_iou_avg.index(max(val_iou_avg))+1})")
    
    print(f"\n各类别Dice系数变化:")
    print(f"  种胚: {val_dice_embryo[0]:.4f} -> {val_dice_embryo[-1]:.4f} (变化: {val_dice_embryo[-1] - val_dice_embryo[0]:.4f})")
    print(f"  胚乳: {val_dice_endosperm[0]:.4f} -> {val_dice_endosperm[-1]:.4f} (变化: {val_dice_endosperm[-1] - val_dice_endosperm[0]:.4f})")
    print(f"  空腔: {val_dice_cavity[0]:.4f} -> {val_dice_cavity[-1]:.4f} (变化: {val_dice_cavity[-1] - val_dice_cavity[0]:.4f})")
    
    # 计算后期变化（最后10个epoch）
    if len(val_dice_avg) >= 10:
        last_10_dice = val_dice_avg[-10:]
        print(f"\n最后10个epoch的平均Dice:")
        print(f"  平均值: {np.mean(last_10_dice):.4f}")
        print(f"  标准差: {np.std(last_10_dice):.4f}")
        print(f"  变化范围: {max(last_10_dice) - min(last_10_dice):.4f}")
    
    # 对比训练集和验证集
    print(f"\n训练集 vs 验证集 (最终):")
    print(f"  训练集Dice: {train_dice_avg[-1]:.4f}")
    print(f"  验证集Dice: {val_dice_avg[-1]:.4f}")
    print(f"  差距: {train_dice_avg[-1] - val_dice_avg[-1]:.4f}")
    
    # 检查是否过拟合
    if train_dice_avg[-1] - val_dice_avg[-1] > 0.1:
        print(f"\n[警告] 可能存在过拟合 (训练集Dice比验证集高 {train_dice_avg[-1] - val_dice_avg[-1]:.4f})")
    
    # 检查验证集是否停滞
    if len(val_dice_avg) >= 20:
        recent_20 = val_dice_avg[-20:]
        if max(recent_20) - min(recent_20) < 0.01:
            print(f"\n[警告] 验证集指标可能已停滞 (最近20个epoch变化 < 0.01)")
    
    return {
        'val_dice_avg': val_dice_avg,
        'val_iou_avg': val_iou_avg,
        'train_dice_avg': train_dice_avg,
        'train_iou_avg': train_iou_avg,
    }

if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else "../training_history(3).json"
    analyze_training_history(json_path)

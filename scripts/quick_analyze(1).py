"""分析验证集指标变化"""
import json
import os
from pathlib import Path

# 获取脚本所在目录的父目录的父目录（项目根目录）
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
json_path = project_root / 'project/logs/training_history.json'

# 读取日志
with open(json_path, 'r', encoding='utf-8') as f:
    history = json.load(f)

val_metrics = history['val_metrics']
train_metrics = history['train_metrics']

# 提取平均Dice
val_dice = [m['dice']['平均'] for m in val_metrics]
train_dice = [m['dice']['平均'] for m in train_metrics]

print("="*60)
print("验证集指标分析")
print("="*60)
print(f"\n总epoch数: {len(val_dice)}")
print(f"\n平均Dice系数变化:")
print(f"  第1个epoch: {val_dice[0]:.4f}")
print(f"  第10个epoch: {val_dice[9]:.4f}")
print(f"  第20个epoch: {val_dice[19]:.4f}")
print(f"  最后1个epoch: {val_dice[-1]:.4f}")
print(f"  总变化: {val_dice[-1] - val_dice[0]:.4f}")

# 分析后期变化
if len(val_dice) >= 20:
    last_20 = val_dice[-20:]
    print(f"\n最后20个epoch:")
    print(f"  平均值: {sum(last_20)/len(last_20):.4f}")
    print(f"  最大值: {max(last_20):.4f}")
    print(f"  最小值: {min(last_20):.4f}")
    print(f"  变化范围: {max(last_20) - min(last_20):.4f}")
    
    if max(last_20) - min(last_20) < 0.01:
        print(f"  [警告] 指标已停滞 (变化 < 0.01)")

# 对比训练集
print(f"\n训练集 vs 验证集 (最后):")
print(f"  训练集Dice: {train_dice[-1]:.4f}")
print(f"  验证集Dice: {val_dice[-1]:.4f}")
print(f"  差距: {train_dice[-1] - val_dice[-1]:.4f}")

if train_dice[-1] - val_dice[-1] > 0.05:
    print(f"  [警告] 可能存在过拟合")

# 详细分析停滞情况
print(f"\n" + "="*60)
print("详细停滞分析")
print("="*60)

# 找出指标提升的epoch
improvements = []
for i in range(1, len(val_dice)):
    if val_dice[i] > val_dice[i-1]:
        improvements.append((i+1, val_dice[i] - val_dice[i-1]))

print(f"\n指标提升的epoch数: {len(improvements)}/{len(val_dice)-1}")
if len(improvements) > 0:
    print(f"最大单次提升: {max(improvements, key=lambda x: x[1])[1]:.4f} (epoch {max(improvements, key=lambda x: x[1])[0]})")

# 找出连续不提升的epoch段
stagnant_periods = []
current_start = None
for i in range(1, len(val_dice)):
    if val_dice[i] <= val_dice[i-1]:
        if current_start is None:
            current_start = i
    else:
        if current_start is not None:
            stagnant_periods.append((current_start+1, i))
            current_start = None
if current_start is not None:
    stagnant_periods.append((current_start+1, len(val_dice)))

if stagnant_periods:
    print(f"\n连续不提升的epoch段:")
    for start, end in stagnant_periods:
        if end - start >= 5:  # 只显示连续5个epoch以上的停滞
            print(f"  Epoch {start}-{end}: {end-start+1}个epoch (Dice: {val_dice[start-1]:.4f} -> {val_dice[end-1]:.4f})")

# 分析最佳epoch
best_epoch = val_dice.index(max(val_dice)) + 1
print(f"\n最佳验证集Dice: {max(val_dice):.4f} (epoch {best_epoch})")
if best_epoch < len(val_dice) - 10:
    print(f"  [警告] 最佳epoch出现在早期，后期可能过拟合或停滞")

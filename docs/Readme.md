# Readme

[Download the trained model weights](https://pan.baidu.com/s/1QlU8bFInYxgXWQxiHFIuIA?pwd=mwc0) 
Click the link to download from Baidu Cloud, or contact 872041345@qq.com.
## 完整流程

### 1. 数据预处理

```bash
python scripts/preprocess_data.py --config configs/config.yaml
```

**作用**：
- 合并分散的标注文件（胚与胚乳、空腔）为统一的多类别标签
- 过滤不完整的样本（只有单一类别的样本）
- 生成处理后的数据到 `data/processed/` 目录

**输出**：
- `data/processed/images/` - 原始CT图像
- `data/processed/labels/` - 合并后的多类别标签（种胚=1, 胚乳=2, 空腔=3）

---

### 2. 开始训练（下一步）

```bash
python scripts/train.py --config configs/config.yaml
```

**训练过程**：
- 自动划分数据集（训练集70%，验证集15%，测试集15%）
- 加载数据并进行数据增强
- 初始化 Swin UNETR 模型
- 开始训练，每个 epoch 会显示：
  - 训练损失
  - 验证损失
  - **所有类别的 Dice 和 IoU 指标**（种胚、胚乳、空腔）
  - 最佳模型会自动保存

**训练输出**：
- `models/swin_unetr_seed_seg.pth` - 最佳模型权重
- `logs/training_history.json` - 训练历史记录
- 控制台实时显示训练进度和指标

**训练时间**：
- 根据配置的 `num_epochs`（默认200个epoch）
- 每个 epoch 的时间取决于数据量和 batch_size

---

### 3. 评估模型（训练完成后）

```bash
python scripts/evaluate.py --config configs/config.yaml
```

**作用**：
- 在测试集上评估模型性能
- 计算详细的 Dice 和 IoU 指标
- 保存预测结果（可选）

---

### 4. 推理新数据（可选）

```bash
python scripts/inference.py --config configs/config.yaml --input <新数据路径>
```

**作用**：
- 使用训练好的模型对新数据进行分割
- 生成预测结果

---

## 快速开始

预处理完成后，直接运行：

```bash
cd F:\A31 Micro-CT\project
conda activate pytorch_env
python scripts/train.py --config configs/config.yaml
```

## 训练监控

训练过程中会显示：
- 每个 epoch 的进度条
- 训练损失和验证损失
- **所有类别的指标**（种胚、胚乳、空腔的 Dice 和 IoU）
- 最佳模型保存提示

## 注意事项

1. **确保 GPU 可用**：训练需要 GPU，检查 CUDA 是否可用
2. **内存管理**：如果显存不足，可以：
   - 减小 `batch_size`（config.yaml 中）
   - 减小 `img_size`（config.yaml 中）
   - 启用 `use_checkpoint: true`（节省显存）
3. **训练中断**：可以随时 Ctrl+C 中断，已保存的最佳模型不会丢失
4. **查看日志**：训练历史保存在 `logs/training_history.json`

## 训练参数调整

在 `configs/config.yaml` 中可以调整：
- `batch_size`: 批次大小（默认2）
- `num_epochs`: 训练轮数（默认200）
- `learning_rate`: 学习率（默认0.0001）
- `img_size`: 输入图像尺寸（默认[96,96,96]）


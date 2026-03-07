# 模型对比实验

本目录包含三个对比模型的实验配置，用于与**Swin UNETR微调模型**进行性能对比。

**注意**: 本项目的主模型是**Swin UNETR**（微调），这五个实验是对比基准模型，用于评估不同架构在种子CT精细分割任务上的性能。

## 项目结构

```
compare/
├── README.md                    # 本文件
├── 1/                           # 实验1：UNet 3D
│   ├── config.yaml             # 配置文件
│   ├── doc/                    # 文档目录
│   │   └── 实验说明.md        # 实验详细说明
│   ├── models/                 # 模型权重保存目录（训练后生成）
│   │   └── unet3d_model.pth
│   └── results/                # 结果保存目录（训练后生成）
│       ├── predictions/        # 预测结果
│       └── visualizations/     # 可视化结果
├── 2/                           # 实验2：Attention UNet 3D
│   ├── config.yaml
│   ├── doc/
│   │   └── 实验说明.md
│   ├── models/
│   │   └── attention_unet3d_model.pth
│   └── results/
```

## 环境要求

* Python 3.8+

* PyTorch 2.0+

* MONAI 1.3+

* CUDA（推荐，用于GPU加速）

* 其他依赖见项目根目录的 `requirements.txt`

## 快速开始

### 1. 环境准备

```bash
# 进入项目根目录
cd F:\A31 Micro-CT\project

# 检查环境
python check_environment.py
```

### 2. 数据准备

确保数据已经预处理完成：

```bash
# 如果还未预处理数据，先运行数据预处理
python scripts/preprocess_data.py --config configs/config.yaml
```

### 3. 运行实验

### 3.1 运行单个实验

```bash
# 确保在项目根目录
cd F:\A31 Micro-CT\project

# 运行实验1（UNet 3D）
python scripts/train.py --config compare/1/config.yaml
```

### 3.2 批量运行所有实验

**Windows PowerShell:**

```powershell
cd F:\A31 Micro-CT\project
for ($i=1; $i -le 5; $i++) {
    Write-Host "开始运行实验 $i..."
    python scripts/train.py --config compare/$i/config.yaml
    Write-Host "实验 $i 完成`n"
}
```

### 4. 查看训练结果

训练完成后，结果保存在各自的目录下：

```bash
# 查看实验1的训练日志
cat compare/1/results/training_log.json
```

### 5. 评估模型性能

```bash
# 评估实验1的模型
python scripts/evaluate.py --config compare/1/config.yaml --checkpoint compare/1/models/unet3d_model.pth
```

### 6. 推理预测

```bash
# 使用实验1的模型进行推理
python scripts/inference.py --config compare/1/config.yaml --input <输入目录> --output <输出目录>
```

## 对比指标

所有实验使用相同的评估指标：

### 主要指标

* **Dice系数**: 分割重叠度

* **IoU**: 交并比

* **各类别Dice**: 种胚、胚乳、空腔分别的Dice系数

### 精度指标

* **验证集Dice**: 验证集上的平均Dice系数

* **测试集Dice**: 测试集上的平均Dice系数

## 实验对比

| 模型                      | 架构类型                              | 特点                  |    |
| ----------------------- | --------------------------------- | ------------------- | :- |
| **Swin UNETR（主模型）**     | Transformer                       | 全局上下文强，精度高          |    |
| UNet 3D（对比）             | CNN                               | 计算效率高               |    |
| Attention UNet 3D（对比）   | CNN+Attention                     | 精度与效率平衡             |    |
| VNet（对比）                | CNN+Residual                      | 3D医学图像优化            |    |
| DynUNet / nnU-Net-style |                                   | Self-adaptive U-Net |    |
| UNETR                   | Transformer encoder + CNN decoder |                     |    |


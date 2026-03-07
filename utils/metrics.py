"""
评估指标：Dice系数和IoU
支持增量计算，避免内存溢出
"""
import torch
import numpy as np
from typing import List, Dict, Optional


class MetricAccumulator:
    """
    指标累积器，支持增量计算避免内存溢出
    
    计算方式：按样本数等权（对所有像素/体素累积）
    - 不管batch_size是多少，都是累积所有样本的intersection和union
    - 最后计算dice/iou时，是总的intersection除以总的union
    - 这样训练集和验证集的指标计算口径完全一致
    """
    def __init__(self, num_classes: int = 4, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置累积器"""
        # 为每个类别累积 intersection 和 union（按像素/体素数累积，按样本数等权）
        self.intersections = {i: 0.0 for i in range(self.num_classes) if i != self.ignore_index}
        self.unions = {i: 0.0 for i in range(self.num_classes) if i != self.ignore_index}
        self.iou_unions = {i: 0.0 for i in range(self.num_classes) if i != self.ignore_index}
        # 追踪目标中是否出现过这个类别（用于正确计算平均）
        self.target_sums = {i: 0.0 for i in range(self.num_classes) if i != self.ignore_index}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, debug: bool = False):
        """
        更新累积器（增量计算，按样本数等权）
        
        注意：这里是对所有像素/体素累积，不管batch_size是多少，
        都是累积所有样本的intersection和union，确保训练集和验证集计算口径一致。
        
        Args:
            pred: 预测结果 [B, H, W, D] 或 [B, C, H, W, D]
            target: 真实标签 [B, H, W, D] 或 [B, C, H, W, D]
            debug: 是否打印调试信息
        """
        # 处理预测结果
        if pred.dim() == 5:
            pred = torch.argmax(pred, dim=1)
        
        # 处理目标标签
        # 情况1: 如果是5维且通道数>1，可能是one-hot格式，转换为类别索引
        if target.dim() == 5:
            if target.shape[1] > 1:
                # one-hot格式，转换为类别索引
                target = torch.argmax(target, dim=1)
            elif target.shape[1] == 1:
                # 单通道格式 [B, 1, H, W, D]，压缩通道维度
                target = target.squeeze(1)  # 变成 [B, H, W, D]
        
        # 确保标签值是整数（MONAI可能返回float）
        if target.dtype != torch.long:
            target = target.long()
        
        # 确保预测值是整数
        if pred.dtype != torch.long:
            pred = pred.long()
        
        # 裁剪标签值到有效范围（避免标签值超出类别范围）
        pred = torch.clamp(pred, 0, self.num_classes - 1)
        target = torch.clamp(target, 0, self.num_classes - 1)
        
        # 确保维度一致
        if pred.shape != target.shape:
            raise ValueError(f"预测和标签形状不匹配: pred {pred.shape} vs target {target.shape}")
        
        # 调试信息（只打印一次）
        if debug:
            unique_pred = torch.unique(pred).cpu().numpy()
            unique_target = torch.unique(target).cpu().numpy()
            print(f"DEBUG - 预测唯一值: {unique_pred}, 标签唯一值: {unique_target}")
            print(f"DEBUG - 预测形状: {pred.shape}, 标签形状: {target.shape}")
        
        # 对每个类别计算并累积
        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue
            
            pred_mask = (pred == class_idx).float()
            target_mask = (target == class_idx).float()
            
            # 计算当前batch的intersection和union（对所有像素/体素求和）
            # 注意：这里是对整个batch的所有样本累积，按样本数等权
            intersection = (pred_mask * target_mask).sum().item()  # 当前batch的intersection
            pred_sum = pred_mask.sum().item()  # 当前batch预测中该类别的像素/体素数
            target_sum = target_mask.sum().item()  # 当前batch目标中该类别的像素/体素数
            union = pred_sum + target_sum  # Dice的union = |pred| + |target|
            iou_union = pred_sum + target_sum - intersection  # IoU的union = |pred ∪ target|
            
            # 累积（按样本数等权：不管batch_size是多少，都是累积所有样本）
            self.intersections[class_idx] += intersection
            self.unions[class_idx] += union
            self.iou_unions[class_idx] += iou_union
            self.target_sums[class_idx] += target_sum  # 累积目标中出现该类别的像素/体素数

    def all_reduce_(self, device: torch.device):
        """
        DDP: Aggregates the statistics from each rank into global statistics (in-place modification).
        This needs to be called after validation and before `compute_all`.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return  # not DDP

        # Let python float dict -> tensor , do all_reduce, and write dict
        cls_ids = sorted(self.intersections.keys())  # ignore_index has already been excluded

        inter = torch.tensor([self.intersections[i] for i in cls_ids],
                             dtype=torch.float64, device=device)
        uni = torch.tensor([self.unions[i] for i in cls_ids],
                           dtype=torch.float64, device=device)
        iou_uni = torch.tensor([self.iou_unions[i] for i in cls_ids],
                               dtype=torch.float64, device=device)
        tgt = torch.tensor([self.target_sums[i] for i in cls_ids],
                           dtype=torch.float64, device=device)

        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(uni, op=dist.ReduceOp.SUM)
        dist.all_reduce(iou_uni, op=dist.ReduceOp.SUM)
        dist.all_reduce(tgt, op=dist.ReduceOp.SUM)

        # Write back
        for idx, c in enumerate(cls_ids):
            self.intersections[c] = float(inter[idx].item())
            self.unions[c] = float(uni[idx].item())
            self.iou_unions[c] = float(iou_uni[idx].item())
            self.target_sums[c] = float(tgt[idx].item())


    def compute_dice(self) -> Dict[str, float]:
        """
        计算Dice系数（始终显示所有类别）
        
        计算方式：按样本数等权
        - Dice = 2 * intersection / union
        - intersection和union是对所有样本累积的，确保训练集和验证集口径一致
        """
        dice_scores = {}
        valid_classes = []  # 记录目标中出现过的类别（用于计算平均）
        
        # 始终显示所有类别，即使union=0也显示为0.0
        for class_idx in sorted(self.intersections.keys()):
            intersection = self.intersections[class_idx]
            union = self.unions[class_idx]
            
            class_name = {1: "种胚", 2: "胚乳", 3: "空腔"}.get(class_idx, f"类别{class_idx}")
            
            if union == 0:
                # 如果union=0，说明预测和真实标签都没有这个类别，dice为0
                dice_scores[class_name] = 0.0
            else:
                # 计算dice
                dice = (2.0 * intersection) / union
                dice_scores[class_name] = dice
                # 只有当目标中有这个类别时，才参与平均计算
                # 这样可以避免因为预测错误导致的dice=0影响平均值的计算
                if self.target_sums[class_idx] > 0:
                    valid_classes.append(dice)
        
        # 计算平均（只统计在目标中出现过的类别）
        if valid_classes:
            dice_scores['平均'] = np.mean(valid_classes)
        else:
            dice_scores['平均'] = 0.0  # 如果所有类别都是空的，平均为0
        
        return dice_scores
    
    def compute_iou(self) -> Dict[str, float]:
        """
        计算IoU（始终显示所有类别）
        
        计算方式：按样本数等权
        - IoU = intersection / union
        - intersection和union是对所有样本累积的，确保训练集和验证集口径一致
        """
        iou_scores = {}
        valid_classes = []  # 记录目标中出现过的类别（用于计算平均）
        
        # 始终显示所有类别，即使union=0也显示为0.0
        for class_idx in sorted(self.intersections.keys()):
            intersection = self.intersections[class_idx]
            union = self.iou_unions[class_idx]
            
            class_name = {1: "种胚", 2: "胚乳", 3: "空腔"}.get(class_idx, f"类别{class_idx}")
            
            if union == 0:
                # 如果union=0，说明预测和真实标签都没有这个类别，iou为0
                iou_scores[class_name] = 0.0
            else:
                # 计算iou
                iou = intersection / union
                iou_scores[class_name] = iou
                # 只有当目标中有这个类别时，才参与平均计算
                # 这样可以避免因为预测错误导致的iou=0影响平均值的计算
                if self.target_sums[class_idx] > 0:
                    valid_classes.append(iou)
        
        # 计算平均（只统计在目标中出现过的类别）
        if valid_classes:
            iou_scores['平均'] = np.mean(valid_classes)
        else:
            iou_scores['平均'] = 0.0  # 如果所有类别都是空的，平均为0
        
        return iou_scores
    
    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """计算所有指标"""
        return {
            'dice': self.compute_dice(),
            'iou': self.compute_iou()
        }


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4, 
               ignore_index: int = 0) -> Dict[str, float]:
    """
    计算Dice系数（兼容旧接口）
    
    Args:
        pred: 预测结果 [B, H, W, D] 或 [B, C, H, W, D]
        target: 真实标签 [B, H, W, D] 或 [B, C, H, W, D]
        num_classes: 类别数量（包括背景）
        ignore_index: 忽略的类别索引
        
    Returns:
        每个类别的Dice系数字典
    """
    accumulator = MetricAccumulator(num_classes, ignore_index)
    accumulator.update(pred, target)
    return accumulator.compute_dice()


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4,
              ignore_index: int = 0) -> Dict[str, float]:
    """
    计算IoU（兼容旧接口）
    
    Args:
        pred: 预测结果 [B, H, W, D] 或 [B, C, H, W, D]
        target: 真实标签 [B, H, W, D] 或 [B, C, H, W, D]
        num_classes: 类别数量（包括背景）
        ignore_index: 忽略的类别索引
        
    Returns:
        每个类别的IoU字典
    """
    accumulator = MetricAccumulator(num_classes, ignore_index)
    accumulator.update(pred, target)
    return accumulator.compute_iou()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, 
                   num_classes: int = 4) -> Dict[str, Dict[str, float]]:
    """
    计算所有评估指标（兼容旧接口）
    
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数量
        
    Returns:
        包含所有指标的字典
    """
    accumulator = MetricAccumulator(num_classes)
    accumulator.update(pred, target)
    return accumulator.compute_all()


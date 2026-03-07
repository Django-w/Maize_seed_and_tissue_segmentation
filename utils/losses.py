"""
损失函数：Dice Loss + Cross-Entropy Loss + 创新损失函数
包含：
1. 类别不平衡加权损失
2. 边界感知损失（种子边界很重要）
3. 形状约束损失（种胚、胚乳的形状先验）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss
import numpy as np


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失：强调种子边界的分割准确性
    
    通过计算边界区域的损失权重，使模型更关注边界像素/体素
    """
    def __init__(self, base_loss_fn, boundary_weight: float = 2.0, boundary_width: int = 2):
        """
        Args:
            base_loss_fn: 基础损失函数（如DiceLoss或CrossEntropyLoss）
            boundary_weight: 边界区域的损失权重倍数
            boundary_width: 边界宽度（体素数）
        """
        super(BoundaryAwareLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width
    
    def compute_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        计算边界掩膜
        
        Args:
            target: 真实标签 [B, H, W, D]，类别索引格式
            
        Returns:
            边界掩膜 [B, H, W, D]，边界区域为1，其他为0
        """
        device = target.device
        batch_size = target.shape[0]
        boundary_mask = torch.zeros_like(target, dtype=torch.float32)
        
        # 使用3D卷积计算边界（纯PyTorch实现，避免scipy依赖）
        # 定义3x3x3的卷积核用于检测边界
        kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
        
        # 对每个batch和每个类别计算边界
        for b in range(batch_size):
            for class_idx in [1, 2, 3]:  # 种胚、胚乳、空腔
                class_mask = (target[b] == class_idx).float().unsqueeze(0).unsqueeze(0)
                if class_mask.sum() == 0:
                    continue
                
                # 使用卷积计算邻居平均值
                # 如果某个体素是类别的一部分，但邻居平均值小于1，说明它在边界上
                neighbor_avg = F.conv3d(class_mask, kernel, padding=1)
                
                # 边界：mask=1但邻居不全为1（即neighbor_avg < 1）
                boundary = (class_mask > 0.5) & (neighbor_avg < 0.9)
                boundary = boundary.squeeze(0).squeeze(0).float()
                
                # 累加到边界掩膜
                boundary_mask[b] += boundary
        
        # 归一化到[0, 1]，边界区域为1，非边界区域为0
        boundary_mask = torch.clamp(boundary_mask, 0, 1)
        return boundary_mask
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算边界感知损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]
            target: 真实标签 [B, H, W, D] 或 [B, 1, H, W, D]
            
        Returns:
            加权后的损失值
        """
        # 准备4维格式的target用于其他计算
        target_4d = target.clone()
        if target_4d.dim() == 5:
            if target_4d.shape[1] == 1:
                target_4d = target_4d.squeeze(1)  # [B, 1, H, W, D] -> [B, H, W, D]
            elif target_4d.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_4d = torch.argmax(target_4d, dim=1)  # one-hot -> 类别索引
        
        if target_4d.dtype != torch.long:
            target_4d = target_4d.long()
        
        # 准备5维格式的target用于MONAI损失函数
        target_for_loss = target.clone()
        if target_for_loss.dim() == 4:
            # 如果是4维，扩展为5维 [B, H, W, D] -> [B, 1, H, W, D]
            target_for_loss = target_for_loss.unsqueeze(1)
        elif target_for_loss.dim() == 5:
            if target_for_loss.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_for_loss = torch.argmax(target_for_loss, dim=1).unsqueeze(1)
            # 如果已经是[B, 1, H, W, D]格式，保持不变
        
        if target_for_loss.dtype != torch.long:
            target_for_loss = target_for_loss.long()
        
        # 计算基础损失（使用5维格式）
        base_loss = self.base_loss_fn(pred, target_for_loss)
        
        # 计算边界掩膜（使用4维格式）
        boundary_mask = self.compute_boundary_mask(target_4d)
        
        # 计算边界区域的额外损失
        # 将预测转换为类别索引
        pred_classes = torch.argmax(pred, dim=1)  # [B, H, W, D]
        
        # 计算边界区域的分类错误（使用4维格式）
        boundary_errors = (pred_classes != target_4d).float() * boundary_mask
        boundary_loss = boundary_errors.mean()
        
        # 组合损失：基础损失 + 边界损失
        total_loss = base_loss + self.boundary_weight * boundary_loss
        
        return total_loss


class ShapeConstraintLoss(nn.Module):
    """
    形状约束损失：利用种胚、胚乳的形状先验知识
    
    种胚和胚乳通常具有特定的形状特征：
    - 种胚：相对紧凑、球形或椭圆形
    - 胚乳：较大、占据大部分空间
    - 空腔：不规则但通常连通
    
    通过计算形状特征（如紧致度、球形度）来约束预测
    """
    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: 形状约束损失的权重
        """
        super(ShapeConstraintLoss, self).__init__()
        self.weight = weight
    
    def compute_compactness(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算紧致度：体积 / 表面积
        
        紧致度越高，形状越紧凑（球形）
        种胚应该比胚乳更紧凑
        """
        # 计算体积（体素数）
        volume = mask.sum().float()
        
        if volume == 0:
            return torch.tensor(0.0, device=mask.device)
        
        # 计算表面积（边界体素数）
        # 使用3D卷积计算边界
        kernel = torch.ones(1, 1, 3, 3, 3, device=mask.device)
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        convolved = F.conv3d(mask_float, kernel, padding=1)
        # 边界：mask=1但邻居不全为1
        boundary = ((mask_float > 0) & (convolved < 27)).float()
        surface_area = boundary.sum().float()
        
        if surface_area == 0:
            return torch.tensor(0.0, device=mask.device)
        
        # 紧致度 = 体积 / 表面积（归一化）
        compactness = volume / (surface_area + 1e-6)
        return compactness
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算形状约束损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]
            target: 真实标签 [B, H, W, D] 或 [B, 1, H, W, D]
            
        Returns:
            形状约束损失
        """
        # 处理标签形状：确保是 [B, H, W, D] 格式
        if target.dim() == 5:
            if target.shape[1] == 1:
                target = target.squeeze(1)  # [B, 1, H, W, D] -> [B, H, W, D]
            elif target.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target = torch.argmax(target, dim=1)  # one-hot -> 类别索引
        
        # 确保标签是整数类型且维度正确
        if target.dtype != torch.long:
            target = target.long()
        
        # 确保标签是4维 [B, H, W, D]
        if target.dim() != 4:
            raise ValueError(f"形状损失：标签维度错误: 期望4维 [B, H, W, D]，得到 {target.dim()}维，形状: {target.shape}")
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        # 将预测转换为类别索引
        pred_classes = torch.argmax(pred, dim=1)  # [B, H, W, D]
        
        for b in range(batch_size):
            # 对每个类别计算形状约束
            for class_idx in [1, 2]:  # 种胚和胚乳
                # 真实掩膜
                target_mask = (target[b] == class_idx).float()
                # 预测掩膜
                pred_mask = (pred_classes[b] == class_idx).float()
                
                if target_mask.sum() == 0:
                    continue
                
                # 计算紧致度
                target_compactness = self.compute_compactness(target_mask)
                pred_compactness = self.compute_compactness(pred_mask)
                
                # 形状约束：预测的紧致度应该接近真实值
                # 种胚（class_idx=1）应该比胚乳（class_idx=2）更紧凑
                if class_idx == 1:  # 种胚
                    # 种胚应该更紧凑，如果预测不够紧凑则惩罚
                    compactness_loss = F.mse_loss(
                        pred_compactness.unsqueeze(0),
                        target_compactness.unsqueeze(0)
                    )
                else:  # 胚乳
                    # 胚乳可以相对不紧凑，但应该与真实值接近
                    compactness_loss = F.mse_loss(
                        pred_compactness.unsqueeze(0),
                        target_compactness.unsqueeze(0)
                    )
                
                total_loss += compactness_loss
        
        return self.weight * total_loss / (batch_size * 2)  # 归一化


class WeightedClassLoss(nn.Module):
    """
    类别不平衡加权损失：根据类别频率自动调整权重
    
    对于种子分割任务：
    - 背景：最多
    - 胚乳：较多
    - 种胚：中等
    - 空腔：较少（可能最少）
    """
    def __init__(self, base_loss_fn, class_weights: dict = None, 
                 use_inverse_frequency: bool = True):
        """
        Args:
            base_loss_fn: 基础损失函数
            class_weights: 手动指定的类别权重 {class_idx: weight}
            use_inverse_frequency: 是否使用逆频率自动计算权重
        """
        super(WeightedClassLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.use_inverse_frequency = use_inverse_frequency
        self.class_weights = class_weights or {}
    
    def compute_class_weights(self, target: torch.Tensor) -> dict:
        """
        根据类别频率计算权重（逆频率）
        
        Args:
            target: 真实标签 [B, H, W, D]
            
        Returns:
            类别权重字典 {class_idx: weight}
        """
        weights = {}
        total_pixels = target.numel()
        
        for class_idx in [1, 2, 3]:  # 种胚、胚乳、空腔
            class_pixels = (target == class_idx).sum().float()
            if class_pixels > 0:
                # 逆频率权重：总像素数 / (类别数 * 该类像素数)
                frequency = class_pixels / total_pixels
                weights[class_idx] = 1.0 / (frequency + 1e-6)
            else:
                weights[class_idx] = 1.0
        
        # 归一化权重
        max_weight = max(weights.values()) if weights else 1.0
        weights = {k: v / max_weight for k, v in weights.items()}
        
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权类别损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]
            target: 真实标签 [B, H, W, D] 或 [B, 1, H, W, D]
            
        Returns:
            加权后的损失值
        """
        # 准备4维格式的target用于其他计算
        target_4d = target.clone()
        if target_4d.dim() == 5:
            if target_4d.shape[1] == 1:
                target_4d = target_4d.squeeze(1)  # [B, 1, H, W, D] -> [B, H, W, D]
            elif target_4d.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_4d = torch.argmax(target_4d, dim=1)  # one-hot -> 类别索引
        
        if target_4d.dtype != torch.long:
            target_4d = target_4d.long()
        
        # 准备5维格式的target用于MONAI损失函数
        target_for_loss = target.clone()
        if target_for_loss.dim() == 4:
            # 如果是4维，扩展为5维 [B, H, W, D] -> [B, 1, H, W, D]
            target_for_loss = target_for_loss.unsqueeze(1)
        elif target_for_loss.dim() == 5:
            if target_for_loss.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_for_loss = torch.argmax(target_for_loss, dim=1).unsqueeze(1)
            # 如果已经是[B, 1, H, W, D]格式，保持不变
        
        if target_for_loss.dtype != torch.long:
            target_for_loss = target_for_loss.long()
        
        # 计算基础损失（使用5维格式）
        base_loss = self.base_loss_fn(pred, target_for_loss)
        
        # 如果使用逆频率，计算类别权重（使用4维格式）
        if self.use_inverse_frequency:
            class_weights = self.compute_class_weights(target_4d)
        else:
            class_weights = self.class_weights
        
        # 对每个类别计算加权损失
        weighted_loss = 0.0
        batch_size = pred.shape[0]
        
        # 将预测转换为one-hot
        pred_classes = torch.argmax(pred, dim=1)  # [B, H, W, D]
        
        for class_idx in [1, 2, 3]:
            weight = class_weights.get(class_idx, 1.0)
            
            # 计算该类别的分类错误（使用4维格式）
            class_target = (target_4d == class_idx).float()
            class_pred = (pred_classes == class_idx).float()
            
            # 使用交叉熵或MSE计算该类别的损失
            class_loss = F.mse_loss(class_pred, class_target)
            
            weighted_loss += weight * class_loss
        
        # 组合基础损失和加权损失
        total_loss = base_loss + weighted_loss / (batch_size * 3)
        
        return total_loss


class AdvancedCombinedLoss(nn.Module):
    """
    高级组合损失函数：包含所有创新损失
    
    组合：
    1. Dice Loss + Cross-Entropy Loss（基础）
    2. 类别不平衡加权损失
    3. 边界感知损失
    4. 形状约束损失
    """
    def __init__(self, 
                 dice_weight: float = 0.3,
                 ce_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 shape_weight: float = 0.1,
                 class_weight_weight: float = 0.1,
                 num_classes: int = 4,
                 smooth: float = 1e-5,
                 boundary_loss_weight: float = 2.0,
                 boundary_width: int = 2,
                 use_boundary_loss: bool = True,
                 use_shape_loss: bool = True,
                 use_class_weight: bool = True):
        """
        Args:
            dice_weight: Dice Loss权重
            ce_weight: Cross-Entropy Loss权重
            boundary_weight: 边界感知损失权重
            shape_weight: 形状约束损失权重
            class_weight_weight: 类别加权损失权重
            num_classes: 类别数
            smooth: 平滑参数
            boundary_loss_weight: 边界损失的额外权重倍数
            boundary_width: 边界宽度
            use_boundary_loss: 是否使用边界损失
            use_shape_loss: 是否使用形状损失
            use_class_weight: 是否使用类别加权
        """
        super(AdvancedCombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.shape_weight = shape_weight
        self.class_weight_weight = class_weight_weight
        
        self.use_boundary_loss = use_boundary_loss
        self.use_shape_loss = use_shape_loss
        self.use_class_weight = use_class_weight
        
        # 基础损失：Dice + CE
        self.base_dice_ce_loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
        )
        
        # 类别不平衡加权损失
        if use_class_weight:
            self.weighted_loss = WeightedClassLoss(
                base_loss_fn=self.base_dice_ce_loss,
                use_inverse_frequency=True
            )
        else:
            self.weighted_loss = None
        
        # 边界感知损失
        if use_boundary_loss:
            self.boundary_loss = BoundaryAwareLoss(
                base_loss_fn=self.base_dice_ce_loss,
                boundary_weight=boundary_loss_weight,
                boundary_width=boundary_width
            )
        else:
            self.boundary_loss = None
        
        # 形状约束损失
        if use_shape_loss:
            self.shape_loss = ShapeConstraintLoss(weight=shape_weight)
        else:
            self.shape_loss = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]，logits（未应用softmax，损失函数内部会处理）
            target: 真实标签 [B, H, W, D] 或 [B, 1, H, W, D]，类别索引格式
            
        Returns:
            总损失值
        """
        # 处理标签形状：MONAI的DiceCELoss在to_onehot_y=True时，如果target不是one-hot编码，
        # 期望target是[B, 1, H, W, D]格式（5维），而不是[B, H, W, D]（4维）
        target_for_loss = target.clone()
        if target_for_loss.dim() == 4:
            # 如果是4维，扩展为5维 [B, H, W, D] -> [B, 1, H, W, D]
            target_for_loss = target_for_loss.unsqueeze(1)
        elif target_for_loss.dim() == 5:
            if target_for_loss.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_for_loss = torch.argmax(target_for_loss, dim=1).unsqueeze(1)
            # 如果已经是[B, 1, H, W, D]格式，保持不变
        
        # 确保标签是整数类型
        if target_for_loss.dtype != torch.long:
            target_for_loss = target_for_loss.long()
        
        # 为了其他损失函数计算，准备4维格式的target
        target_4d = target.clone()
        if target_4d.dim() == 5:
            if target_4d.shape[1] == 1:
                target_4d = target_4d.squeeze(1)  # [B, 1, H, W, D] -> [B, H, W, D]
            else:
                target_4d = torch.argmax(target_4d, dim=1)  # one-hot -> 类别索引
        
        if target_4d.dtype != torch.long:
            target_4d = target_4d.long()
        
        total_loss = 0.0
        
        # 1. 基础损失：Dice + CE
        # 使用5维格式 [B, 1, H, W, D] 传递给MONAI损失函数
        base_loss = self.base_dice_ce_loss(pred, target_for_loss)
        total_loss += (self.dice_weight + self.ce_weight) * base_loss
        
        # 2. 类别不平衡加权损失（使用4维格式）
        if self.use_class_weight and self.weighted_loss is not None:
            weighted_loss = self.weighted_loss(pred, target_4d)
            total_loss += self.class_weight_weight * weighted_loss
        
        # 3. 边界感知损失（使用4维格式）
        if self.use_boundary_loss and self.boundary_loss is not None:
            boundary_loss = self.boundary_loss(pred, target_4d)
            total_loss += self.boundary_weight * boundary_loss
        
        # 4. 形状约束损失（使用4维格式）
        if self.use_shape_loss and self.shape_loss is not None:
            shape_loss = self.shape_loss(pred, target_4d)
            total_loss += self.shape_weight * shape_loss
        
        return total_loss


# 保持向后兼容
class CombinedLoss(nn.Module):
    """
    组合损失函数：Dice Loss + Cross-Entropy Loss（原始版本，保持向后兼容）
    """
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5,
                 num_classes: int = 4, smooth: float = 1e-5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # 使用MONAI的DiceCELoss
        self.dice_ce_loss = DiceCELoss(
            include_background=False,  # 不包括背景
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]，logits（未应用softmax，损失函数内部会处理）
            target: 真实标签 [B, H, W, D] 或 [B, 1, H, W, D]，类别索引格式
            
        Returns:
            损失值
        """
        # MONAI的DiceCELoss在to_onehot_y=True时，如果target不是one-hot编码，
        # 期望target是[B, 1, H, W, D]格式（5维）
        target_for_loss = target.clone()
        if target_for_loss.dim() == 4:
            # 如果是4维，扩展为5维 [B, H, W, D] -> [B, 1, H, W, D]
            target_for_loss = target_for_loss.unsqueeze(1)
        elif target_for_loss.dim() == 5:
            if target_for_loss.shape[1] > 1:
                # 如果是one-hot格式，转换为类别索引
                target_for_loss = torch.argmax(target_for_loss, dim=1).unsqueeze(1)
            # 如果已经是[B, 1, H, W, D]格式，保持不变
        
        # 确保标签是整数类型
        if target_for_loss.dtype != torch.long:
            target_for_loss = target_for_loss.long()
        
        return self.dice_ce_loss(pred, target_for_loss)



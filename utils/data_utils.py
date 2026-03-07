"""
数据工具函数：处理nii.gz格式的3D医学图像数据
"""
import os
from marshal import version

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional
import random
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ToTensord,
    Spacingd,
    MapTransform,
    RandZoomd,
    DataStatsd
)

"""songar:20260120"""
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
    val_list = valid_samples[n_train:n_train + n_val]
    test_list = valid_samples[n_train + n_val:]

    print(f"数据集划分:")
    print(f"  训练集: {len(train_list)} 个样本")
    print(f"  验证集: {len(val_list)} 个样本")
    print(f"  测试集: {len(test_list)} 个样本")

    return train_list, val_list, test_list
"""songar:20260120"""

def load_nii_gz(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    加载nii.gz文件
    
    Args:
        file_path: nii.gz文件路径
        
    Returns:
        data: 图像数据数组
        header: 文件头信息字典
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    info = {
        'shape': data.shape,
        'affine': affine,
        'header': header,
        'spacing': header.get_zooms()[:3] if len(header.get_zooms()) >= 3 else (1.0, 1.0, 1.0),
    }
    
    return data, info


def save_nii_gz(data: np.ndarray, file_path: str, affine: Optional[np.ndarray] = None, 
                header: Optional = None):
    """
    保存为nii.gz文件
    
    Args:
        data: 图像数据数组
        file_path: 保存路径
        affine: 仿射变换矩阵（可选）
        header: 文件头（可选）
    """
    if affine is None:
        affine = np.eye(4)
    
    nii_img = nib.Nifti1Image(data, affine, header)
    nib.save(nii_img, file_path)


def check_label_completeness(label_path: str, required_classes: List[int] = [1, 2, 3]) -> Dict:
    """
    检查标签文件的完整性
    
    Args:
        label_path: 标签文件路径
        required_classes: 需要的类别列表 [种胚, 胚乳, 空腔]
        
    Returns:
        包含检查结果的字典
    """
    try:
        label_data, _ = load_nii_gz(label_path)
        unique_labels = np.unique(label_data)
        
        # 检查哪些类别存在
        present_classes = {}
        for class_idx in required_classes:
            present_classes[class_idx] = class_idx in unique_labels
        
        # 判断是否完整（至少需要2个类别，且必须包含胚乳或种胚）
        has_embryo = present_classes.get(1, False)  # 种胚
        has_endosperm = present_classes.get(2, False)  # 胚乳
        has_cavity = present_classes.get(3, False)  # 空腔
        
        # 完整性判断：至少需要种胚+胚乳，或者种胚+空腔，或者胚乳+空腔
        # 不能只有单一类别
        num_present = sum(present_classes.values())
        is_complete = num_present >= 2 and (has_embryo or has_endosperm)
        
        return {
            'is_complete': is_complete,
            'present_classes': present_classes,
            'num_classes': num_present,
            'has_embryo': has_embryo,
            'has_endosperm': has_endosperm,
            'has_cavity': has_cavity,
        }
    except Exception as e:
        return {
            'is_complete': False,
            'error': str(e),
            'present_classes': {},
            'num_classes': 0,
            'has_embryo': False,
            'has_endosperm': False,
            'has_cavity': False,
        }


def merge_labels(embryo_endosperm_path: str, cavity_path: str, 
                 output_path: str, embryo_label: int = 1, 
                 endosperm_label: int = 2, cavity_label: int = 3):
    """
    合并分散的标注文件为统一的多类别标签
    
    Args:
        embryo_endosperm_path: 胚与胚乳标注文件路径
        cavity_path: 空腔标注文件路径
        output_path: 输出合并后的标签文件路径
        embryo_label: 种胚标签值
        endosperm_label: 胚乳标签值
        cavity_label: 空腔标签值
    """
    # 加载胚与胚乳标注
    embryo_endosperm_data = None
    if os.path.exists(embryo_endosperm_path):
        embryo_endosperm_data, info = load_nii_gz(embryo_endosperm_path)
        affine = info['affine']
        header = info['header']
    else:
        raise FileNotFoundError(f"找不到胚与胚乳标注文件: {embryo_endosperm_path}")
    
    # 加载空腔标注
    cavity_data = None
    if os.path.exists(cavity_path):
        cavity_data, _ = load_nii_gz(cavity_path)
    else:
        print(f"警告: 找不到空腔标注文件: {cavity_path}，将只使用胚与胚乳标注")
        cavity_data = np.zeros_like(embryo_endosperm_data)
    
    # 确保两个标注尺寸一致
    if embryo_endosperm_data.shape != cavity_data.shape:
        print(f"警告: 标注尺寸不一致，将调整空腔标注尺寸")
        from scipy.ndimage import zoom
        zoom_factors = [s1/s2 for s1, s2 in zip(embryo_endosperm_data.shape, cavity_data.shape)]
        cavity_data = zoom(cavity_data, zoom_factors, order=0)
    
    # 合并标签
    # 假设胚与胚乳标注中：1=胚，2=胚乳（需要根据实际标注调整）
    merged_label = np.zeros_like(embryo_endosperm_data, dtype=np.uint8)
    
    # 提取种胚（通常标注值较小或特定值）
    # 这里需要根据实际标注格式调整
    if np.max(embryo_endosperm_data) > 0:
        # 简单策略：假设标注中较小的值为胚，较大的值为胚乳
        # 实际使用时需要根据ITK-SNAP标注的具体格式调整
        unique_values = np.unique(embryo_endosperm_data[embryo_endosperm_data > 0])
        if len(unique_values) >= 2:
            # 假设有两个类别
            sorted_values = np.sort(unique_values)
            merged_label[embryo_endosperm_data == sorted_values[0]] = embryo_label
            merged_label[embryo_endosperm_data == sorted_values[1]] = endosperm_label
        else:
            # 只有一个类别，假设是胚乳
            merged_label[embryo_endosperm_data > 0] = endosperm_label
    
    # 添加空腔
    merged_label[cavity_data > 0] = cavity_label
    
    # 保存合并后的标签
    save_nii_gz(merged_label, output_path, affine, header)
    
    return merged_label

# =========================
# Custom: Randomly select a transform for spacing(most stable, not dependent on MONAI version differences)
# =========================
class RandSpacingSelectd(MapTransform):
    """
    Randomly select a pixdim for every sample, and call the Spacingd of MONAI
    Using for simulating the differences in voxel spacing caused by different CT equipment/protocols.
    """
    def __init__(
        self,
        keys,
        pixdim_candidates: List[Tuple[float, float, float]],
        mode=("bilinear", "nearest"),
        prob: float = 1.0,
    ):
        super().__init__(keys)
        self.pixdim_candidates = pixdim_candidates
        self.mode = mode
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if random.random() > self.prob:
            return d
        pixdim = random.choice(self.pixdim_candidates)
        sp = Spacingd(keys=self.keys, pixdim=pixdim, mode=self.mode)
        return sp(d)


# =========================
# 2) Tool: get config
# =========================
def _get(cfg: Dict[str, Any], path: str, default):
    """
    path like "augmentation.rotate90.prob"
    """
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =========================
# 3) V1-V4 new augment
# =========================
def get_train_transforms_augmented(config: Dict[str, Any]):
    """
    new augment method
    - version: v1/v2/v3/v4
    - for train transform;
    """
    img_size = config["model"]["img_size"]
    aug = config.get("augmentation", {})
    version = str(aug.get("version", "v1")).lower()

    # --- default parameters (It can also be overridden in Yaml ---
    spacing_candidates = aug.get(
        "spacing_candidates_mm",
        [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
    )

    # base Preprocessing
    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    # ===== Resolution Strategy =====
    # v1: Fixed spacing (consistent with your old code's approach
    # v2+: Random spacing (Key: Simulating different devices/layer/thicknesses/voxel spacings
    if version == "v1":
        t += [
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ]
    else:
        t += [
            RandSpacingSelectd(
                keys=["image", "label"],
                pixdim_candidates=spacing_candidates,
                prob=float(aug.get("rand_spacing_prob", 0.8)),
                mode=("bilinear", "nearest"),
            )
        ]

    # Intensity Normalization (CT:HU ->[0,1])
    # Note: a_min/a_max can be adjusted here according to your task; window jitter can also be added
    t += [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=float(aug.get("a_min", -1000)),
            a_max=float(aug.get("a_max", 1000)),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=img_size, mode=("trilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
    ]

    # ===== V1: Basic geometry + light intensity (an upgraded version of your existing enhancement)=====
    # Using RandAffined instead of a bunch of separate rotate/scale/translate operations provides a more unified approach.
    if version == "v1":
        t += [
            RandAffined(
                keys=["image", "label"],
                prob=float(aug.get("affine_prob", 0.7)),
                rotate_range=(
                    float(aug.get("rotate_x", 0.0)),
                    float(aug.get("rotate_y", 0.0)),
                    float(aug.get("rotate_z", 0.0)),
                ),  # 0.26 approximately equal to 15 angle
                scale_range=(float(aug.get("scale", 0.10)),) * 3,
                translate_range=(float(aug.get("translate", 10.0)),) * 3,  # µ¥Î»ÊÇÏñËØ/ÌåËØ
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandRotate90d(keys=["image", "label"], prob=float(aug.get("rotate90_prob", 0.5)), spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=1),
            RandShiftIntensityd(keys=["image"], offsets=float(aug.get("shift_offsets", 0.10)), prob=float(aug.get("shift_prob", 0.5))),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(t)

    # ===== V2: Resolution domain + low-resolution simulation + minor window/contrast perturbation =====
    if version == "v2":
        t += [
            RandAffined(
                keys=["image", "label"],
                prob=float(aug.get("affine_prob", 0.8)),
                rotate_range=(0.26, 0.26, 0.26),   # ~15¡ã
                scale_range=(0.10, 0.10, 0.10),
                translate_range=(10.0, 10.0, 10.0),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=1),

            # Low resolution/thick layers: Smooth first (can be considered a simplified version of low-resolution simulation)
            RandGaussianSmoothd(
                keys=["image"],
                prob=float(aug.get("smooth_prob", 0.25)),
                sigma_x=aug.get("smooth_sigma", (0.0, 1.2)),
                sigma_y=aug.get("smooth_sigma", (0.0, 1.2)),
                sigma_z=aug.get("smooth_sigma", (0.0, 1.2)),
            ),

            # Low noise (simulated dose/equipment differences)
            RandGaussianNoised(
                keys=["image"],
                prob=float(aug.get("gauss_noise_prob", 0.20)),
                mean=0.0,
                std=float(aug.get("gauss_noise_std", 0.02)),
            ),

            # Contrast/Gamma Slight Perturbation
            RandAdjustContrastd(
                keys=["image"],
                prob=float(aug.get("contrast_prob", 0.20)),
                gamma=aug.get("gamma_range", (0.9, 1.1)),
            ),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(t)

    # ===== Device domain (reconstructed kernel/stronger noise) + stronger geometric perturbation =====
    if version == "v3":
        t += [
            RandAffined(
                keys=["image", "label"],
                prob=float(aug.get("affine_prob", 0.85)),
                rotate_range=(0.35, 0.35, 0.35),  # ~20¡ã
                scale_range=(0.12, 0.12, 0.12),
                translate_range=(12.0, 12.0, 12.0),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=1),

            # Ä£ÄâÖØ½¨ºË£º¸ü¶àÆ½»¬£¨soft kernel£©+ Ç¿Ò»µãÔëÉù
            RandGaussianSmoothd(
                keys=["image"],
                prob=float(aug.get("smooth_prob", 0.35)),
                sigma_x=aug.get("smooth_sigma", (0.0, 1.5)),
                sigma_y=aug.get("smooth_sigma", (0.0, 1.5)),
                sigma_z=aug.get("smooth_sigma", (0.0, 1.5)),
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=float(aug.get("gauss_noise_prob", 0.30)),
                mean=0.0,
                std=float(aug.get("gauss_noise_std", 0.03)),
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=float(aug.get("contrast_prob", 0.25)),
                gamma=aug.get("gamma_range", (0.8, 1.2)),
            ),

            # Intensity scaling (simulating different calibrations/dosages)
            RandScaleIntensityd(
                keys=["image"],
                prob=float(aug.get("scale_int_prob", 0.25)),
                factors=aug.get("scale_int_factors", (0.9, 1.1)),
            ),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(t)

    # ===== V4: Strong Regularization (Stronger Geometry + Stronger Device Domain Perturbation) =====
    if version == "v4":
        t += [
            RandAffined(
                keys=["image", "label"],
                prob=float(aug.get("affine_prob", 0.90)),
                rotate_range=(0.52, 0.52, 0.52),  # ~30¡ã
                scale_range=(0.15, 0.15, 0.15),
                translate_range=(15.0, 15.0, 15.0),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=float(aug.get("flip_prob", 0.5)), spatial_axis=1),

            # Stronger low-resolution simulation (using a larger range for smoothing)
            RandGaussianSmoothd(
                keys=["image"],
                prob=float(aug.get("smooth_prob", 0.45)),
                sigma_x=aug.get("smooth_sigma", (0.0, 2.0)),
                sigma_y=aug.get("smooth_sigma", (0.0, 2.0)),
                sigma_z=aug.get("smooth_sigma", (0.0, 2.0)),
            ),
            # Stronger noise
            RandGaussianNoised(
                keys=["image"],
                prob=float(aug.get("gauss_noise_prob", 0.40)),
                mean=0.0,
                std=float(aug.get("gauss_noise_std", 0.05)),
            ),
            # Higher contrast/gamma
            RandAdjustContrastd(
                keys=["image"],
                prob=float(aug.get("contrast_prob", 0.35)),
                gamma=aug.get("gamma_range", (0.7, 1.4)),
            ),
            RandScaleIntensityd(
                keys=["image"],
                prob=float(aug.get("scale_int_prob", 0.35)),
                factors=aug.get("scale_int_factors", (0.85, 1.15)),
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=float(aug.get("shift_offsets", 0.12)),
                prob=float(aug.get("shift_prob", 0.5)),
            ),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(t)

    # Fallback: If the version is unknown, use v2.
    # aug["version"] = "v2"
    return get_train_transforms_augmented({**config, "augmentation": aug})


def build_train_transforms(config):
    """
    augmentation.enabled:
    is flase: get_train_transforms
    if true: get_train_transforms_v2/v3/v4
    """
    img_size = config["model"]["img_size"]
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("enabled"):
        if version == "default":
            print('Using default augmentation!')
            return get_train_transforms(img_size)
        elif version in ["v1", "v2", "v3", "v4"]:
            # v1-v4
            print(f'Using new augmentation ({version})!')
            return get_train_transforms_augmented(config)
        else:
            # default
            print('Invalid version specified, using default augmentation.')
            return get_train_transforms(img_size)
    else:
        print('Augmentation is disabled.')
        return get_noaug_train_transforms(img_size)
    # # ===== key judge =====
    # if aug_cfg.get("enabled", False) is True:
    #     # using new augment v1-v4
    #     print('Using new augmentation!')
    #     return get_train_transforms_augmented(config)
    # else:
    #     # default augment
    #     return get_train_transforms(img_size)


def get_train_transforms(img_size: List[int] = [96, 96, 96]):
    """获取训练时的数据增强变换"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 添加 Resize 确保图像尺寸至少达到 img_size，避免内存溢出
        Resized(
            keys=["image", "label"],
            spatial_size=img_size,
            mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=1,
            num_samples=1,  # 改为1，避免返回列表
            image_key="image",
            image_threshold=0,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            spatial_axes=(0, 1),
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.5,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.5,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.1,
            prob=0.5,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_noaug_train_transforms(img_size: List[int] = [96, 96, 96]):
    """Only transformer no Augmentation"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 添加 Resize 确保图像尺寸至少达到 img_size，避免内存溢出
        Resized(
            keys=["image", "label"],
            spatial_size=img_size,
            mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻
        ),
        # RandCropByPosNegLabeld(   # no positive influence
        #     keys=["image", "label"],
        #     label_key="label",
        #     spatial_size=img_size,
        #     pos=1,
        #     neg=1,
        #     num_samples=1,  # 改为1，避免返回列表
        #     image_key="image",
        #     image_threshold=0,
        # ),
        # RandShiftIntensityd(
        #     keys=["image"],
        #     offsets=0.1,
        #     prob=0.5,
        # ),
        # RandZoomd(
        #     keys=["image", "label"],
        #     prob=0.5,
        #     min_zoom=0.9,
        #     max_zoom=1.1,
        # ),
        # RandAffined(
        #     keys=["image", "label"],
        #     prob=0.5,
        #     rotate_range=(0, 0),
        #     shear_range=(0, 0),
        #     scale_range=(0.9, 1.1),
        #     translate_range=(0, 0),
        # ),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms(img_size: List[int] = [96, 96, 96]):
    """获取验证时的数据变换"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 添加 Resize 确保图像尺寸一致，避免内存溢出
        Resized(
            keys=["image", "label"],
            spatial_size=img_size,
            mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_inference_transforms(img_size: List[int] = [96, 96, 96]):
    """获取推理时的数据变换（只处理image，不需要label）"""
    return Compose([
        LoadImaged(keys=["image"]),  # 只加载image
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        # 添加 Resize 确保图像尺寸一致
        Resized(
            keys=["image"],
            spatial_size=img_size,
            mode="trilinear",  # 图像用三线性插值
        ),
        ToTensord(keys=["image"]),
    ])


class SeedDataset(Dataset):
    """种子CT数据集"""
    def __init__(self, data_list: list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]

        if self.transforms:
            data_dict = self.transforms(data_dict)
            # 处理RandCropByPosNegLabeld可能返回的列表格式
            if isinstance(data_dict, list):
                data_dict = data_dict[0]  # 取第一个元素

        return data_dict

"""
数据预处理脚本：合并分散的标注文件为统一的多类别标签
"""
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import load_nii_gz, save_nii_gz, merge_labels


def find_raw_files(raw_dir: str) -> dict:
    """
    找到所有原始数据文件
    
    Args:
        raw_dir: 原始数据目录
        
    Returns:
        匹配字典 {样本ID: 原始文件路径}
    """
    raw_files = {}
    if not os.path.exists(raw_dir):
        print(f"警告: 原始数据目录不存在: {raw_dir}")
        return raw_files
    
    for f in os.listdir(raw_dir):
        if f.endswith('.nii.gz'):
            sample_id = f.replace('.nii.gz', '')
            raw_files[sample_id] = os.path.join(raw_dir, f)
    
    return raw_files


def find_annotation_files(annotation_dir: str, annotation_subdir: str) -> dict:
    """
    找到标注文件
    
    Args:
        annotation_dir: 标注数据根目录
        annotation_subdir: 标注子目录名称（如 "yumi-胚与胚乳"）
        
    Returns:
        匹配字典 {样本ID: 标注文件路径}
    """
    annotation_files = {}
    annotation_path = os.path.join(annotation_dir, annotation_subdir)
    
    if not os.path.exists(annotation_path):
        print(f"警告: 标注目录不存在: {annotation_path}")
        return annotation_files
    
    for f in os.listdir(annotation_path):
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            sample_id = f.replace('.nii.gz', '').replace('.nii', '')
            file_path = os.path.join(annotation_path, f)
            # 优先使用.nii.gz，如果没有则使用.nii
            if f.endswith('.nii.gz') or (sample_id not in annotation_files):
                annotation_files[sample_id] = file_path
    
    return annotation_files


def preprocess_all_data(config_path: str = "configs/config.yaml", 
                       filter_incomplete: bool = True):
    """
    预处理所有数据
    
    Args:
        config_path: 配置文件路径
        filter_incomplete: 是否过滤不完整的样本（只有单一类别的样本）
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    raw_dir = data_config['raw_data_root']
    annotation_dir = data_config['annotation_root']
    processed_dir = data_config['processed_data_root']
    
    # 创建输出目录
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'labels'), exist_ok=True)
    
    # 找到所有原始数据文件
    raw_files = find_raw_files(raw_dir)
    
    # 找到标注文件（只从标注目录查找，不从原始数据目录查找）
    embryo_endosperm_files = find_annotation_files(
        annotation_dir, data_config['embryo_endosperm_dir']
    )
    cavity_files = find_annotation_files(
        annotation_dir, data_config['cavity_dir']
    )
    
    print(f"\n找到原始数据文件: {len(raw_files)} 个")
    print(f"找到胚与胚乳标注文件: {len(embryo_endosperm_files)} 个")
    print(f"找到空腔标注文件: {len(cavity_files)} 个")
    
    # 处理每个样本
    processed_count = 0
    skipped_count = 0
    incomplete_count = 0
    
    for sample_id in tqdm(raw_files.keys(), desc="处理数据"):
        raw_path = raw_files[sample_id]
        
        # 检查是否有对应的标注
        embryo_endosperm_path = embryo_endosperm_files.get(sample_id)
        cavity_path = cavity_files.get(sample_id)
        
        if embryo_endosperm_path is None:
            print(f"警告: 样本 {sample_id} 没有胚与胚乳标注，跳过")
            skipped_count += 1
            continue
        
        # 验证路径是否正确（确保是标注文件，不是CT源文件）
        if embryo_endosperm_path == raw_path:
            print(f"错误: 样本 {sample_id} 的胚与胚乳标注路径与原始数据路径相同，跳过")
            print(f"  原始数据路径: {raw_path}")
            print(f"  标注路径: {embryo_endosperm_path}")
            skipped_count += 1
            continue
        
        try:
            # 合并标注
            output_label_path = os.path.join(processed_dir, 'labels', f'{sample_id}.nii.gz')
            if not os.path.exists(output_label_path):
                # 打印调试信息（前几个样本）
                if processed_count < 3:
                    print(f"\n处理样本 {sample_id}:")
                    print(f"  胚与胚乳标注: {embryo_endosperm_path}")
                    print(f"  空腔标注: {cavity_path if cavity_path else '(无)'}")
                
                merge_labels(
                    embryo_endosperm_path,
                    cavity_path if cavity_path else '',
                    output_label_path,
                    embryo_label=data_config['label_mapping']['embryo'],
                    endosperm_label=data_config['label_mapping']['endosperm'],
                    cavity_label=data_config['label_mapping']['cavity']
                )
            
            # 检查标签完整性（如果启用过滤）
            if filter_incomplete:
                from utils.data_utils import check_label_completeness
                completeness = check_label_completeness(
                    output_label_path, 
                    required_classes=[1, 2, 3]  # 种胚、胚乳、空腔
                )
                
                if not completeness['is_complete']:
                    # 删除不完整的标签文件
                    if os.path.exists(output_label_path):
                        os.remove(output_label_path)
                    print(f"过滤: 样本 {sample_id} 标签不完整 "
                          f"(种胚:{completeness['has_embryo']}, "
                          f"胚乳:{completeness['has_endosperm']}, "
                          f"空腔:{completeness['has_cavity']})，已跳过")
                    incomplete_count += 1
                    skipped_count += 1
                    continue
            
            # 复制原始图像到processed目录
            output_image_path = os.path.join(processed_dir, 'images', f'{sample_id}.nii.gz')
            if not os.path.exists(output_image_path):
                import shutil
                shutil.copy2(raw_path, output_image_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个样本")
    print(f"跳过: {skipped_count} 个样本")
    if filter_incomplete:
        print(f"  - 其中不完整样本: {incomplete_count} 个")
    print(f"处理后的数据保存在: {processed_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="预处理种子CT数据")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--filter-incomplete", action="store_true", default=True,
                       help="过滤不完整的样本（只有单一类别的样本）")
    parser.add_argument("--no-filter", action="store_false", dest="filter_incomplete",
                       help="不过滤不完整的样本")
    args = parser.parse_args()
    
    preprocess_all_data(args.config, filter_incomplete=args.filter_incomplete)



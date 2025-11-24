"""
数据集划分工具
用于创建训练集/验证集的CSV文件
"""

import csv
from pathlib import Path
from random import Random
from typing import Tuple, List, Dict, Optional


def create_train_val_split(
    raw_audio_dir: Path,
    train_csv_path: Path,
    val_csv_path: Path,
    feature_dir_name: str,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    dataset_name: Optional[str] = "Unknown Dataset",
    xlsr: bool = True
) -> Tuple[Path, Path]:
    """
    创建训练集和验证集的CSV文件
    
    Args:
        raw_audio_dir: 原始音频文件目录 (包含Control和Dementia子文件夹)
        train_csv_path: 训练集CSV输出路径
        val_csv_path: 验证集CSV输出路径
        feature_dir_name: 特征目录名称 (例如: "Address_xlsr_features" 或 "features")
        train_ratio: 训练集比例 (默认: 0.8)
        random_seed: 随机种子 (默认: 42)
        dataset_name: 数据集名称，用于打印信息 (可选)
        xlsr: 是否使用XLSR特征模式 (True: xlsr_path/.xlsr.pt, False: egemaps_path/.egemaps.pt, 默认: True)
    
    Returns:
        Tuple[Path, Path]: (训练集CSV路径, 验证集CSV路径)
    """
    
    # 确保输出目录存在
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    val_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查原始音频目录是否存在
    if not raw_audio_dir.exists():
        raise FileNotFoundError(f"原始音频目录不存在: {raw_audio_dir}")
    
    # 收集所有音频文件
    all_samples = []
    
    # 处理 Control 组
    control_dir = raw_audio_dir / "Control"
    if control_dir.exists():
        # 支持 .wav 和 .mp3 格式
        audio_files = list(control_dir.glob("*.wav")) + list(control_dir.glob("*.mp3"))
        for audio_file in sorted(audio_files):
            session_id = audio_file.stem
            all_samples.append({
                'session_id': session_id,
                'ad': 0
            })
    
    # 处理 Dementia 组
    dementia_dir = raw_audio_dir / "Dementia"
    if dementia_dir.exists():
        # 支持 .wav 和 .mp3 格式
        audio_files = list(dementia_dir.glob("*.wav")) + list(dementia_dir.glob("*.mp3"))
        for audio_file in sorted(audio_files):
            session_id = audio_file.stem
            all_samples.append({
                'session_id': session_id,
                'ad': 1
            })
    
    if len(all_samples) == 0:
        raise ValueError(f"在 {raw_audio_dir} 中未找到任何音频文件")
    
    # 按类别分开处理
    control_samples = [s for s in all_samples if s['ad'] == 0]
    dementia_samples = [s for s in all_samples if s['ad'] == 1]
    
    # 创建随机数生成器
    rdm = Random(random_seed)
    
    # 打乱
    rdm.shuffle(control_samples)
    rdm.shuffle(dementia_samples)
    
    # 划分 Control
    n_control_train = int(len(control_samples) * train_ratio)
    control_train = control_samples[:n_control_train]
    control_val = control_samples[n_control_train:]
    
    # 划分 Dementia
    n_dementia_train = int(len(dementia_samples) * train_ratio)
    dementia_train = dementia_samples[:n_dementia_train]
    dementia_val = dementia_samples[n_dementia_train:]
    
    # 合并
    train_samples = control_train + dementia_train
    val_samples = control_val + dementia_val
    
    # 再次打乱
    rdm.shuffle(train_samples)
    rdm.shuffle(val_samples)
    
    # 根据特征类型设置列名和文件扩展名
    if xlsr:
        feature_col = 'xlsr_path'
        feature_ext = '.xlsr.pt'
    else:
        feature_col = 'egemaps_path'
        feature_ext = '.egemaps.pt'
    
    # 生成训练集 CSV
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in train_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # 生成验证集 CSV
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in val_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # 打印统计信息
    print(f"============= {dataset_name} Train/Val Split Complete! =============")
    print(f"训练集: {len(train_samples)} 样本 (Control: {len(control_train)}, Dementia: {len(dementia_train)})")
    print(f"验证集: {len(val_samples)} 样本 (Control: {len(control_val)}, Dementia: {len(dementia_val)})")
    print(f"\n训练集CSV 路径: {train_csv_path}")
    print(f"验证集CSV 路径: {val_csv_path}")

    return train_csv_path, val_csv_path

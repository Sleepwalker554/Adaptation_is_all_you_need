import csv
from pathlib import Path
import warnings

import librosa
import numpy as np
import torch
from opensmile.core.smile import Smile
from opensmile.core.define import FeatureSet, FeatureLevel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from config import FEAT_SEQ_LEN, SAMPLING_RATE

# ====== 路径配置 ======
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_audio(file_path: str, sampling_rate: int) -> np.ndarray:
    """
    加载音频文件并重采样
    
    参数:
        file_path: 音频文件路径
        sampling_rate: 目标采样率
    
    返回:
        audio_array: numpy array, 单声道音频
    """
    # 使用 librosa 加载（支持 MP3 和 WAV）
    array, _ = librosa.load(file_path, sr=sampling_rate, res_type="kaiser_best")
    # 确保单声道
    array = librosa.to_mono(array)
    # 转换为 float32
    array = np.float32(array)
    return array


# ====== 数据集类 ======
class CsvDataset(Dataset):
    """
    简单的 CSV 数据集类
    
    从 CSV 读取 session_id 和音频路径
    """
    
    def __init__(self, csv_path: Path, raw_audio_dir: Path):
        super().__init__()
        
        self.csv_path = csv_path
        self.raw_audio_dir = raw_audio_dir
        self.data = []
        
        # 读取 CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row['session_id']
                egemaps_path = row['egemaps_path']
                
                # 构建音频路径（从 session_id 推断）
                # 我们从 CSV 的第三列（ad）来判断音频在 Control 还是 Dementia 文件夹
                ad = int(row['ad'])
                if ad == 0:
                    folder = "Control"
                else:
                    folder = "Dementia"
                
                # 使用传入的原始音频目录
                audio_path = self.raw_audio_dir / folder / f"{session_id}.wav"
                
                self.data.append({
                    'session_id': session_id,
                    'audio_path': str(audio_path.relative_to(PROJECT_ROOT)),
                    'egemaps_path': egemaps_path,
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['audio_path'], item['egemaps_path'], item['session_id']


# ====== 特征提取函数 ======
def extract_features_from_csv(csv_path: Path, raw_audio_dir: Path):
    """
    从 CSV 提取特征
    
    参数:
        csv_path: CSV 文件路径
        raw_audio_dir: 原始音频文件目录（包含 Control 和 Dementia 子文件夹）
    """
    # 创建数据集
    dataset = CsvDataset(csv_path, raw_audio_dir=raw_audio_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # 逐个处理
        shuffle=False,
        num_workers=0,     # OpenSMILE 不支持多进程
        persistent_workers=False,
    )
    
    print(f"共 {len(dataset)} 个音频文件")
    print()
    
    # 初始化 OpenSMILE
    smile_lld = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.LowLevelDescriptors,
    )
    
    # 统计提取的特征数量
    extracted = 0
    skipped = 0
    
    # 忽略 OpenSMILE 警告
    warnings.simplefilter('ignore')
    
    # 提取特征
    for audio_path, egemaps_path, session_id in tqdm(dataloader, desc="提取中"):
        
        # 转换为绝对路径
        audio_path_abs = PROJECT_ROOT / audio_path
        egemaps_path_abs = PROJECT_ROOT / egemaps_path
        
        # 检查是否已存在
        if egemaps_path_abs.exists():
            skipped += 1
            continue
        
        # 检查音频文件是否存在
        if not audio_path_abs.exists():
            print(f"\n⚠️  音频文件不存在: {audio_path}")
            continue
        
        try:
            # 1. 加载音频
            audio_np = load_audio(str(audio_path_abs), sampling_rate=SAMPLING_RATE)
            
            # 2. 音频分段
            # 计算可用长度（去除末尾不足一段的部分）
            usable_length = (audio_np.shape[0] // FEAT_SEQ_LEN) * FEAT_SEQ_LEN
            
            if usable_length == 0:
                print(f"\n音频太短，跳过提取: {session_id}")
                continue
            
            # 截取并分段
            audio_segments = np.split(audio_np[:usable_length], FEAT_SEQ_LEN)
            
            # 3. 逐段提取 eGeMAPS 特征
            egemaps_list = []
            for segment in audio_segments:
                # OpenSMILE 处理
                _, _, features = smile_lld.process(segment, SAMPLING_RATE)
                # features shape: (1, 25) - 取第一帧
                feat_np = np.array(features[0, :], dtype=np.float32)
                egemaps_list.append(torch.from_numpy(feat_np))
            
            # 4. 堆叠为 (FEAT_SEQ_LEN, 25) 的 Tensor
            egemaps = torch.stack(egemaps_list, dim=0)
 
            # 5. 确保输出目录存在
            egemaps_path_abs.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(egemaps, egemaps_path_abs)
            
            extracted += 1
            
        except Exception as e:
            print(f"\n❌ 提取失败 {session_id}: {e}")
            continue
    
    # 恢复警告
    warnings.simplefilter('always')
    
    # 打印统计
    print(f"\n============= 提取完成！ =============")
    print(f"成功提取: {extracted} 个")
    print(f"已存在跳过: {skipped} 个")
    print(f"总计: {len(dataset)} 个")
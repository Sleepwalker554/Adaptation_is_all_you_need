import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import FEAT_SEQ_LEN

########################通用特征数据集（预提取）####################################
class FeatureDataset(Dataset):
    """
    从 CSV 加载数据，预加载所有特征到 RAM
    支持 eGeMAPS 和 XLSR 两种特征类型
    """
    def __init__(
            self,
            csv_path: Path,
            project_root: Path,
            xlsr: bool = False,
    ):
        """
        初始化数据集

        参数:
            csv_path: CSV 文件路径
            project_root: 项目根目录
            xlsr: True 使用 XLSR 特征，False 使用 eGeMAPS 特征
        """
        super().__init__()

        self.csv_path = csv_path
        self.project_root = project_root
        self.xlsr = xlsr

        # 根据特征类型设置不同的参数
        if xlsr:
            self.feature_path_key = 'xlsr_path'
            self.feature_name = 'XLSR'
            self.expected_shape = (1, 1024)
        else:
            self.feature_path_key = 'egemaps_path'
            self.feature_name = 'eGeMAPS'
            self.expected_shape = (FEAT_SEQ_LEN, 25)

        # 存储数据
        self.features = []  # 特征 (预加载到 RAM)
        self.labels = []  # AD 标签 (0 或 1)
        self.session_ids = []  # session_id (用于调试)

        # 加载 CSV 并预加载特征
        self._load_data()

    def _load_data(self):
        """
        从 CSV 加载数据并预加载所有特征到 RAM
        """
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                session_id = row['session_id']
                feature_path = row[self.feature_path_key]
                ad = int(row['ad'])

                # 根据特征类型确定路径
                if self.xlsr:
                    csv_dir = Path(self.csv_path).parent
                    feature_path_abs = (csv_dir / feature_path).resolve()
                else:
                    feature_path_abs = self.project_root / feature_path

                # 检查文件是否存在
                if not feature_path_abs.exists():
                    print(f"⚠️  {self.feature_name} 特征文件不存在: {feature_path}")
                    continue

                # 加载特征
                try:
                    features = torch.load(feature_path_abs)
                    
                    # 检查 NaN 和 Inf
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        print(f"⚠️  特征包含 NaN/Inf，跳过: {session_id}")
                        continue
                     
                    # 验证形状
                    if features.shape != self.expected_shape:
                        print(f"⚠️  特征形状错误 {session_id}: {features.shape}, 期望 {self.expected_shape}")
                        continue

                    # 存储
                    self.features.append(features)
                    self.labels.append(ad)
                    self.session_ids.append(session_id)

                except Exception as e:
                    print(f"❌ 加载特征失败 {session_id}: {e}")
                    continue

        # 统计
        num_control = sum(1 for label in self.labels if label == 0)
        num_dementia = sum(1 for label in self.labels if label == 1)

        if len(self.features) == 0:
            print(f"❌ 错误：没有找到任何 {self.feature_name} 特征文件！")
            raise ValueError("数据集为空。")

        print(f"✅ 加载完成: {len(self)} 个样本")
        print(f"   Control: {num_control}, Dementia: {num_dementia}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        获取单个样本

        参数:
            index: 样本索引

        返回:
            features: eGeMAPS 特征 (10, 25) 或 XLSR 特征 (1, 1024)
            label: 0 (Control) 或 1 (Dementia)
        """
        features = self.features[index]
        label = self.labels[index]

        return features, label

    def get_session_id(self, index):
        """
        获取 session_id（用于调试）
        """
        return self.session_ids[index]


def create_dataloaders(
        train_csv: Path,
        val_csv: Path,
        project_root: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        xlsr: bool = False,
):
    """
    创建训练和验证数据加载器

    参数:
        train_csv: 训练集 CSV 路径
        val_csv: 验证集 CSV 路径
        project_root: 项目根目录
        batch_size: 批大小
        num_workers: 工作进程数
        xlsr: True 使用 XLSR 特征，False 使用 eGeMAPS 特征

    返回:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
    """
    from torch.utils.data import DataLoader

    feature_name = "XLSR" if xlsr else "eGeMAPS"

    # 创建数据集
    print(f"=============创建训练集（{feature_name} 特征）=============")
    try:
        train_dataset = FeatureDataset(train_csv, project_root, xlsr=xlsr)
    except ValueError as e:
        print(f"\n❌ 训练集加载失败: {e}")
        raise

    print(f"\n=============创建验证集（{feature_name} 特征）=============")
    try:
        val_dataset = FeatureDataset(val_csv, project_root, xlsr=xlsr)
    except ValueError as e:
        print(f"\n❌ 验证集加载失败: {e}")
        raise
    print()

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空！")
    if len(val_dataset) == 0:
        raise ValueError(f"验证集为空！")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,  # 加速 GPU 传输
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )

    return train_loader, val_loader
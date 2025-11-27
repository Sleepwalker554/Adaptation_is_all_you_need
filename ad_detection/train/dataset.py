import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from config import FEAT_SEQ_LEN, XLSR_FEATURE_DIM, PROJECT_ROOT
class FeatureDataset(Dataset):
    """
    Load data from CSV, preload features to memory
    """
    def __init__(
            self,
            csv_path: Path,
            project_root: Path = None,
            xlsr: bool = False,
    ):
        """
        Args:
            csv_path: CSV file path
            project_root: Project root directory (defaults to config.PROJECT_ROOT if None)
            xlsr: True to use XLSR features, False to use eGeMAPS features
        """
        super().__init__()

        self.csv_path = csv_path
        self.project_root = project_root if project_root is not None else PROJECT_ROOT
        self.xlsr = xlsr

        if xlsr:
            self.feature_path_key = 'xlsr_path'
            self.feature_name = 'XLSR'
            self.expected_shape = (1, XLSR_FEATURE_DIM)
        else:
            self.feature_path_key = 'egemaps_path'
            self.feature_name = 'eGeMAPS'
            self.expected_shape = (FEAT_SEQ_LEN, 25)

        # Store data
        self.features = []  # features
        self.labels = []  # AD labels (0 or 1)
        self.session_ids = []  # session_id

        # Load CSV and preload features
        self._load_data()

    def _load_data(self):
        """
        Load data from CSV and preload all features to memory
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

                # Check if file exists
                if not feature_path_abs.exists():
                    print(f"⚠️  {self.feature_name} feature file does not exist: {feature_path_abs}")
                    continue

                # Load features
                try:
                    features = torch.load(feature_path_abs)
                    
                    # Check for NaN and Inf
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        print(f"⚠️  Features contain NaN/Inf, skipping: {session_id}")
                        continue
                     
                    # Verify shape
                    if features.shape != self.expected_shape:
                        print(f"⚠️  Feature shape error {session_id}: {features.shape}, expected {self.expected_shape}")
                        continue

                    # Store data
                    self.features.append(features)
                    self.labels.append(ad)
                    self.session_ids.append(session_id)

                except Exception as e:
                    print(f"❌ Loading features failed {session_id}: {e}")
                    continue

        num_control = sum(1 for label in self.labels if label == 0)
        num_dementia = sum(1 for label in self.labels if label == 1)

        if len(self.features) == 0:
            print(f"❌ Error: No {self.feature_name} feature files found!")
            raise ValueError("Dataset is empty.")

        print(f"✅ Loading completed: {len(self)} samples")
        print(f"   Control: {num_control}, Dementia: {num_dementia}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        Args:
            index: sample index

        Returns:
            features: eGeMAPS features (FEAT_SEQ_LEN, 25) or XLSR features (1, XLSR_FEATURE_DIM)
            label: 0 (Control) or 1 (Dementia)
        """
        features = self.features[index]
        label = self.labels[index]

        return features, label

    def get_session_id(self, index):
        return self.session_ids[index]


def create_dataloaders(
        train_csv: Path,
        val_csv: Path,
        project_root: Path = None,
        batch_size: int = 32,
        num_workers: int = 4,
        xlsr: bool = False,
):
    """
    Args:
        train_csv: Training set CSV path
        val_csv: Validation set CSV path
        project_root: Project root directory (defaults to config.PROJECT_ROOT if None)
        batch_size: Batch size
        num_workers: Number of worker processes
        xlsr: True to use XLSR features, False to use eGeMAPS features

    Returns:
        train_loader: Training set DataLoader
        val_loader: Validation set DataLoader
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    from torch.utils.data import DataLoader

    feature_name = "XLSR" if xlsr else "eGeMAPS"

    # 创建数据集
    print(f"============= Creating training set ({feature_name} features) =============")
    try:
        train_dataset = FeatureDataset(train_csv, project_root, xlsr=xlsr)
    except ValueError as e:
        print(f"\n❌ Loading training set failed: {e}")
        raise

    print(f"\n============= Creating validation set ({feature_name} features) =============")
    try:
        val_dataset = FeatureDataset(val_csv, project_root, xlsr=xlsr)
    except ValueError as e:
        print(f"\n❌ 验证集加载失败: {e}")
        raise
    print()

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"Training set is empty!")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation set is empty!")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training set
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,  # Speed up GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle validation set
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )

    return train_loader, val_loader
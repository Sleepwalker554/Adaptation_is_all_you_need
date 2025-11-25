"""
Dataset splitting tool
Used to create CSV files for training and validation sets
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
    Create training and validation CSV files
    
    Args:
        raw_audio_dir: Directory containing raw audio files 
                       (with subfolders Control and Dementia)
        train_csv_path: Output path for the training CSV
        val_csv_path: Output path for the validation CSV
        feature_dir_name: Feature directory name 
                          (e.g., "Address_xlsr_features" or "features")
        train_ratio: Ratio of training samples (default: 0.8)
        random_seed: Random seed (default: 42)
        dataset_name: Dataset name for printing information (optional)
        xlsr: Whether to use XLSR feature mode 
              (True: xlsr_path/.xlsr.pt, False: egemaps_path/.egemaps.pt, default: True)
    
    Returns:
        Tuple[Path, Path]: (Training CSV path, Validation CSV path)
    """
    
    # Ensure output directories exist
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    val_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check whether raw audio directory exists
    if not raw_audio_dir.exists():
        raise FileNotFoundError(f"Raw audio directory does not exist: {raw_audio_dir}")
    
    # Collect all audio files
    all_samples = []
    
    # Process Control group
    control_dir = raw_audio_dir / "Control"
    if control_dir.exists():
        # Support .wav and .mp3 formats
        audio_files = list(control_dir.glob("*.wav")) + list(control_dir.glob("*.mp3"))
        for audio_file in sorted(audio_files):
            session_id = audio_file.stem
            all_samples.append({
                'session_id': session_id,
                'ad': 0
            })
    
    # Process Dementia group
    dementia_dir = raw_audio_dir / "Dementia"
    if dementia_dir.exists():
        # Support .wav and .mp3 formats
        audio_files = list(dementia_dir.glob("*.wav")) + list(dementia_dir.glob("*.mp3"))
        for audio_file in sorted(audio_files):
            session_id = audio_file.stem
            all_samples.append({
                'session_id': session_id,
                'ad': 1
            })
    
    if len(all_samples) == 0:
        raise ValueError(f"No audio files found in {raw_audio_dir}")
    
    # Separate by class
    control_samples = [s for s in all_samples if s['ad'] == 0]
    dementia_samples = [s for s in all_samples if s['ad'] == 1]
    
    # Create random generator
    rdm = Random(random_seed)
    
    # Shuffle
    rdm.shuffle(control_samples)
    rdm.shuffle(dementia_samples)
    
    # Split Control
    n_control_train = int(len(control_samples) * train_ratio)
    control_train = control_samples[:n_control_train]
    control_val = control_samples[n_control_train:]
    
    # Split Dementia
    n_dementia_train = int(len(dementia_samples) * train_ratio)
    dementia_train = dementia_samples[:n_dementia_train]
    dementia_val = dementia_samples[n_dementia_train:]
    
    # Combine
    train_samples = control_train + dementia_train
    val_samples = control_val + dementia_val
    
    # Shuffle again
    rdm.shuffle(train_samples)
    rdm.shuffle(val_samples)
    
    # Set column names and file extension based on feature type
    if xlsr:
        feature_col = 'xlsr_path'
        feature_ext = '.xlsr.pt'
    else:
        feature_col = 'egemaps_path'
        feature_ext = '.egemaps.pt'
    
    # Generate training CSV
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in train_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # Generate validation CSV
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in val_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # Print statistics
    print(f"============= {dataset_name} Train/Val Split Complete! =============")
    print(f"Training set: {len(train_samples)} samples (Control: {len(control_train)}, Dementia: {len(dementia_train)})")
    print(f"Validation set: {len(val_samples)} samples (Control: {len(control_val)}, Dementia: {len(dementia_val)})")
    print(f"\nTraining CSV path: {train_csv_path}")
    print(f"Validation CSV path: {val_csv_path}")

    return train_csv_path, val_csv_path

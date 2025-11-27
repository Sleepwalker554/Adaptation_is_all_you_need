import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, Optional
from model import SSLModel, getAttenF
from config import SECOND_LENGTH

def extract_xlsr_features_from_csv(
    train_csv_path: Union[str, Path],
    val_csv_path: Union[str, Path],
    raw_audio_dir: Union[str, Path],
    xlsr_features_dir: Union[str, Path],
    project_root: Union[str, Path],
    sampling_rate: int = 16000,
    device = 'cpu',
):
    """
    Extract XLSR features from audio files listed in train and validation CSV files.
    """
    # Convert all paths to Path objects
    train_csv_path = Path(train_csv_path)
    val_csv_path = Path(val_csv_path)
    raw_audio_dir = Path(raw_audio_dir)
    xlsr_features_dir = Path(xlsr_features_dir)
    project_root = Path(project_root)
    
    # Create features directory if not exists
    xlsr_features_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    
    ssl_model = SSLModel(device, freeze_xlsr=True)

    def _extract_features_from_single_csv(csv_path: Path, split_name: str):
        """
        Internal function to extract features from a single CSV file.
        
        Args:
            csv_path: Path to CSV file
            split_name: Name of the split (e.g., "Train Set", "Val Set")
        """
        print(f"\n============= Extracting XLSR features for {split_name} =============")
        
        extracted = 0
        skipped = 0
        errors = 0
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"{len(rows)} Audio Files")
        
        # Process each row
        for row in tqdm(rows, desc=f"Extracting {split_name}"):
            session_id = row['session_id']
            ad = int(row['ad'])
            
            # Build paths
            folder = "Control" if ad == 0 else "Dementia"
            # Try to find audio file in .wav or .mp3 format
            audio_path_wav = raw_audio_dir / folder / f"{session_id}.wav"
            audio_path_mp3 = raw_audio_dir / folder / f"{session_id}.mp3"
            
            # Determine which audio file exists
            if audio_path_wav.exists():
                audio_path = audio_path_wav
            elif audio_path_mp3.exists():
                audio_path = audio_path_mp3
            else:
                audio_path = None
            
            xlsr_path = xlsr_features_dir / f"{session_id}.xlsr.pt"
            
            # Check if feature already exists
            if xlsr_path.exists():
                skipped += 1
                continue
            
            # Check if audio file exists
            if audio_path is None or not audio_path.exists():
                print(f"\n⚠️  Audio file does not exist (tried .wav and .mp3): {session_id}")
                errors += 1
                continue
            
            try:
                # Load audio
                audio_np, _ = librosa.load(
                    str(audio_path), 
                    sr=sampling_rate, 
                    res_type="kaiser_best"
                )
                audio_np = librosa.to_mono(audio_np)
                audio_np = np.float32(audio_np)

                # Set all audio to SECOND_LENGTH seconds
                max_length = sampling_rate * SECOND_LENGTH  # 45 seconds = 720000 samples at 16kHz
                # Cutting
                if len(audio_np) > max_length:
                    print(f"\nCutting: Audio {session_id}: {len(audio_np)/sampling_rate:.1f}s -> {SECOND_LENGTH:.1f}s")
                    audio_np = audio_np[:max_length]
                # Padding
                if len(audio_np) < max_length:
                    print(f"\nPadding: Audio {session_id}: {len(audio_np)/sampling_rate:.1f}s -> {SECOND_LENGTH:.1f}s")
                    np.pad(audio_np, (0, max_length - len(audio_np)), mode='constant')
 
                # Convert to tensor and move to device
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)
                
                # Extract XLSR features
                emb, layerresult = ssl_model.extract_feat(audio_tensor)
                
                # Use getAttenF for average pooling
                layery, fullfeature = getAttenF(layerresult)
                
                # Use only the last layer's average pooled features
                xlsr_feat = layery[:, -1, :].squeeze(0).cpu()  # Shape: (1024,)
                
                # Stack into list format
                xlsr_features_list = [xlsr_feat]
                xlsr_features = torch.stack(xlsr_features_list, dim=0).detach()  # Shape: (1, 1024)
                
                # Save features
                torch.save(xlsr_features, xlsr_path)
                extracted += 1
                
            except Exception as e:
                print(f"\n❌ Extraction failed for {session_id}: {e}")
                errors += 1
                continue
        
        print(f"\n============= {split_name} Extraction Complete! =============")
        print(f"Successfully extracted: {extracted}")
        print(f"Already exists (skipped): {skipped}")
        print(f"Errors: {errors}")
        print(f"Total: {len(rows)}")
        
        return extracted, skipped, len(rows)
    
    # Extract features for training set
    train_extracted, train_skipped, train_total = _extract_features_from_single_csv(
        train_csv_path, "Train Set"
    )
    
    # Extract features for validation set
    val_extracted, val_skipped, val_total = _extract_features_from_single_csv(
        val_csv_path, "Val Set"
    )
    
    print("\n============= All XLSR Feature Extraction Complete! =============")
    print(f"Training Set:   {train_extracted} extracted, {train_skipped} skipped, {train_total} total")
    print(f"Validation Set: {val_extracted} extracted, {val_skipped} skipped, {val_total} total")   
    return
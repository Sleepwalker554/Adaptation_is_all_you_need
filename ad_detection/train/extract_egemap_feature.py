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
from config import FEAT_SEQ_LEN, SAMPLING_RATE, PROJECT_ROOT

def load_audio(file_path: str, sampling_rate: int) -> np.ndarray:
    """ 
    Args:
        file_path: Audio file path
        sampling_rate: Target sampling rate
    
    Returns:
        audio_array: numpy array, mono audio
    """
    # Use librosa to load (supports MP3 and WAV)
    array, _ = librosa.load(file_path, sr=sampling_rate, res_type="kaiser_best")
    # Ensure mono
    array = librosa.to_mono(array)
    # Convert to float32
    array = np.float32(array)
    return array

class CsvDataset(Dataset):
    """ 
    Load session_id and audio path from CSV
    """
    
    def __init__(self, csv_path: Path, raw_audio_dir: Path, project_root: Path = None):
        super().__init__()
        
        self.csv_path = csv_path
        self.raw_audio_dir = raw_audio_dir
        self.project_root = project_root if project_root is not None else PROJECT_ROOT
        self.data = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row['session_id']
                egemaps_path = row['egemaps_path']
                
                # Build audio path
                # From CSV's third column (ad) to determine audio path in Control or Dementia folder
                ad = int(row['ad'])
                if ad == 0:
                    folder = "Control"
                else:
                    folder = "Dementia"
                
                # Try to find audio file in .wav or .mp3 format
                audio_path_wav = self.raw_audio_dir / folder / f"{session_id}.wav"
                audio_path_mp3 = self.raw_audio_dir / folder / f"{session_id}.mp3"
                
                # Determine which audio file exists
                if audio_path_wav.exists():
                    audio_path = audio_path_wav
                elif audio_path_mp3.exists():
                    audio_path = audio_path_mp3
                else:
                    # Default to .wav (will be caught as not existing later)
                    audio_path = audio_path_wav
                
                self.data.append({
                    'session_id': session_id,
                    'audio_path': str(audio_path.relative_to(self.project_root)),
                    'egemaps_path': egemaps_path,
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['audio_path'], item['egemaps_path'], item['session_id']


def extract_egemaps_features_from_csv(csv_path: Path, raw_audio_dir: Path, project_root: Path = None):
    """
    Args:
        csv_path: CSV file path
        raw_audio_dir: Original audio directory (contains Control and Dementia subfolders)
        project_root: Project root directory (optional, defaults to module-level PROJECT_ROOT)
    """
    # Use provided project_root or fall back to module-level PROJECT_ROOT
    if project_root is None:
        project_root = PROJECT_ROOT
    
    # Create dataset
    dataset = CsvDataset(csv_path, raw_audio_dir=raw_audio_dir, project_root=project_root)
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Process one by one
        shuffle=False,
        num_workers=0,     # OpenSMILE does not support multiple processes
        persistent_workers=False,
    )
    
    print(f"{len(dataset)} Audio Files")
    print()
    
    # Initialize OpenSMILE
    smile_lld = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.LowLevelDescriptors,
    )
    
    # Count the number of extracted features
    extracted = 0
    skipped = 0
    
    # Ignore OpenSMILE warnings
    warnings.simplefilter('ignore')
    
    # Extract features
    for audio_path, egemaps_path, session_id in tqdm(dataloader, desc="Extracting"):
        
        # Convert to absolute path
        audio_path_abs = project_root / audio_path
        egemaps_path_abs = project_root / egemaps_path
        
        # Check if it already exists
        if egemaps_path_abs.exists():
            skipped += 1
            continue
        
        # Check if the audio file exists
        if not audio_path_abs.exists():
            print(f"\n⚠️  Audio file does not exist: {audio_path_abs}")
            continue
        
        try:
            # Load audio
            audio_np = load_audio(str(audio_path_abs), sampling_rate=SAMPLING_RATE)
            
            # Audio segmentation (remove the part that is not enough for one segment)
            usable_length = (audio_np.shape[0] // FEAT_SEQ_LEN) * FEAT_SEQ_LEN
            
            if usable_length == 0:
                print(f"\nAudio too short, skipping extraction: {session_id}")
                continue
            
            # Extract and segment
            audio_segments = np.split(audio_np[:usable_length], FEAT_SEQ_LEN)
            
            # Extract eGeMAPS features segment by segment
            egemaps_list = []
            for segment in audio_segments:
                # OpenSMILE processing
                _, _, features = smile_lld.process(segment, SAMPLING_RATE)
                # features shape: (1, 25) - Take the first frame
                feat_np = np.array(features[0, :], dtype=np.float32)
                egemaps_list.append(torch.from_numpy(feat_np))
            
            # (FEAT_SEQ_LEN, 25) Tensor
            egemaps = torch.stack(egemaps_list, dim=0)
 
            egemaps_path_abs.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(egemaps, egemaps_path_abs)
            
            extracted += 1
            
        except Exception as e:
            print(f"\n❌ Extraction failed {session_id}: {e}")
            continue
    
    print(f"\n============= Extraction completed! =============")
    print(f"Successfully extracted: {extracted} 个")
    print(f"Skipped: {skipped} 个")
    print(f"Total: {len(dataset)} 个")
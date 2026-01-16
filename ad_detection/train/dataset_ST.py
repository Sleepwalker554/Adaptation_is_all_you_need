"""
Self-Training Dataset Components

This module provides dataset classes for self-training with pseudo-labels:
1. PseudoLabelDataset: Dataset that holds pseudo-labeled target data
2. CombinedDataset: Combines source (real labels) + target (pseudo-labels)
3. create_st_dataloaders: Create combined dataloaders for ST training
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from config import BATCH_SIZE, NUM_WORKERS
from dataset import xlsr_pad_mask


class PseudoLabelDataset(Dataset):
    """
    Dataset for target domain samples with pseudo-labels

    Unlike FeatureDataset which loads from CSV, this class:
    - Takes pre-loaded features and pseudo-labels from memory
    - Used for storing high-confidence predictions during ST iterations
    - Maintains session_ids for tracking

    Args:
        features_list: List of torch tensors (seq_len, 1024)
        pseudo_labels_list: List of int (0 or 1)
        session_ids_list: List of str (session identifiers)
        confidences_list: List of float (confidence scores, optional for tracking)
    """

    def __init__(self, features_list, pseudo_labels_list, session_ids_list, confidences_list=None):
        self.features = features_list
        self.labels = pseudo_labels_list
        self.session_ids = session_ids_list
        self.confidences = confidences_list if confidences_list is not None else [1.0] * len(features_list)

        assert len(self.features) == len(self.labels) == len(self.session_ids), \
            f"Mismatched lengths: features={len(self.features)}, labels={len(self.labels)}, ids={len(self.session_ids)}"

        # Print statistics (suppressed for cleaner output)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Returns (features, label) - same format as FeatureDataset"""
        return self.features[index], self.labels[index]

    def get_session_id(self, index):
        """Get session ID for a specific sample"""
        return self.session_ids[index]

    def get_confidence(self, index):
        """Get confidence score for a specific sample"""
        return self.confidences[index]


class CombinedDataset(ConcatDataset):
    """
    Combines source dataset (real labels) and pseudo-labeled target dataset

    This is a thin wrapper around ConcatDataset that provides:
    - Combined iteration over source + target
    - Tracking of dataset sizes for metrics

    Args:
        source_dataset: FeatureDataset with real labels
        pseudo_dataset: PseudoLabelDataset with pseudo-labels
    """

    def __init__(self, source_dataset, pseudo_dataset):
        super().__init__([source_dataset, pseudo_dataset])
        self.source_dataset = source_dataset
        self.pseudo_dataset = pseudo_dataset
        self.source_size = len(source_dataset)
        self.pseudo_size = len(pseudo_dataset)

def create_st_dataloaders(source_dataset, pseudo_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=None):
    """
    Create combined dataloader for self-training

    Args:
        source_dataset: FeatureDataset with real source labels
        pseudo_dataset: PseudoLabelDataset with pseudo-labels (can be empty)
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed for reproducible shuffling (optional)

    Returns:
        combined_loader: DataLoader iterating over source + pseudo-labeled data
    """
    if len(pseudo_dataset) == 0:
        # First iteration: no pseudo-labels yet, use only source
        combined_dataset = source_dataset
    else:
        combined_dataset = CombinedDataset(source_dataset, pseudo_dataset)

    # Create generator for reproducible shuffling
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Worker init function for reproducible worker states
    def worker_init_fn(worker_id):
        if seed is not None:
            worker_seed = seed + worker_id
            import numpy as np
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to mix source and pseudo-labeled samples
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=xlsr_pad_mask,  # Use same collate function for XLSR
        generator=generator,
        worker_init_fn=worker_init_fn if seed is not None else None,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )

    return combined_loader


def create_st_dann_dataloaders(source_dataset, pseudo_dataset, target_full_dataset, 
                                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=None):
    """
    Create dataloaders for ST+DANN parallel training
    
    Creates three loaders:
    1. Combined source + pseudo-labeled loader (for classification loss)
    2. Target full loader (for domain adversarial loss)
    
    Note: For simplicity, we return a combined loader that includes both source and pseudo,
    and a separate target loader for domain adversarial training.
    
    Args:
        source_dataset: FeatureDataset with real source labels
        pseudo_dataset: PseudoLabelDataset with pseudo-labels (can be empty)
        target_full_dataset: FeatureDataset with all target samples (for domain loss)
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed for reproducible shuffling (optional)
    
    Returns:
        combined_loader: DataLoader for source + pseudo-labeled data
        target_loader: DataLoader for all target data (domain adversarial)
    """
    # Create combined source + pseudo loader (same as regular ST)
    combined_loader = create_st_dataloaders(
        source_dataset, pseudo_dataset, batch_size, num_workers, seed
    )
    
    # Create target loader for domain adversarial
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed + 1)  # Different seed for target loader
    
    def worker_init_fn(worker_id):
        if seed is not None:
            worker_seed = seed + 1 + worker_id
            import numpy as np
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
    
    target_loader = DataLoader(
        target_full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=xlsr_pad_mask,
        generator=generator,
        worker_init_fn=worker_init_fn if seed is not None else None,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    
    return combined_loader, target_loader

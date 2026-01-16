"""
Weight Transfer Utilities for DANN+ST Hybrid Training

This module provides utilities to transfer weights from DANN models to ST models,
enabling the ST phase to initialize from domain-invariant features learned by DANN.
"""

import torch
import torch.nn as nn
from model import AD_XLSR_Model
from model_DANN import AD_XLSR_Model_DANN


# Parameter name mapping from DANN to ST
DANN_TO_ST_MAPPING = {
    # Normalization layers
    'norm.weight': 'norm.weight',
    'norm.bias': 'norm.bias',
    'norm.running_mean': 'norm.running_mean',
    'norm.running_var': 'norm.running_var',
    'norm.num_batches_tracked': 'norm.num_batches_tracked',

    # Down projection layers (DANN and ST now use same naming)
    'down_proj1.weight': 'down_proj1.weight',
    'down_proj1.bias': 'down_proj1.bias',
    'down_proj2.weight': 'down_proj2.weight',
    'down_proj2.bias': 'down_proj2.bias',
    'down_proj3.weight': 'down_proj3.weight',
    'down_proj3.bias': 'down_proj3.bias',
    'down_proj4.weight': 'down_proj4.weight',
    'down_proj4.bias': 'down_proj4.bias',
    'down_proj5.weight': 'down_proj5.weight',
    'down_proj5.bias': 'down_proj5.bias',

    # Batch normalization layers
    'bn1.weight': 'bn1.weight',
    'bn1.bias': 'bn1.bias',
    'bn1.running_mean': 'bn1.running_mean',
    'bn1.running_var': 'bn1.running_var',
    'bn1.num_batches_tracked': 'bn1.num_batches_tracked',

    'bn2.weight': 'bn2.weight',
    'bn2.bias': 'bn2.bias',
    'bn2.running_mean': 'bn2.running_mean',
    'bn2.running_var': 'bn2.running_var',
    'bn2.num_batches_tracked': 'bn2.num_batches_tracked',

    'bn3.weight': 'bn3.weight',
    'bn3.bias': 'bn3.bias',
    'bn3.running_mean': 'bn3.running_mean',
    'bn3.running_var': 'bn3.running_var',
    'bn3.num_batches_tracked': 'bn3.num_batches_tracked',

    'bn4.weight': 'bn4.weight',
    'bn4.bias': 'bn4.bias',
    'bn4.running_mean': 'bn4.running_mean',
    'bn4.running_var': 'bn4.running_var',
    'bn4.num_batches_tracked': 'bn4.num_batches_tracked',

    'bn5.weight': 'bn5.weight',
    'bn5.bias': 'bn5.bias',
    'bn5.running_mean': 'bn5.running_mean',
    'bn5.running_var': 'bn5.running_var',
    'bn5.num_batches_tracked': 'bn5.num_batches_tracked',

    # Attention pooling (pool_att in DANN → pool_ad in ST)
    'pool_att.linear1.weight': 'pool_ad.linear1.weight',
    'pool_att.linear1.bias': 'pool_ad.linear1.bias',
    'pool_att.linear2.weight': 'pool_ad.linear2.weight',
    'pool_att.linear2.bias': 'pool_ad.linear2.bias',

    # Classification layer (class_classifier in DANN → output_layer in ST)
    'class_classifier.weight': 'output_layer.weight',
    'class_classifier.bias': 'output_layer.bias',
}


def load_dann_weights_to_st_model(st_model, dann_checkpoint_path, device):
    """
    Transfer feature extractor weights from DANN checkpoint to ST model.

    This function loads a DANN model checkpoint and transfers all relevant weights
    to an ST model, excluding DANN-specific components (GRL, domain classifier).

    Args:
        st_model: AD_XLSR_Model instance (ST model to receive weights)
        dann_checkpoint_path: Path to DANN checkpoint file
        device: Device to load checkpoint on

    Returns:
        st_model: ST model with transferred weights

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
        KeyError: If checkpoint doesn't contain expected keys
        RuntimeError: If weight shapes don't match
    """
    # Load DANN checkpoint
    checkpoint = torch.load(dann_checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint must contain 'model_state_dict' key")

    dann_state_dict = checkpoint['model_state_dict']
    st_state_dict = st_model.state_dict()

    # Transfer weights according to mapping
    transferred_count = 0
    skipped_count = 0

    for dann_key, st_key in DANN_TO_ST_MAPPING.items():
        if dann_key in dann_state_dict:
            if st_key in st_state_dict:
                # Check shape compatibility
                dann_shape = dann_state_dict[dann_key].shape
                st_shape = st_state_dict[st_key].shape

                if dann_shape == st_shape:
                    st_state_dict[st_key] = dann_state_dict[dann_key].clone()
                    transferred_count += 1
                    print(f"✓ {dann_key:35s} → {st_key:35s} {list(dann_shape)}")
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {dann_key} → {st_key}: "
                        f"DANN {dann_shape} vs ST {st_shape}"
                    )
            else:
                print(f"⚠ ST model missing key: {st_key}")
                skipped_count += 1
        else:
            print(f"⚠ DANN checkpoint missing key: {dann_key}")
            skipped_count += 1

    # Load the updated state dict
    st_model.load_state_dict(st_state_dict)

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"\nDAN checkpoint info:")
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_avg_acc' in checkpoint:
        print(f"  Best avg accuracy: {checkpoint['best_avg_acc']*100:.2f}%")
    if 'source_val_acc' in checkpoint:
        print(f"  Source val accuracy: {checkpoint['source_val_acc']*100:.2f}%")
    if 'target_val_acc' in checkpoint:
        print(f"  Target val accuracy: {checkpoint['target_val_acc']*100:.2f}%")

    print(f"{'='*60}\n")

    return st_model


def create_st_model_from_dann(dann_checkpoint_path, device, dropout=0.2):
    """
    Create and initialize ST model from DANN checkpoint.

    This is a convenience function that:
    1. Creates a new AD_XLSR_Model (ST model)
    2. Loads DANN weights into it
    3. Returns the initialized model

    Args:
        dann_checkpoint_path: Path to DANN checkpoint file
        device: Device to create model on
        dropout: Dropout rate for ST model (default: 0.2)

    Returns:
        st_model: AD_XLSR_Model initialized with DANN weights

    Example:
        >>> st_model = create_st_model_from_dann(
        ...     'checkpoints/dann_best.pth',
        ...     device='cuda',
        ...     dropout=0.2
        ... )
    """
    # Create fresh ST model
    st_model = AD_XLSR_Model(dropout=dropout).to(device)

    # Load DANN weights
    st_model = load_dann_weights_to_st_model(st_model, dann_checkpoint_path, device)

    return st_model


def verify_weight_transfer(dann_model, st_model, verbose=True):
    """
    Verify that weights were correctly transferred from DANN to ST model.

    Compares parameter shapes and values between DANN and ST models to ensure
    the weight transfer was successful.

    Args:
        dann_model: AD_XLSR_Model_DANN instance
        st_model: AD_XLSR_Model instance
        verbose: Whether to print detailed verification results (default: True)

    Returns:
        bool: True if all weights match, False otherwise

    Example:
        >>> dann_model = AD_XLSR_Model_DANN(dropout=0.2)
        >>> st_model = create_st_model_from_dann('dann_checkpoint.pth', 'cuda')
        >>> verify_weight_transfer(dann_model, st_model)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying Weight Transfer")
        print(f"{'='*60}\n")

    dann_state_dict = dann_model.state_dict()
    st_state_dict = st_model.state_dict()

    all_match = True
    match_count = 0
    mismatch_count = 0

    for dann_key, st_key in DANN_TO_ST_MAPPING.items():
        if dann_key in dann_state_dict and st_key in st_state_dict:
            dann_param = dann_state_dict[dann_key]
            st_param = st_state_dict[st_key]

            # Check shape
            if dann_param.shape != st_param.shape:
                if verbose:
                    print(f"✗ Shape mismatch: {dann_key} → {st_key}")
                    print(f"  DANN: {dann_param.shape}, ST: {st_param.shape}")
                all_match = False
                mismatch_count += 1
                continue

            # Check values (allowing for small numerical differences)
            if torch.allclose(dann_param, st_param, rtol=1e-5, atol=1e-7):
                if verbose:
                    print(f"✓ {dann_key:35s} → {st_key:35s} [MATCH]")
                match_count += 1
            else:
                if verbose:
                    print(f"✗ Value mismatch: {dann_key} → {st_key}")
                    max_diff = (dann_param - st_param).abs().max().item()
                    print(f"  Max difference: {max_diff:.2e}")
                all_match = False
                mismatch_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Verification Summary:")
        print(f"  ✓ Matched: {match_count} parameters")
        print(f"  ✗ Mismatched: {mismatch_count} parameters")
        print(f"  Result: {'SUCCESS' if all_match else 'FAILED'}")
        print(f"{'='*60}\n")

    return all_match


def get_transferable_parameter_count(dann_model):
    """
    Count the number of transferable parameters from DANN to ST model.

    Args:
        dann_model: AD_XLSR_Model_DANN instance

    Returns:
        dict: Dictionary with parameter counts
            - 'transferable': Number of transferable parameters
            - 'total': Total number of parameters in DANN
            - 'dann_only': Number of DANN-specific parameters (GRL, domain classifier)
    """
    dann_state_dict = dann_model.state_dict()

    transferable_params = 0
    total_params = 0

    for key, param in dann_state_dict.items():
        param_count = param.numel()
        total_params += param_count

        if key in DANN_TO_ST_MAPPING:
            transferable_params += param_count

    dann_only_params = total_params - transferable_params

    return {
        'transferable': transferable_params,
        'total': total_params,
        'dann_only': dann_only_params,
        'transfer_ratio': transferable_params / total_params if total_params > 0 else 0
    }
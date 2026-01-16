"""
Weight Transfer Utilities for DANN+ST Hybrid Training (eGeMAPS Version)

This module provides utilities to transfer weights from DANN models to ST models
for eGeMAPS features, enabling the ST phase to initialize from domain-invariant
features learned by DANN.
"""

import torch
import torch.nn as nn
from model import AD_EGE_Model
from model_DANN import AD_EGE_Model_DANN


# Parameter name mapping from DANN to ST for eGeMAPS
DANN_TO_ST_MAPPING_EGEMAPS = {
    # Layer 1
    'linear_layer1.weight': 'linear_layer1.weight',
    'linear_layer1.bias': 'linear_layer1.bias',
    'norm1.weight': 'norm1.weight',
    'norm1.bias': 'norm1.bias',
    'norm1.running_mean': 'norm1.running_mean',
    'norm1.running_var': 'norm1.running_var',
    'norm1.num_batches_tracked': 'norm1.num_batches_tracked',
    
    # Layer 2
    'linear_layer2.weight': 'linear_layer2.weight',
    'linear_layer2.bias': 'linear_layer2.bias',
    'norm2.weight': 'norm2.weight',
    'norm2.bias': 'norm2.bias',
    'norm2.running_mean': 'norm2.running_mean',
    'norm2.running_var': 'norm2.running_var',
    'norm2.num_batches_tracked': 'norm2.num_batches_tracked',
    
    # Attention pooling (pool_att in DANN → pool_ad in ST)
    'pool_att.linear1.weight': 'pool_ad.linear1.weight',
    'pool_att.linear1.bias': 'pool_ad.linear1.bias',
    'pool_att.linear2.weight': 'pool_ad.linear2.weight',
    'pool_att.linear2.bias': 'pool_ad.linear2.bias',
    
    # Classification layer (class_classifier in DANN → output_layer in ST)
    'class_classifier.weight': 'output_layer.weight',
    'class_classifier.bias': 'output_layer.bias',
}


def load_dann_weights_to_st_model_egemaps(st_model, dann_checkpoint_path, device):
    """
    Transfer feature extractor weights from DANN checkpoint to ST model (eGeMAPS version).

    This function loads a DANN model checkpoint and transfers all relevant weights
    to an ST model, excluding DANN-specific components (GRL, domain classifier).

    Args:
        st_model: AD_EGE_Model instance (ST model to receive weights)
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

    for dann_key, st_key in DANN_TO_ST_MAPPING_EGEMAPS.items():
        if dann_key in dann_state_dict:
            if st_key in st_state_dict:
                # Check shape compatibility
                dann_shape = dann_state_dict[dann_key].shape
                st_shape = st_state_dict[st_key].shape

                if dann_shape == st_shape:
                    st_state_dict[st_key] = dann_state_dict[dann_key].clone()
                    transferred_count += 1
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {dann_key} → {st_key}: "
                        f"DANN {dann_shape} vs ST {st_shape}"
                    )
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    # Load the updated state dict
    st_model.load_state_dict(st_state_dict)

    return st_model


def create_st_model_from_dann_egemaps(dann_checkpoint_path, device, dim_input=25, dim_hidden=32, dropout=0.3):
    """
    Create and initialize ST model from DANN checkpoint (eGeMAPS version).

    This is a convenience function that:
    1. Creates a new AD_EGE_Model (ST model)
    2. Loads DANN weights into it
    3. Returns the initialized model

    Args:
        dann_checkpoint_path: Path to DANN checkpoint file
        device: Device to create model on
        dim_input: Input dimension (default: 25 for eGeMAPS)
        dim_hidden: Hidden dimension (default: 32 for eGeMAPS)
        dropout: Dropout rate for ST model (default: 0.3 for eGeMAPS)

    Returns:
        st_model: AD_EGE_Model initialized with DANN weights

    Example:
        >>> st_model = create_st_model_from_dann_egemaps(
        ...     'checkpoints/dann_best.pth',
        ...     device='cuda',
        ...     dropout=0.3
        ... )
    """
    # Create fresh ST model
    st_model = AD_EGE_Model(dim_input=dim_input, dim_hidden=dim_hidden, dropout=dropout).to(device)

    # Load DANN weights
    st_model = load_dann_weights_to_st_model_egemaps(st_model, dann_checkpoint_path, device)

    return st_model


def verify_weight_transfer_egemaps(dann_model, st_model, verbose=True):
    """
    Verify that weights were correctly transferred from DANN to ST model (eGeMAPS version).

    Compares parameter shapes and values between DANN and ST models to ensure
    the weight transfer was successful.

    Args:
        dann_model: AD_EGE_Model_DANN instance
        st_model: AD_EGE_Model instance
        verbose: Whether to print detailed verification results (default: True)

    Returns:
        bool: True if all weights match, False otherwise

    Example:
        >>> dann_model = AD_EGE_Model_DANN(dropout=0.3)
        >>> st_model = create_st_model_from_dann_egemaps('dann_checkpoint.pth', 'cuda')
        >>> verify_weight_transfer_egemaps(dann_model, st_model)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying Weight Transfer (eGeMAPS)")
        print(f"{'='*60}\n")

    dann_state_dict = dann_model.state_dict()
    st_state_dict = st_model.state_dict()

    all_match = True
    match_count = 0
    mismatch_count = 0

    for dann_key, st_key in DANN_TO_ST_MAPPING_EGEMAPS.items():
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


def get_transferable_parameter_count_egemaps(dann_model):
    """
    Count the number of transferable parameters from DANN to ST model (eGeMAPS version).

    Args:
        dann_model: AD_EGE_Model_DANN instance

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

        if key in DANN_TO_ST_MAPPING_EGEMAPS:
            transferable_params += param_count

    dann_only_params = total_params - transferable_params

    return {
        'transferable': transferable_params,
        'total': total_params,
        'dann_only': dann_only_params,
        'transfer_ratio': transferable_params / total_params if total_params > 0 else 0
    }


if __name__ == "__main__":
    # Example usage and testing
    print("\nDANN to ST Weight Transfer Utilities (eGeMAPS)")
    print("=" * 60)

    # Create sample models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create models
    dann_model = AD_EGE_Model_DANN(dim_input=25, dim_hidden=32, dropout=0.3).to(device)
    st_model = AD_EGE_Model(dim_input=25, dim_hidden=32, dropout=0.3).to(device)

    print("Models created:")
    print(f"  DANN model: AD_EGE_Model_DANN")
    print(f"  ST model: AD_EGE_Model")

    # Check parameter counts
    param_info = get_transferable_parameter_count_egemaps(dann_model)
    print(f"\nParameter Statistics:")
    print(f"  Total DANN parameters: {param_info['total']:,}")
    print(f"  Transferable parameters: {param_info['transferable']:,}")
    print(f"  DANN-only parameters: {param_info['dann_only']:,}")
    print(f"  Transfer ratio: {param_info['transfer_ratio']*100:.1f}%")

    # Verify architecture compatibility
    print(f"\n{'='*60}")
    print(f"Architecture Compatibility Check")
    print(f"{'='*60}\n")

    dann_state_dict = dann_model.state_dict()
    st_state_dict = st_model.state_dict()

    compatible = True
    for dann_key, st_key in DANN_TO_ST_MAPPING_EGEMAPS.items():
        if dann_key in dann_state_dict and st_key in st_state_dict:
            dann_shape = dann_state_dict[dann_key].shape
            st_shape = st_state_dict[st_key].shape

            if dann_shape == st_shape:
                print(f"✓ {dann_key:35s} → {st_key:35s} {list(dann_shape)}")
            else:
                print(f"✗ {dann_key:35s} → {st_key:35s} MISMATCH!")
                print(f"  DANN: {dann_shape}, ST: {st_shape}")
                compatible = False

    print(f"\n{'='*60}")
    if compatible:
        print(f"✓ Architecture compatibility: PASS")
        print(f"  All parameter shapes match between DANN and ST models")
    else:
        print(f"✗ Architecture compatibility: FAIL")
        print(f"  Some parameter shapes do not match")
    print(f"{'='*60}\n")

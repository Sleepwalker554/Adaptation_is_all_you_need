"""
DANN+ST Hybrid Training Pipeline for eGeMAPS Features

This module implements a two-phase domain adaptation approach for eGeMAPS:
- Phase 1: Train DANN model for domain-invariant features
- Phase 2: Run ST iterations, each reinitializing from DANN checkpoint

Key Innovation:
    Each ST iteration reinitializes from DANN checkpoint (not from scratch or incremental),
    preventing catastrophic forgetting while leveraging pseudo-labels.

Key differences from XLSR version:
- Uses AD_EGE_Model_DANN and AD_EGE_Model
- No mask/padding needed (eGeMAPS features are fixed-size vectors)
- Different hyperparameters (EGEMAPS_DROPOUT=0.3, lambda_domain=1.0, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import LEARNING_RATE, WEIGHT_DECAY, EGEMAPS_DROPOUT
from config import DANN_LAMBDA_CLASS
from model import AD_EGE_Model
from model_DANN import AD_EGE_Model_DANN, compute_alpha
from model_DANN_ST_egemaps import create_st_model_from_dann_egemaps, verify_weight_transfer_egemaps
from train_ST_egemaps import generate_pseudo_labels_egemaps, PseudoLabelDataset_EGE, create_st_dataloaders_egemaps


# eGeMAPS-specific defaults (different from XLSR)
EGEMAPS_LAMBDA_DOMAIN = 1.0  # Higher than XLSR's 0.1
EGEMAPS_DOMAIN_WARMUP_EPOCHS = 5  # Lower than XLSR's 30
EGEMAPS_DOMAIN_ANNEAL_RATIO = 0  # No annealing (vs XLSR's 0.25)
EGEMAPS_BATCH_SIZE = 12  # Larger than XLSR's 4
EGEMAPS_DIM_INPUT = 25
EGEMAPS_DIM_HIDDEN = 32

# Default class weights for handling imbalance (aligned with eGeMAPS baseline)
DEFAULT_CLASS_WEIGHT_CONTROL = 1.1
DEFAULT_CLASS_WEIGHT_DEMENTIA = 1.0


############################################################
# Phase 1: DANN Training for eGeMAPS
############################################################

def compute_domain_lambda(epoch, total_epochs, base_lambda=EGEMAPS_LAMBDA_DOMAIN,
                         warmup_epochs=EGEMAPS_DOMAIN_WARMUP_EPOCHS, anneal_ratio=EGEMAPS_DOMAIN_ANNEAL_RATIO):
    """
    Schedule domain loss weight based on epoch.

    Schedule:
    - Warmup phase: First warmup_epochs epochs, lambda=0 (only train source classifier)
    - Anneal phase: Linearly increase to base_lambda over anneal_ratio proportion of epochs
    - Stable phase: Maintain base_lambda

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lambda: Maximum domain loss weight (default: 1.0 for eGeMAPS)
        warmup_epochs: Number of warmup epochs (default: 5 for eGeMAPS)
        anneal_ratio: Proportion of epochs for annealing (default: 0 for eGeMAPS)

    Returns:
        lambda_domain: Domain loss weight for current epoch
    """
    warmup_epochs = min(warmup_epochs, total_epochs)
    anneal_epochs = int(total_epochs * anneal_ratio)

    if epoch <= warmup_epochs:
        return 0.0

    if anneal_epochs <= 0:
        return base_lambda

    # Linear anneal from warmup_epochs to warmup_epochs + anneal_epochs
    progress = min(1.0, max(0, epoch - warmup_epochs) / max(1, anneal_epochs))
    return base_lambda * progress


def train_one_epoch_dann_egemaps(model, source_loader, target_loader, optimizer, device, epoch, total_epochs,
                                  lambda_class=DANN_LAMBDA_CLASS, lambda_domain=EGEMAPS_LAMBDA_DOMAIN,
                                  class_weight_control=DEFAULT_CLASS_WEIGHT_CONTROL, 
                                  class_weight_dementia=DEFAULT_CLASS_WEIGHT_DEMENTIA):
    """
    Train one epoch with DANN for eGeMAPS.

    Args:
        model: AD_EGE_Model_DANN instance
        source_loader: Source domain data loader
        target_loader: Target domain data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch (1-indexed)
        total_epochs: Total epochs
        lambda_class: Classification loss weight (default: 1.0)
        lambda_domain: Domain adversarial loss weight for current epoch

    Returns:
        avg_class_loss: Average classification loss
        avg_domain_loss: Average domain loss
        train_acc: Training accuracy on source domain
        domain_acc: Domain classification accuracy
    """
    model.train()
    
    # Class weights for handling imbalance
    class_weights = torch.tensor([class_weight_control, class_weight_dementia]).to(device)

    total_class_loss = 0
    total_domain_loss = 0
    correct = 0
    total = 0
    domain_correct = 0
    domain_total = 0

    target_iter = iter(target_loader)
    len_source = len(source_loader)
    total_steps = total_epochs * len_source
    lambda_active = lambda_domain > 0

    pbar = tqdm(enumerate(source_loader), total=len_source, desc=f"Epoch {epoch}", leave=False)

    for i, source_batch in pbar:
        # Compute alpha (gradient reversal strength)
        current_step = (epoch - 1) * len_source + i
        alpha = compute_alpha(current_step, total_steps)

        # ===== Process source domain data =====
        # Note: eGeMAPS dataloader returns (features, labels) without masks
        source_features, source_labels = source_batch
        source_features = source_features.to(device)
        source_labels = source_labels.to(device)

        if lambda_active:
            class_output, domain_output = model(
                source_features,
                mask=None,  # eGeMAPS doesn't need mask
                alpha=alpha,
                return_features=True
            )
        else:
            class_output = model(
                source_features,
                mask=None,
                alpha=alpha,
                return_features=False
            )
            domain_output = None

        # Source domain label = 0
        source_domain_labels = torch.zeros(source_labels.size(0), dtype=torch.long).to(device)

        # Compute classification loss with class weights
        loss_class = F.cross_entropy(class_output, source_labels, weight=class_weights)

        # Compute domain loss if active
        if lambda_active:
            loss_domain_source = F.cross_entropy(domain_output, source_domain_labels)

            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_features, _ = target_batch  # Ignore target labels, no masks
            target_features = target_features.to(device)

            _, domain_output_target = model(
                target_features,
                mask=None,
                alpha=alpha,
                return_features=True
            )

            # Target domain label = 1
            target_domain_labels = torch.ones(target_features.size(0), dtype=torch.long).to(device)
            loss_domain_target = F.cross_entropy(domain_output_target, target_domain_labels)
        else:
            loss_domain_source = torch.zeros(1, device=device)
            loss_domain_target = torch.zeros(1, device=device)
            target_domain_labels = None
            domain_output_target = None

        # Total loss
        total_batch_loss = (lambda_class * loss_class +
                           lambda_domain * (loss_domain_source + loss_domain_target))

        # Backward and optimize
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # Statistics
        total_class_loss += loss_class.item()
        total_domain_loss += (loss_domain_source.item() + loss_domain_target.item())

        predictions = torch.argmax(class_output, dim=1)
        correct += (predictions == source_labels).sum().item()
        total += source_labels.size(0)

        if lambda_active:
            domain_pred_source = torch.argmax(domain_output, dim=1)
            domain_pred_target = torch.argmax(domain_output_target, dim=1)
            domain_correct += (domain_pred_source == source_domain_labels).sum().item()
            domain_correct += (domain_pred_target == target_domain_labels).sum().item()
            domain_total += source_domain_labels.size(0) + target_domain_labels.size(0)

        pbar.set_postfix({
            'α': f'{alpha:.2f}',
            'λ_dom': f'{lambda_domain:.3f}',
            'cls': f'{loss_class.item():.3f}',
            'dom': f'{(loss_domain_source.item() + loss_domain_target.item()):.3f}',
            'acc': f'{correct/total:.3f}'
        })

    avg_class_loss = total_class_loss / len_source
    avg_domain_loss = total_domain_loss / len_source
    train_acc = correct / total
    domain_acc = (domain_correct / domain_total) if domain_total > 0 else 0.0

    return avg_class_loss, avg_domain_loss, train_acc, domain_acc


def validate_dann_egemaps(model, val_loader, device):
    """
    Validate DANN model for eGeMAPS.

    Args:
        model: AD_EGE_Model_DANN instance
        val_loader: Validation data loader
        device: Device

    Returns:
        avg_loss: Average loss
        accuracy: Overall accuracy
        control_acc: Control class accuracy
        dementia_acc: Dementia class accuracy
        f1_score: F1 score for dementia class
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    control_correct, control_total = 0, 0
    dementia_correct, dementia_total = 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for features, labels in val_loader:  # Note: no masks for eGeMAPS
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features, mask=None, return_features=False)

            loss = F.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(predictions, labels):
                if label == 0:
                    control_total += 1
                    if pred == label:
                        control_correct += 1
                else:
                    dementia_total += 1
                    if pred == label:
                        dementia_correct += 1

                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    control_acc = control_correct / control_total if control_total > 0 else 0
    dementia_acc = dementia_correct / dementia_total if dementia_total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, control_acc, dementia_acc, f1_score


def train_phase1_dann_egemaps(seed, source_train_loader, target_train_loader,
                               source_val_loader, target_val_loader, output_dir, device,
                               max_epochs=60, domain_warmup_epochs=EGEMAPS_DOMAIN_WARMUP_EPOCHS, 
                               domain_anneal_ratio=EGEMAPS_DOMAIN_ANNEAL_RATIO,
                               lambda_domain=EGEMAPS_LAMBDA_DOMAIN, lambda_class=DANN_LAMBDA_CLASS,
                               learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                               dropout=EGEMAPS_DROPOUT, dim_input=EGEMAPS_DIM_INPUT, 
                               dim_hidden=EGEMAPS_DIM_HIDDEN, source_weight=0.5, target_weight=0.5,
                               min_source_acc=0.0, min_target_acc=0.0,
                               class_weight_control=DEFAULT_CLASS_WEIGHT_CONTROL,
                               class_weight_dementia=DEFAULT_CLASS_WEIGHT_DEMENTIA):
    """
    Phase 1: Train DANN model for domain-invariant features (eGeMAPS version).

    Args:
        seed: Random seed
        source_train_loader: Source training data loader
        target_train_loader: Target training data loader
        source_val_loader: Source validation data loader
        target_val_loader: Target validation data loader
        output_dir: Output directory for checkpoints
        device: Device
        max_epochs: Maximum training epochs (default: 60)
        domain_warmup_epochs: Warmup epochs with no domain loss (default: 5 for eGeMAPS)
        domain_anneal_ratio: Proportion of epochs for annealing (default: 0 for eGeMAPS)
        lambda_domain: Domain loss weight (default: 1.0 for eGeMAPS)
        lambda_class: Classification loss weight (default: 1.0)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.3 for eGeMAPS)
        dim_input: Input dimension (default: 25 for eGeMAPS)
        dim_hidden: Hidden dimension (default: 32 for eGeMAPS)
        source_weight: Weight for source validation accuracy (default: 0.5)
        target_weight: Weight for target validation accuracy (default: 0.5)
        min_source_acc: Minimum source accuracy threshold (default: 0.0)
        min_target_acc: Minimum target accuracy threshold (default: 0.0)

    Returns:
        best_checkpoint_path: Path to best DANN checkpoint
        phase1_metrics: Dict with training history and best metrics
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 1: DANN Training (Seed {seed})")
    print(f"{'='*70}")

    # Create model and optimizer
    model = AD_EGE_Model_DANN(dim_input=dim_input, dim_hidden=dim_hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training history
    history = {
        'epochs': [],
        'train_class_losses': [],
        'train_domain_losses': [],
        'train_accs': [],
        'source_val_accs': [],
        'target_val_accs': [],
        'avg_val_accs': [],
        'val_losses': [],
        'domain_accs': [],
        'lambda_domains': []
    }

    # Best model tracking (primary: with thresholds)
    best_avg_acc = 0
    best_metrics = {}
    best_checkpoint_path = None

    # Fallback model tracking (best avg regardless of thresholds)
    fallback_best_avg_acc = 0
    fallback_best_metrics = {}
    fallback_checkpoint_path = None

    # Training loop
    for epoch in range(1, max_epochs + 1):
        current_lambda_domain = compute_domain_lambda(
            epoch, max_epochs,
            base_lambda=lambda_domain,
            warmup_epochs=domain_warmup_epochs,
            anneal_ratio=domain_anneal_ratio
        )

        # Train one epoch
        class_loss, domain_loss, train_acc, domain_acc = train_one_epoch_dann_egemaps(
            model, source_train_loader, target_train_loader, optimizer, device,
            epoch, max_epochs,
            lambda_class=lambda_class,
            lambda_domain=current_lambda_domain,
            class_weight_control=class_weight_control,
            class_weight_dementia=class_weight_dementia
        )

        # Validate on source and target
        source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = validate_dann_egemaps(
            model, source_val_loader, device
        )

        target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = validate_dann_egemaps(
            model, target_val_loader, device
        )

        # Compute weighted average accuracy
        avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc

        # Update history
        history['epochs'].append(epoch)
        history['train_class_losses'].append(class_loss)
        history['train_domain_losses'].append(domain_loss)
        history['train_accs'].append(train_acc)
        history['source_val_accs'].append(source_val_acc)
        history['target_val_accs'].append(target_val_acc)
        history['avg_val_accs'].append(avg_val_acc)
        history['val_losses'].append(target_val_loss)
        history['domain_accs'].append(domain_acc)
        history['lambda_domains'].append(current_lambda_domain)

        print(f"Epoch {epoch:3d} | Train: {train_acc:.3f} | Source Val: {source_val_acc:.3f} | "
              f"Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f} | Domain: {domain_acc:.3f} | "
              f"λ_dom: {current_lambda_domain:.3f}")

        # Primary: Save if better avg AND meets thresholds
        if (avg_val_acc > best_avg_acc and
            source_val_acc >= min_source_acc and
            target_val_acc >= min_target_acc):
            best_avg_acc = avg_val_acc
            best_metrics = {
                'epoch': epoch,
                'avg_val_acc': avg_val_acc,
                'source_val_acc': source_val_acc,
                'target_val_acc': target_val_acc,
                'source_val_loss': source_val_loss,
                'target_val_loss': target_val_loss,
                'source_control_acc': source_control_acc,
                'source_dementia_acc': source_dementia_acc,
                'target_control_acc': target_control_acc,
                'target_dementia_acc': target_dementia_acc,
                'source_f1': source_f1,
                'target_f1': target_f1,
            }
            best_checkpoint_path = output_dir / 'best_dann.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_avg_acc': best_avg_acc,
                'source_val_acc': source_val_acc,
                'target_val_acc': target_val_acc,
            }, best_checkpoint_path)
            print(f"  → Primary best model saved (Avg: {avg_val_acc:.4f})")

        # Fallback: Save best avg regardless of thresholds
        if avg_val_acc > fallback_best_avg_acc:
            fallback_best_avg_acc = avg_val_acc
            fallback_best_metrics = {
                'epoch': epoch,
                'avg_val_acc': avg_val_acc,
                'source_val_acc': source_val_acc,
                'target_val_acc': target_val_acc,
                'source_val_loss': source_val_loss,
                'target_val_loss': target_val_loss,
                'source_control_acc': source_control_acc,
                'source_dementia_acc': source_dementia_acc,
                'target_control_acc': target_control_acc,
                'target_dementia_acc': target_dementia_acc,
                'source_f1': source_f1,
                'target_f1': target_f1,
            }
            fallback_checkpoint_path = output_dir / 'fallback_best_dann.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_avg_acc': fallback_best_avg_acc,
                'source_val_acc': source_val_acc,
                'target_val_acc': target_val_acc,
            }, fallback_checkpoint_path)

    # Determine which checkpoint to use
    if best_metrics:
        print(f"\n✓ Best model at epoch {best_metrics['epoch']}: "
              f"Avg {best_metrics['avg_val_acc']*100:.2f}% | "
              f"Source {best_metrics['source_val_acc']*100:.2f}% | "
              f"Target {best_metrics['target_val_acc']*100:.2f}%")
        final_checkpoint = best_checkpoint_path
        final_metrics = best_metrics
    else:
        print(f"\n✓ Best model at epoch {fallback_best_metrics['epoch']}: "
              f"Avg {fallback_best_metrics['avg_val_acc']*100:.2f}% | "
              f"Source {fallback_best_metrics['source_val_acc']*100:.2f}% | "
              f"Target {fallback_best_metrics['target_val_acc']*100:.2f}%")
        final_checkpoint = fallback_checkpoint_path
        final_metrics = fallback_best_metrics

    print(f"Phase 1 completed.\n")

    phase1_metrics = {
        'history': history,
        'best_metrics': final_metrics,
        'best_checkpoint': str(final_checkpoint)
    }

    return str(final_checkpoint), phase1_metrics


############################################################
# Phase 2: Self-Training with DANN Initialization
############################################################

def train_one_epoch_st_egemaps(model, train_loader, optimizer, device, epoch,
                                class_weight_control=DEFAULT_CLASS_WEIGHT_CONTROL,
                                class_weight_dementia=DEFAULT_CLASS_WEIGHT_DEMENTIA):
    """
    Train one epoch for ST (simple classification, no domain adversarial) for eGeMAPS.

    Args:
        model: AD_EGE_Model instance
        train_loader: Training data loader (source + pseudo-labeled target)
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        class_weight_control: Weight for Control class
        class_weight_dementia: Weight for Dementia class

    Returns:
        avg_loss: Average loss
        train_acc: Training accuracy
    """
    model.train()
    
    # Class weights for handling imbalance
    class_weights = torch.tensor([class_weight_control, class_weight_dementia]).to(device)

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for features, labels in pbar:  # Note: no masks for eGeMAPS
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(features)

        # Compute loss with class weights
        loss = F.cross_entropy(logits, labels, weight=class_weights)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{correct/total:.3f}'
        })

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total

    return avg_loss, train_acc


def validate_st_egemaps(model, val_loader, device):
    """
    Validate ST model for eGeMAPS.

    Args:
        model: AD_EGE_Model instance
        val_loader: Validation data loader
        device: Device

    Returns:
        avg_loss: Average loss
        accuracy: Overall accuracy
        control_acc: Control class accuracy
        dementia_acc: Dementia class accuracy
        f1_score: F1 score for dementia class
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    control_correct, control_total = 0, 0
    dementia_correct, dementia_total = 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for features, labels in val_loader:  # Note: no masks for eGeMAPS
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            loss = F.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(predictions, labels):
                if label == 0:
                    control_total += 1
                    if pred == label:
                        control_correct += 1
                else:
                    dementia_total += 1
                    if pred == label:
                        dementia_correct += 1

                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    control_acc = control_correct / control_total if control_total > 0 else 0
    dementia_acc = dementia_correct / dementia_total if dementia_total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, control_acc, dementia_acc, f1_score


def train_phase2_st_with_dann_init_egemaps(seed, dann_checkpoint_path,
                                            source_train_dataset, target_train_dataset,
                                            source_val_loader, target_val_loader,
                                            output_dir, device,
                                            max_iterations=5, max_epochs_per_iter=60,
                                            confidence_threshold=0.9, learning_rate=LEARNING_RATE,
                                            weight_decay=WEIGHT_DECAY, dropout=EGEMAPS_DROPOUT,
                                            dim_input=EGEMAPS_DIM_INPUT, dim_hidden=EGEMAPS_DIM_HIDDEN,
                                            batch_size=EGEMAPS_BATCH_SIZE,
                                            source_weight=0.5, target_weight=0.5,
                                            min_source_acc=0.0, min_target_acc=0.0,
                                            patience=10, min_pseudo_samples=5,
                                            class_weight_control=DEFAULT_CLASS_WEIGHT_CONTROL,
                                            class_weight_dementia=DEFAULT_CLASS_WEIGHT_DEMENTIA):
    """
    Phase 2: Self-Training with DANN Initialization for eGeMAPS.

    Each ST iteration:
    1. Generate pseudo-labels from target domain using current model
    2. Reinitialize ST model from DANN checkpoint (not from previous iteration!)
    3. Train on source + pseudo-labeled target
    4. Validate and track best model

    Args:
        seed: Random seed
        dann_checkpoint_path: Path to DANN checkpoint from Phase 1
        source_train_dataset: FeatureDataset for source training
        target_train_dataset: FeatureDataset for target training (unlabeled)
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Output directory
        device: Device
        max_iterations: Maximum ST iterations (default: 5)
        max_epochs_per_iter: Max epochs per iteration (default: 60)
        confidence_threshold: Pseudo-label confidence threshold (default: 0.9)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.3)
        dim_input: Input dimension (default: 25)
        dim_hidden: Hidden dimension (default: 32)
        batch_size: Batch size (default: 12)
        source_weight: Weight for source validation (default: 0.5)
        target_weight: Weight for target validation (default: 0.5)
        min_source_acc: Minimum source accuracy (default: 0.0)
        min_target_acc: Minimum target accuracy (default: 0.0)
        patience: Early stopping patience (default: 10)
        min_pseudo_samples: Minimum pseudo-samples required (default: 5)

    Returns:
        all_metrics: List of metrics dicts for each iteration
        all_histories: List of training histories for each iteration
        all_pseudo_stats: List of pseudo-label statistics for each iteration
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 2: Self-Training (Seed {seed})")
    print(f"{'='*70}")

    # Initialize model for pseudo-label generation (from DANN checkpoint)
    pseudo_gen_model = create_st_model_from_dann_egemaps(
        dann_checkpoint_path, device,
        dim_input=dim_input, dim_hidden=dim_hidden, dropout=dropout
    )

    all_metrics = []
    all_histories = []
    all_pseudo_stats = []

    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"ST Iteration {iteration}")
        print(f"{'='*50}")

        # ===== Step 1: Generate pseudo-labels =====
        pseudo_features, pseudo_labels, pseudo_session_ids, pseudo_confidences, pseudo_stats = generate_pseudo_labels_egemaps(
            pseudo_gen_model, target_train_dataset, device, confidence_threshold=confidence_threshold
        )

        all_pseudo_stats.append(pseudo_stats)

        # Check if we have enough pseudo-labels
        if len(pseudo_features) < min_pseudo_samples:
            print(f"\n⚠️  Only {len(pseudo_features)} pseudo-samples (< {min_pseudo_samples}), stopping ST")
            break

        # Create pseudo-label dataset
        pseudo_dataset = PseudoLabelDataset_EGE(
            pseudo_features, pseudo_labels, pseudo_session_ids, pseudo_confidences
        )

        # ===== Step 2: Reinitialize model from DANN checkpoint =====
        st_model = create_st_model_from_dann_egemaps(
            dann_checkpoint_path, device,
            dim_input=dim_input, dim_hidden=dim_hidden, dropout=dropout
        )

        # ===== Step 3: Train on source + pseudo-labeled target =====
        optimizer = torch.optim.AdamW(st_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Create combined dataloader
        train_loader = create_st_dataloaders_egemaps(source_train_dataset, pseudo_dataset, batch_size=batch_size, seed=seed)

        # Training history for this iteration
        history = {
            'epochs': [],
            'train_losses': [],
            'train_accs': [],
            'source_val_accs': [],
            'target_val_accs': [],
            'avg_val_accs': [],
            'val_losses': []
        }

        best_avg_acc = 0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(1, max_epochs_per_iter + 1):
            # Train
            train_loss, train_acc = train_one_epoch_st_egemaps(
                st_model, train_loader, optimizer, device, epoch,
                class_weight_control=class_weight_control,
                class_weight_dementia=class_weight_dementia
            )

            # Validate on both domains
            source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = validate_st_egemaps(
                st_model, source_val_loader, device
            )

            target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = validate_st_egemaps(
                st_model, target_val_loader, device
            )

            # Compute weighted average
            avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc

            # Update history
            history['epochs'].append(epoch)
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            history['source_val_accs'].append(source_val_acc)
            history['target_val_accs'].append(target_val_acc)
            history['avg_val_accs'].append(avg_val_acc)
            history['val_losses'].append(target_val_loss)

            print(f"Epoch {epoch:3d} | Train: {train_acc:.3f} | Source Val: {source_val_acc:.3f} | "
                  f"Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f}")

            # Save best model for this iteration
            if (avg_val_acc > best_avg_acc and
                source_val_acc >= min_source_acc and
                target_val_acc >= min_target_acc):
                best_avg_acc = avg_val_acc
                best_metrics = {
                    'iteration': iteration,
                    'best_epoch': epoch,
                    'avg_val_acc': avg_val_acc,
                    'source_val_acc': source_val_acc,
                    'target_val_acc': target_val_acc,
                    'source_val_loss': source_val_loss,
                    'target_val_loss': target_val_loss,
                    'source_control_acc': source_control_acc,
                    'source_dementia_acc': source_dementia_acc,
                    'target_control_acc': target_control_acc,
                    'target_dementia_acc': target_dementia_acc,
                    'source_f1': source_f1,
                    'target_f1': target_f1,
                    'pseudo_count': len(pseudo_features),
                    'pseudo_stats': pseudo_stats
                }
                torch.save(st_model.state_dict(), output_dir / f'st_iter_{iteration}.pth')
                patience_counter = 0
                print(f"  → Best model saved (Avg: {avg_val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        all_metrics.append(best_metrics if best_metrics else None)
        all_histories.append(history)

        # Update pseudo-generation model for next iteration
        if best_metrics:
            pseudo_gen_model = st_model

        if best_metrics:
            print(f"Iteration {iteration} completed: "
                  f"Avg {best_metrics['avg_val_acc']*100:.2f}% | "
                  f"Source {best_metrics['source_val_acc']*100:.2f}% | "
                  f"Target {best_metrics['target_val_acc']*100:.2f}%")

    # Find best iteration overall
    valid_metrics = [(i, m) for i, m in enumerate(all_metrics) if m is not None]
    if valid_metrics:
        best_iter_idx, best_iter_metrics = max(valid_metrics, key=lambda x: x[1].get('avg_val_acc', 0))
        print(f"\nPhase 2 completed. Best iteration: {best_iter_idx}, "
              f"Avg {best_iter_metrics['avg_val_acc']*100:.2f}%\n")
    else:
        print(f"\n⚠️  Phase 2: No valid iterations found")

    return all_metrics, all_histories, all_pseudo_stats


############################################################
# Complete DANN+ST Pipeline with Warmup
############################################################

def train_st_dann_parallel_egemaps(seed, source_train_dataset, target_train_dataset,
                                    source_val_loader, target_val_loader,
                                    output_dir, device,
                                    warmup_epochs=20, max_iterations=10, max_epochs_per_iter=60,
                                    confidence_threshold=0.9, min_pseudo_samples=5,
                                    domain_warmup_epochs=EGEMAPS_DOMAIN_WARMUP_EPOCHS,
                                    domain_anneal_ratio=EGEMAPS_DOMAIN_ANNEAL_RATIO,
                                    lambda_class=DANN_LAMBDA_CLASS, lambda_domain=EGEMAPS_LAMBDA_DOMAIN,
                                    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                    dropout=EGEMAPS_DROPOUT, dim_input=EGEMAPS_DIM_INPUT,
                                    dim_hidden=EGEMAPS_DIM_HIDDEN, batch_size=EGEMAPS_BATCH_SIZE,
                                    source_weight=0.5, target_weight=0.5,
                                    min_source_acc=0.0, min_target_acc=0.0,
                                    class_weight_control=DEFAULT_CLASS_WEIGHT_CONTROL,
                                    class_weight_dementia=DEFAULT_CLASS_WEIGHT_DEMENTIA):
    """
    Complete DANN+ST pipeline for eGeMAPS with warmup phase.

    Training Flow:
    1. Warmup: Train DANN on source only (warmup_epochs)
    2. Phase 1: Full DANN training (source + target adversarial)
    3. Phase 2: ST iterations (reinitialize from DANN checkpoint each time)

    Args:
        seed: Random seed
        source_train_dataset: Source training dataset
        target_train_dataset: Target training dataset
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Output directory
        device: Device
        warmup_epochs: Warmup phase epochs (DANN on source only) (default: 20)
        max_iterations: Max ST iterations (default: 10)
        max_epochs_per_iter: Max epochs per ST iteration (default: 60)
        confidence_threshold: Pseudo-label threshold (default: 0.9)
        min_pseudo_samples: Minimum pseudo-samples (default: 5)
        domain_warmup_epochs: Domain loss warmup epochs (default: 5)
        domain_anneal_ratio: Domain anneal ratio (default: 0)
        lambda_class: Classification loss weight (default: 1.0)
        lambda_domain: Domain loss weight (default: 1.0)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.3)
        dim_input: Input dimension (default: 25)
        dim_hidden: Hidden dimension (default: 32)
        batch_size: Batch size (default: 12)
        source_weight: Source validation weight (default: 0.5)
        target_weight: Target validation weight (default: 0.5)
        min_source_acc: Minimum source accuracy (default: 0.0)
        min_target_acc: Minimum target accuracy (default: 0.0)

    Returns:
        seed: Random seed used
        all_metrics: List of metrics for each ST iteration
        all_histories: List of training histories for each ST iteration
        all_pseudo_stats: List of pseudo-label statistics for each ST iteration
    """
    from dataset import create_dataloaders

    # Create output directory for this seed
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders for DANN training
    source_train_loader = create_dataloaders(
        source_train_dataset.csv_path, batch_size=batch_size, xlsr=False
    )
    target_train_loader = create_dataloaders(
        target_train_dataset.csv_path, batch_size=batch_size, xlsr=False
    )

    # ===== Warmup Phase: Train DANN on source only =====
    if warmup_epochs > 0:
        print(f"\nWARMUP: Training on source only ({warmup_epochs} epochs)")

        # Train DANN but only on source (simulate by using same loader for both)
        warmup_checkpoint, warmup_metrics = train_phase1_dann_egemaps(
            seed=seed,
            source_train_loader=source_train_loader,
            target_train_loader=source_train_loader,  # Use source as "target" (no real adaptation)
            source_val_loader=source_val_loader,
            target_val_loader=target_val_loader,
            output_dir=seed_dir,
            device=device,
            max_epochs=warmup_epochs,
            domain_warmup_epochs=warmup_epochs,  # Keep lambda=0 for entire warmup
            domain_anneal_ratio=0,
            lambda_domain=0,  # No domain loss in warmup
            lambda_class=lambda_class,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            source_weight=source_weight,
            target_weight=target_weight,
            min_source_acc=min_source_acc,
            min_target_acc=min_target_acc,
            class_weight_control=class_weight_control,
            class_weight_dementia=class_weight_dementia
        )

    # ===== Phase 1: Full DANN Training =====
    dann_checkpoint, phase1_metrics = train_phase1_dann_egemaps(
        seed=seed,
        source_train_loader=source_train_loader,
        target_train_loader=target_train_loader,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader,
        output_dir=seed_dir,
        device=device,
        max_epochs=max_epochs_per_iter,
        domain_warmup_epochs=domain_warmup_epochs,
        domain_anneal_ratio=domain_anneal_ratio,
        lambda_domain=lambda_domain,
        lambda_class=lambda_class,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        source_weight=source_weight,
        target_weight=target_weight,
        min_source_acc=min_source_acc,
        min_target_acc=min_target_acc,
        class_weight_control=class_weight_control,
        class_weight_dementia=class_weight_dementia
    )

    # ===== Phase 2: ST with DANN Initialization =====
    all_metrics, all_histories, all_pseudo_stats = train_phase2_st_with_dann_init_egemaps(
        seed=seed,
        dann_checkpoint_path=dann_checkpoint,
        source_train_dataset=source_train_dataset,
        target_train_dataset=target_train_dataset,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader,
        output_dir=seed_dir,
        device=device,
        max_iterations=max_iterations,
        max_epochs_per_iter=max_epochs_per_iter,
        confidence_threshold=confidence_threshold,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        source_weight=source_weight,
        target_weight=target_weight,
        min_source_acc=min_source_acc,
        min_target_acc=min_target_acc,
        patience=10,
        min_pseudo_samples=min_pseudo_samples,
        class_weight_control=class_weight_control,
        class_weight_dementia=class_weight_dementia
    )

    print(f"\nTraining Completed (Seed {seed})\n")

    return seed, all_metrics, all_histories, all_pseudo_stats

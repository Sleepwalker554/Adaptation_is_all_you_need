"""
DANN+ST Hybrid Training Pipeline

This module implements a two-phase domain adaptation approach:
- Phase 1: Train DANN model for domain-invariant features
- Phase 2: Run ST iterations, each reinitializing from DANN checkpoint

Key Innovation:
    Each ST iteration reinitializes from DANN checkpoint (not from scratch or incremental),
    preventing catastrophic forgetting while leveraging pseudo-labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, XLSR_DROPOUT
from config import DANN_LAMBDA_CLASS, DANN_LAMBDA_DOMAIN
from model import AD_XLSR_Model
from model_DANN import AD_XLSR_Model_DANN, compute_alpha
from model_DANN_ST import create_st_model_from_dann, verify_weight_transfer
from train_ST import generate_pseudo_labels, compute_class_weights
from dataset_ST import create_st_dataloaders


############################################################
# Phase 1: DANN Training
############################################################

def compute_domain_lambda(epoch, total_epochs, base_lambda=DANN_LAMBDA_DOMAIN,
                         warmup_epochs=30, anneal_ratio=0.25):
    """
    Schedule domain loss weight based on epoch.

    Schedule:
    - Warmup phase: First warmup_epochs epochs, lambda=0 (only train source classifier)
    - Anneal phase: Linearly increase to base_lambda over anneal_ratio proportion of epochs
    - Stable phase: Maintain base_lambda

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lambda: Maximum domain loss weight (default: from config)
        warmup_epochs: Number of warmup epochs (default: 30)
        anneal_ratio: Proportion of epochs for annealing (default: 0.25)

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


def train_one_epoch_dann(model, source_loader, target_loader, optimizer, device, epoch, total_epochs,
                         lambda_class=DANN_LAMBDA_CLASS, lambda_domain=DANN_LAMBDA_DOMAIN):
    """
    Train one epoch with DANN.

    Args:
        model: AD_XLSR_Model_DANN instance
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
        source_features, source_labels, source_masks = source_batch
        source_features = source_features.to(device)
        source_labels = source_labels.to(device)
        source_masks = source_masks.to(device)

        if lambda_active:
            class_output, domain_output = model(
                source_features,
                mask=source_masks,
                alpha=alpha,
                return_features=True
            )
        else:
            class_output = model(
                source_features,
                mask=source_masks,
                alpha=alpha,
                return_features=False
            )
            domain_output = None

        # Source domain label = 0
        source_domain_labels = torch.zeros(source_labels.size(0), dtype=torch.long).to(device)

        # Compute classification loss
        loss_class = F.cross_entropy(class_output, source_labels)

        # Compute domain loss if active
        if lambda_active:
            loss_domain_source = F.cross_entropy(domain_output, source_domain_labels)

            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_features, _, target_masks = target_batch  # Ignore target labels
            target_features = target_features.to(device)
            target_masks = target_masks.to(device)

            _, domain_output_target = model(
                target_features,
                mask=target_masks,
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


def validate_dann(model, val_loader, device):
    """
    Validate DANN model.

    Args:
        model: AD_XLSR_Model_DANN instance
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
        for features, labels, masks in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features, mask=masks, return_features=False)

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


def train_phase1_dann(seed, source_train_loader, target_train_loader,
                      source_val_loader, target_val_loader, output_dir, device,
                      max_epochs=60, domain_warmup_epochs=30, domain_anneal_ratio=0.25,
                      lambda_domain=DANN_LAMBDA_DOMAIN, lambda_class=DANN_LAMBDA_CLASS,
                      learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                      dropout=XLSR_DROPOUT, source_weight=0.5, target_weight=0.5,
                      min_source_acc=0.0, min_target_acc=0.0):
    """
    Phase 1: Train DANN model for domain-invariant features.

    Args:
        seed: Random seed
        source_train_loader: Source training data loader
        target_train_loader: Target training data loader
        source_val_loader: Source validation data loader
        target_val_loader: Target validation data loader
        output_dir: Output directory for checkpoints
        device: Device
        max_epochs: Maximum training epochs (default: 60)
        domain_warmup_epochs: Warmup epochs with no domain loss (default: 30)
        domain_anneal_ratio: Proportion of epochs for annealing (default: 0.25)
        lambda_domain: Domain loss weight (default: 0.1)
        lambda_class: Classification loss weight (default: 1.0)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.2)
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

    print(f"\n{'='*80}")
    print(f"PHASE 1: DANN Training (Seed {seed})")
    print(f"{'='*80}")
    print(f"Model Selection: Weighted Average (Source {source_weight:.2f}, Target {target_weight:.2f})")
    print(f"Accuracy Thresholds: Source >= {min_source_acc:.2f}, Target >= {min_target_acc:.2f}")
    print(f"Lambda Schedule: Warmup {domain_warmup_epochs} epochs → Anneal {domain_anneal_ratio*100:.0f}% → Stable {lambda_domain:.3f}")
    print(f"{'='*80}\n")

    # Create model and optimizer
    model = AD_XLSR_Model_DANN(dropout=dropout).to(device)
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
        class_loss, domain_loss, train_acc, domain_acc = train_one_epoch_dann(
            model, source_train_loader, target_train_loader, optimizer, device,
            epoch, max_epochs,
            lambda_class=lambda_class,
            lambda_domain=current_lambda_domain
        )

        # Validate on source and target
        source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = validate_dann(
            model, source_val_loader, device
        )

        target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = validate_dann(
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
        print(f"\n✓ Primary best model (with thresholds) at epoch {best_metrics['epoch']}:")
        print(f"  Avg: {best_metrics['avg_val_acc']*100:.2f}% | "
              f"Source: {best_metrics['source_val_acc']*100:.2f}% | "
              f"Target: {best_metrics['target_val_acc']*100:.2f}%")
        final_checkpoint = best_checkpoint_path
        final_metrics = best_metrics
    else:
        print(f"\n⚠ No model met thresholds (Source >= {min_source_acc:.2f}, Target >= {min_target_acc:.2f})")
        print(f"✓ Using fallback best model at epoch {fallback_best_metrics['epoch']}:")
        print(f"  Avg: {fallback_best_metrics['avg_val_acc']*100:.2f}% | "
              f"Source: {fallback_best_metrics['source_val_acc']*100:.2f}% | "
              f"Target: {fallback_best_metrics['target_val_acc']*100:.2f}%")
        final_checkpoint = fallback_checkpoint_path
        final_metrics = fallback_best_metrics

    print(f"\nPhase 1 (DANN) completed. Best checkpoint: {final_checkpoint}")

    phase1_metrics = {
        'history': history,
        'best_metrics': final_metrics,
        'best_checkpoint': str(final_checkpoint)
    }

    return str(final_checkpoint), phase1_metrics


############################################################
# Phase 2: Self-Training with DANN Initialization
############################################################

def train_one_epoch_st(model, train_loader, optimizer, device, epoch):
    """
    Train one epoch for ST (simple classification, no domain adversarial).

    Args:
        model: AD_XLSR_Model instance
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch

    Returns:
        avg_loss: Average loss
        train_acc: Training accuracy
    """
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"ST Epoch {epoch}", leave=False)

    for features, labels, masks in pbar:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Forward pass
        logits = model(features, mask=masks)
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total

    return avg_loss, train_acc


def validate_st(model, val_loader, device):
    """
    Validate ST model (same as validate_dann but for AD_XLSR_Model).

    Args:
        model: AD_XLSR_Model instance
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
        for features, labels, masks in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features, mask=masks)

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


def train_phase2_st_with_dann_init(seed, dann_checkpoint_path,
                                    source_train_dataset, target_train_dataset,
                                    source_val_loader, target_val_loader,
                                    output_dir, device,
                                    max_iterations=5, max_epochs_per_iter=60,
                                    confidence_threshold=0.9, learning_rate=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY, dropout=XLSR_DROPOUT,
                                    source_weight=0.5, target_weight=0.5,
                                    min_source_acc=0.0, min_target_acc=0.0,
                                    patience=10, min_pseudo_samples=5):
    """
    Phase 2: Self-Training with DANN initialization.

    Key: Each iteration reinitializes from DANN checkpoint to prevent catastrophic forgetting.

    Args:
        seed: Random seed
        dann_checkpoint_path: Path to best DANN checkpoint
        source_train_dataset: Source training dataset (not loader!)
        target_train_dataset: Target training dataset (not loader!)
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Output directory for checkpoints
        device: Device
        max_iterations: Maximum ST iterations (default: 5)
        max_epochs_per_iter: Max epochs per iteration (default: 60)
        confidence_threshold: Pseudo-label confidence threshold (default: 0.9)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.2)
        source_weight: Weight for source validation accuracy (default: 0.5)
        target_weight: Weight for target validation accuracy (default: 0.5)
        min_source_acc: Minimum source accuracy threshold (default: 0.0)
        min_target_acc: Minimum target accuracy threshold (default: 0.0)
        patience: Early stopping patience (default: 10)
        min_pseudo_samples: Minimum pseudo-labeled samples (default: 5)

    Returns:
        all_metrics: List of dicts for each iteration
        best_iteration: Index of best iteration
        best_checkpoint_path: Path to best ST checkpoint
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PHASE 2: Self-Training with DANN Initialization (Seed {seed})")
    print(f"{'='*80}")
    print(f"DANN Checkpoint: {dann_checkpoint_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Max Epochs/Iter: {max_epochs_per_iter}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Model Selection: Weighted Average (Source {source_weight:.2f}, Target {target_weight:.2f})")
    print(f"Accuracy Thresholds: Source >= {min_source_acc:.2f}, Target >= {min_target_acc:.2f}")
    print(f"{'='*80}\n")

    all_metrics = []

    # Best model tracking (primary: with thresholds)
    best_avg_acc = 0
    best_iteration = -1
    best_checkpoint_path = None

    # Fallback tracking (best avg regardless of thresholds)
    fallback_best_avg_acc = 0
    fallback_best_iteration = -1
    fallback_checkpoint_path = None

    # Start from iteration 1 (skip warmup iteration 0, use DANN directly)
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ST Iteration {iteration}/{max_iterations}")
        print(f"{'='*80}")

        # ===== Step 1: Create ST model from DANN checkpoint =====
        print(f"\nStep 1: Initializing ST model from DANN checkpoint...")
        model = create_st_model_from_dann(dann_checkpoint_path, device, dropout=dropout)

        # ===== Step 2: Generate pseudo-labels =====
        print(f"\nStep 2: Generating pseudo-labels (threshold >= {confidence_threshold})...")
        selected_features, selected_labels, selected_session_ids, selected_confidences, stats = \
            generate_pseudo_labels(model, target_train_dataset, device, confidence_threshold)

        print(f"\nPseudo-label Statistics:")
        print(f"  Total target samples: {stats['total_samples']}")
        print(f"  Selected samples: {stats['selected_count']}")
        
        # Check minimum pseudo-samples
        if stats['selected_count'] < min_pseudo_samples:
            print(f"\n⚠ Insufficient pseudo-labeled samples ({stats['selected_count']} < {min_pseudo_samples})")
            print(f"Stopping ST iterations at iteration {iteration-1}")
            break

        # ===== Step 3: Create combined dataset =====
        print(f"\nStep 3: Creating combined dataset (source + pseudo-labeled target)...")
        from dataset_ST import PseudoLabelDataset
        pseudo_dataset = PseudoLabelDataset(
            features_list=selected_features,
            pseudo_labels_list=selected_labels,
            session_ids_list=selected_session_ids,
            confidences_list=selected_confidences
        )
        combined_train_loader = create_st_dataloaders(
            source_dataset=source_train_dataset,
            pseudo_dataset=pseudo_dataset
        )

        # Compute class weights
        class_weights = compute_class_weights(combined_train_loader.dataset, device)
        print(f"Class weights: Control={class_weights[0]:.3f}, Dementia={class_weights[1]:.3f}")

        # ===== Step 4: Train ST model =====
        print(f"\nStep 4: Training ST model ({max_epochs_per_iter} epochs)...")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training history for this iteration
        iter_history = {
            'epochs': [],
            'train_losses': [],
            'train_accs': [],
            'source_val_accs': [],
            'target_val_accs': [],
            'avg_val_accs': []
        }

        # Early stopping
        iter_best_avg_acc = 0
        no_improve_count = 0

        for epoch in range(1, max_epochs_per_iter + 1):
            # Train
            train_loss, train_acc = train_one_epoch_st(model, combined_train_loader, optimizer, device, epoch)

            # Validate on source and target
            source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = \
                validate_st(model, source_val_loader, device)

            target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = \
                validate_st(model, target_val_loader, device)

            # Weighted average
            avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc

            # Update history
            iter_history['epochs'].append(epoch)
            iter_history['train_losses'].append(train_loss)
            iter_history['train_accs'].append(train_acc)
            iter_history['source_val_accs'].append(source_val_acc)
            iter_history['target_val_accs'].append(target_val_acc)
            iter_history['avg_val_accs'].append(avg_val_acc)

            print(f"  Epoch {epoch:3d} | Train: {train_acc:.3f} | Source Val: {source_val_acc:.3f} | "
                  f"Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f}")

            # Track best for this iteration (for early stopping)
            if avg_val_acc > iter_best_avg_acc:
                iter_best_avg_acc = avg_val_acc
                iter_best_source_acc = source_val_acc
                iter_best_target_acc = target_val_acc
                iter_best_epoch = epoch
                no_improve_count = 0

                # Save iteration checkpoint
                iter_checkpoint = output_dir / f'st_iter_{iteration}_best.pth'
                torch.save({
                    'iteration': iteration,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_val_acc': avg_val_acc,
                    'source_val_acc': source_val_acc,
                    'target_val_acc': target_val_acc,
                }, iter_checkpoint)
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # ===== Step 5: Evaluate iteration =====
        print(f"\nIteration {iteration} completed:")
        print(f"  Best Epoch: {iter_best_epoch}")
        print(f"  Best Avg Acc: {iter_best_avg_acc:.4f} (Source: {iter_best_source_acc:.4f}, Target: {iter_best_target_acc:.4f})")

        # Store metrics
        iter_metrics = {
            'iteration': iteration,
            'best_epoch': iter_best_epoch,
            'avg_val_acc': iter_best_avg_acc,
            'source_val_acc': iter_best_source_acc,
            'target_val_acc': iter_best_target_acc,
            'pseudo_count': stats['selected_count'],
            'history': iter_history
        }
        all_metrics.append(iter_metrics)

        # Primary: Check if best across all iterations (with thresholds)
        if (iter_best_avg_acc > best_avg_acc and
            iter_best_source_acc >= min_source_acc and
            iter_best_target_acc >= min_target_acc):
            best_avg_acc = iter_best_avg_acc
            best_iteration = iteration
            best_checkpoint_path = iter_checkpoint
            print(f"  → Primary best model across all iterations!")

        # Fallback: Best avg regardless of thresholds
        if iter_best_avg_acc > fallback_best_avg_acc:
            fallback_best_avg_acc = iter_best_avg_acc
            fallback_best_iteration = iteration
            fallback_checkpoint_path = iter_checkpoint

    # Determine final best
    if best_checkpoint_path is not None:
        print(f"\n{'='*80}")
        print(f"✓ Primary best model (with thresholds): Iteration {best_iteration}")
        print(f"  Avg: {best_avg_acc*100:.2f}%")
        print(f"  Checkpoint: {best_checkpoint_path}")
        print(f"{'='*80}\n")
        final_best_iteration = best_iteration
        final_checkpoint = best_checkpoint_path
    else:
        print(f"\n{'='*80}")
        print(f"⚠ No iteration met thresholds (Source >= {min_source_acc:.2f}, Target >= {min_target_acc:.2f})")
        print(f"✓ Using fallback best model: Iteration {fallback_best_iteration}")
        print(f"  Avg: {fallback_best_avg_acc*100:.2f}%")
        print(f"  Checkpoint: {fallback_checkpoint_path}")
        print(f"{'='*80}\n")
        final_best_iteration = fallback_best_iteration
        final_checkpoint = fallback_checkpoint_path

    return all_metrics, final_best_iteration, str(final_checkpoint)


############################################################
# Complete DANN+ST Pipeline
############################################################

def train_dann_st(seed, source_train_loader, target_train_loader,
                  source_train_dataset, target_train_dataset,
                  source_val_loader, target_val_loader, output_dir, device,
                  # Phase 1 params
                  max_epochs_phase1=60, domain_warmup_epochs=30, domain_anneal_ratio=0.25,
                  lambda_domain_phase1=DANN_LAMBDA_DOMAIN, lambda_class_phase1=DANN_LAMBDA_CLASS,
                  # Phase 2 params
                  max_iterations=5, max_epochs_per_iter=60, confidence_threshold=0.9,
                  # Shared params
                  learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, dropout=XLSR_DROPOUT,
                  # Model selection params
                  source_weight=0.5, target_weight=0.5,
                  min_source_acc=0.0, min_target_acc=0.0,
                  # ST params
                  patience=10, min_pseudo_samples=5):
    """
    Complete DANN+ST hybrid training pipeline.

    Args:
        seed: Random seed
        source_train_loader: Source training loader
        target_train_loader: Target training loader
        source_train_dataset: Source training dataset (for ST)
        target_train_dataset: Target training dataset (for ST)
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Output directory
        device: Device

        # Phase 1 (DANN) parameters
        max_epochs_phase1: Max epochs for Phase 1 (default: 60)
        domain_warmup_epochs: Warmup epochs (default: 30)
        domain_anneal_ratio: Anneal ratio (default: 0.25)
        lambda_domain_phase1: Domain loss weight (default: 0.1)
        lambda_class_phase1: Classification loss weight (default: 1.0)

        # Phase 2 (ST) parameters
        max_iterations: Max ST iterations (default: 5)
        max_epochs_per_iter: Max epochs per ST iteration (default: 60)
        confidence_threshold: Pseudo-label threshold (default: 0.9)

        # Shared parameters
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay (default: 1e-2)
        dropout: Dropout rate (default: 0.2)

        # Model selection parameters
        source_weight: Source validation weight (default: 0.5)
        target_weight: Target validation weight (default: 0.5)
        min_source_acc: Minimum source accuracy (default: 0.0)
        min_target_acc: Minimum target accuracy (default: 0.0)

        # ST-specific parameters
        patience: Early stopping patience (default: 10)
        min_pseudo_samples: Minimum pseudo-samples (default: 5)

    Returns:
        results: Dict with phase1_metrics, phase2_metrics, best_checkpoint_path
    """
    print(f"\n{'#'*80}")
    print(f"DANN+ST Hybrid Training Pipeline (Seed {seed})")
    print(f"{'#'*80}\n")

    # ===== Phase 1: DANN Training =====
    dann_checkpoint_path, phase1_metrics = train_phase1_dann(
        seed=seed,
        source_train_loader=source_train_loader,
        target_train_loader=target_train_loader,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader,
        output_dir=output_dir,
        device=device,
        max_epochs=max_epochs_phase1,
        domain_warmup_epochs=domain_warmup_epochs,
        domain_anneal_ratio=domain_anneal_ratio,
        lambda_domain=lambda_domain_phase1,
        lambda_class=lambda_class_phase1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        source_weight=source_weight,
        target_weight=target_weight,
        min_source_acc=min_source_acc,
        min_target_acc=min_target_acc
    )

    # ===== Phase 2: ST with DANN Initialization =====
    all_st_metrics, best_st_iteration, best_st_checkpoint = train_phase2_st_with_dann_init(
        seed=seed,
        dann_checkpoint_path=dann_checkpoint_path,
        source_train_dataset=source_train_dataset,
        target_train_dataset=target_train_dataset,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader,
        output_dir=output_dir,
        device=device,
        max_iterations=max_iterations,
        max_epochs_per_iter=max_epochs_per_iter,
        confidence_threshold=confidence_threshold,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        source_weight=source_weight,
        target_weight=target_weight,
        min_source_acc=min_source_acc,
        min_target_acc=min_target_acc,
        patience=patience,
        min_pseudo_samples=min_pseudo_samples
    )

    # ===== Summary =====
    print(f"\n{'#'*80}")
    print(f"DANN+ST Pipeline Completed (Seed {seed})")
    print(f"{'#'*80}")
    print(f"\nPhase 1 (DANN):")
    print(f"  Best Epoch: {phase1_metrics['best_metrics']['epoch']}")
    print(f"  Avg Acc: {phase1_metrics['best_metrics']['avg_val_acc']*100:.2f}%")
    print(f"  Source: {phase1_metrics['best_metrics']['source_val_acc']*100:.2f}%")
    print(f"  Target: {phase1_metrics['best_metrics']['target_val_acc']*100:.2f}%")

    if all_st_metrics:
        best_st_metrics = all_st_metrics[best_st_iteration - 1]
        print(f"\nPhase 2 (ST):")
        print(f"  Best Iteration: {best_st_iteration}")
        print(f"  Best Epoch: {best_st_metrics['best_epoch']}")
        print(f"  Avg Acc: {best_st_metrics['avg_val_acc']*100:.2f}%")
        print(f"  Source: {best_st_metrics['source_val_acc']*100:.2f}%")
        print(f"  Target: {best_st_metrics['target_val_acc']*100:.2f}%")
        print(f"  Pseudo-samples: {best_st_metrics['pseudo_count']}")
    else:
        print(f"\nPhase 2 (ST): No valid iterations")

    print(f"\nFinal Best Checkpoint: {best_st_checkpoint}")
    print(f"{'#'*80}\n")

    results = {
        'phase1_metrics': phase1_metrics,
        'phase2_metrics': {
            'all_iterations': all_st_metrics,
            'best_iteration': best_st_iteration,
            'best_checkpoint': best_st_checkpoint
        },
        'best_checkpoint_path': best_st_checkpoint
    }

    return results

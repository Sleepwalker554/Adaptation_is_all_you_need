"""
Self-Training with DANN Parallel Training (Scheme 2)

This module implements ST where EVERY iteration uses DANN:
- Iteration 0: source only + DANN (warmup)
- Iteration 1+: source + pseudo-labels + DANN

Key differences from sequential approach (Scheme 1):
- Uses AD_XLSR_Model_DANN throughout (not switching to AD_XLSR_Model)
- Domain adversarial loss is active in ALL ST iterations
- Pseudo-labels and domain alignment are optimized simultaneously

Architecture:
    Each ST iteration:
    1. Generate pseudo-labels using current DANN model
    2. Train DANN on source + pseudo-labeled target
    3. Domain adversarial training continues (alignment never stops)
    4. Save best model for this iteration
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import LEARNING_RATE, WEIGHT_DECAY, XLSR_DROPOUT, XLSR_MAX_TIME_STEPS
from config import DANN_LAMBDA_CLASS, DANN_LAMBDA_DOMAIN
from model_DANN import AD_XLSR_Model_DANN, compute_alpha
from dataset import create_dataloaders
from dataset_ST import PseudoLabelDataset


############################################################
# Pseudo-Label Generation with DANN Model
############################################################

def generate_pseudo_labels_dann(model, target_dataset, device, confidence_threshold=0.9):
    """
    Generate pseudo-labels for target domain using DANN model
    
    Args:
        model: AD_XLSR_Model_DANN instance
        target_dataset: FeatureDataset for target domain (unlabeled)
        device: Device
        confidence_threshold: Minimum confidence for selection
    
    Returns:
        selected_features: List of feature tensors
        selected_labels: List of pseudo-labels (0 or 1)
        selected_session_ids: List of session IDs
        selected_confidences: List of confidence scores
        stats: Dictionary with generation statistics
    """
    model.eval()
    
    selected_features = []
    selected_labels = []
    selected_session_ids = []
    selected_confidences = []
    
    # Track statistics
    all_confidences = []
    class_0_count = 0  # Control
    class_1_count = 0  # Dementia
    rejected_count = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(target_dataset)), desc="Pseudo-labeling", leave=False):
            features, _ = target_dataset[idx]  # Ignore original label (if any)
            features = features.unsqueeze(0).to(device)  # (1, seq_len, 1024)
            
            # Create mask for single sample
            seq_len = features.shape[1]
            if seq_len > XLSR_MAX_TIME_STEPS:
                features = features[:, :XLSR_MAX_TIME_STEPS, :]
                mask = torch.ones(1, XLSR_MAX_TIME_STEPS).to(device)
            else:
                mask = torch.ones(1, seq_len).to(device)
                if seq_len < XLSR_MAX_TIME_STEPS:
                    padding = torch.zeros(1, XLSR_MAX_TIME_STEPS - seq_len, features.shape[2]).to(device)
                    features = torch.cat([features, padding], dim=1)
                    temp_mask = torch.zeros(1, XLSR_MAX_TIME_STEPS).to(device)
                    temp_mask[:, :seq_len] = 1
                    mask = temp_mask
            
            # Forward pass (return_features=False to get only class logits)
            logits = model(features, mask, alpha=0.0, return_features=False)  # (1, 2)
            probs = F.softmax(logits, dim=1)  # (1, 2)
            
            # Get prediction and confidence
            confidence, predicted_class = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            
            all_confidences.append(confidence)
            
            # Select high-confidence predictions
            if confidence >= confidence_threshold:
                # Store original features (before padding/truncation)
                original_features, _ = target_dataset[idx]
                selected_features.append(original_features)
                selected_labels.append(predicted_class)
                selected_session_ids.append(f"target_{idx}")
                selected_confidences.append(confidence)
                
                if predicted_class == 0:
                    class_0_count += 1
                else:
                    class_1_count += 1
            else:
                rejected_count += 1
    
    # Statistics
    stats = {
        'total_samples': len(target_dataset),
        'selected_count': len(selected_features),
        'rejected_count': rejected_count,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'avg_confidence': np.mean(selected_confidences) if selected_confidences else 0.0,
        'min_confidence': min(selected_confidences) if selected_confidences else 0.0,
        'max_confidence': max(selected_confidences) if selected_confidences else 0.0,
        'all_avg_confidence': np.mean(all_confidences) if all_confidences else 0.0,
    }
    
    print(f"Pseudo-labels: {stats['selected_count']}/{stats['total_samples']} ({stats['selected_count']/stats['total_samples']*100:.1f}%) | C0:{class_0_count} C1:{class_1_count} | conf:{stats['avg_confidence']:.3f}")
    
    return selected_features, selected_labels, selected_session_ids, selected_confidences, stats


############################################################
# Training Functions for ST+DANN Parallel
############################################################

def compute_domain_lambda(epoch, total_epochs, base_lambda=DANN_LAMBDA_DOMAIN,
                         warmup_epochs=30, anneal_ratio=0.25):
    """
    Schedule domain loss weight based on epoch.
    
    Same schedule as DANN:
    - Warmup phase: First warmup_epochs, lambda=0
    - Anneal phase: Linearly increase to base_lambda
    - Stable phase: Maintain base_lambda
    """
    warmup_epochs = min(warmup_epochs, total_epochs)
    anneal_epochs = int(total_epochs * anneal_ratio)
    
    if epoch <= warmup_epochs:
        return 0.0
    
    if anneal_epochs <= 0:
        return base_lambda
    
    progress = min(1.0, max(0, epoch - warmup_epochs) / max(1, anneal_epochs))
    return base_lambda * progress


def train_one_epoch_st_dann(model, source_loader, target_loader, optimizer, device,
                             epoch, total_epochs, lambda_class=DANN_LAMBDA_CLASS,
                             lambda_domain=DANN_LAMBDA_DOMAIN):
    """
    Train one epoch with ST+DANN parallel approach
    
    Key: Combines classification on source+pseudo AND domain adversarial on source+all_target
    
    Args:
        model: AD_XLSR_Model_DANN instance
        source_loader: Source domain with real labels (can include pseudo-labeled samples)
        target_loader: Target domain for domain adversarial (all samples, unlabeled)
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        total_epochs: Total epochs
        lambda_class: Classification loss weight
        lambda_domain: Domain adversarial loss weight
    
    Returns:
        avg_class_loss: Average classification loss
        avg_domain_loss: Average domain loss
        train_acc: Training accuracy (on labeled data)
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
        
        # ===== Process source domain data (real labels or pseudo-labels) =====
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
            
            # Get target batch for domain adversarial
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            target_features, _, target_masks = target_batch  # Ignore labels
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
    Validate DANN model
    
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


############################################################
# Complete ST+DANN Parallel Training Pipeline
############################################################

def train_st_dann_parallel(seed, source_train_dataset, target_train_dataset,
                           source_val_loader, target_val_loader,
                           output_dir, device,
                           warmup_epochs=20,
                           max_iterations=5,
                           max_epochs_per_iter=60,
                           confidence_threshold=0.9,
                           min_pseudo_samples=5,
                           domain_warmup_epochs=30,
                           domain_anneal_ratio=0.25,
                           lambda_class=DANN_LAMBDA_CLASS,
                           lambda_domain=DANN_LAMBDA_DOMAIN,
                           learning_rate=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY,
                           dropout=XLSR_DROPOUT,
                           source_weight=0.5,
                           target_weight=0.5,
                           min_source_acc=0.0,
                           min_target_acc=0.0,
                           patience=10):
    """
    Complete ST+DANN parallel training pipeline
    
    Training flow:
    1. Global warmup: Train DANN on source only (warmup_epochs)
    2. For each ST iteration:
       a. Generate pseudo-labels using global warmup model
       b. Train with source + pseudo-labels starting from global warmup model
    
    Each iteration independently starts from the same global warmup model to avoid
    error accumulation and overfitting to pseudo-labels.
    
    Args:
        seed: Random seed
        source_train_dataset: Source training dataset
        target_train_dataset: Target training dataset
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Output directory
        device: Device
        warmup_epochs: Global warmup epochs (DANN on source only)
        iter_warmup_epochs: (DEPRECATED - not used, kept for compatibility)
        max_iterations: Maximum ST iterations
        max_epochs_per_iter: Max epochs per iteration
        confidence_threshold: Pseudo-label confidence threshold
        min_pseudo_samples: Minimum pseudo-labeled samples required
        domain_warmup_epochs: Warmup epochs for domain loss (lambda=0)
        domain_anneal_ratio: Anneal ratio for domain loss
        lambda_class: Classification loss weight
        lambda_domain: Domain loss weight
        learning_rate: Learning rate
        weight_decay: Weight decay
        dropout: Dropout rate
        source_weight: Source validation weight
        target_weight: Target validation weight
        min_source_acc: Minimum source accuracy threshold (for ST iterations)
        min_target_acc: Minimum target accuracy threshold (for ST iterations)
        patience: Early stopping patience
    
    Returns:
        seed: Random seed
        all_metrics: List of metrics for each iteration
        all_histories: List of training histories
        all_pseudo_stats: List of pseudo-label statistics
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ST+DANN Parallel Training (Seed {seed})")
    print(f"{'='*80}")
    
    all_metrics = []
    all_histories = []
    all_pseudo_stats = []
    
    # Track global best
    global_best_avg_acc = 0
    global_best_iteration = -1
    
    # Create loaders (with drop_last=True for training to avoid BatchNorm issues)
    source_train_loader = create_dataloaders(source_train_dataset.csv_path, xlsr=True, drop_last=True)
    target_train_loader = create_dataloaders(target_train_dataset.csv_path, xlsr=True, drop_last=True)
    
    # ========== WARMUP PHASE: Train DANN on source only ==========
    print(f"\n{'='*70}")
    print(f"WARMUP: Training DANN on source only ({warmup_epochs} epochs)")
    
    model = AD_XLSR_Model_DANN(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    warmup_history = {
        'epochs': [],
        'train_class_losses': [],
        'train_domain_losses': [],
        'train_accs': [],
        'source_val_accs': [],
        'target_val_accs': [],
        'avg_val_accs': [],
        'domain_accs': [],
        'lambda_domains': []
    }
    
    warmup_best_avg_acc = 0
    warmup_best_epoch = -1
    warmup_model_path = seed_dir / 'warmup_best.pth'
    
    for epoch in range(1, warmup_epochs + 1):
        # Compute domain lambda with warmup schedule
        current_lambda_domain = compute_domain_lambda(
            epoch, warmup_epochs,
            base_lambda=lambda_domain,
            warmup_epochs=domain_warmup_epochs,
            anneal_ratio=domain_anneal_ratio
        )
        
        # Train one epoch
        class_loss, domain_loss, train_acc, domain_acc = train_one_epoch_st_dann(
            model, source_train_loader, target_train_loader, optimizer, device,
            epoch, warmup_epochs,
            lambda_class=lambda_class,
            lambda_domain=current_lambda_domain
        )
        
        # Validate
        source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = \
            validate_dann(model, source_val_loader, device)
        
        target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = \
            validate_dann(model, target_val_loader, device)
        
        avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc
        
        # Update history
        warmup_history['epochs'].append(epoch)
        warmup_history['train_class_losses'].append(class_loss)
        warmup_history['train_domain_losses'].append(domain_loss)
        warmup_history['train_accs'].append(train_acc)
        warmup_history['source_val_accs'].append(source_val_acc)
        warmup_history['target_val_accs'].append(target_val_acc)
        warmup_history['avg_val_accs'].append(avg_val_acc)
        warmup_history['domain_accs'].append(domain_acc)
        warmup_history['lambda_domains'].append(current_lambda_domain)
        
        print(f"Epoch {epoch:3d} | Train: {train_acc:.3f} | Source Val: {source_val_acc:.3f} | "
              f"Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f} | Domain: {domain_acc:.3f} | "
              f"λ_dom: {current_lambda_domain:.3f}")
        
        # Save best warmup model (no threshold requirement)
        if avg_val_acc > warmup_best_avg_acc:
            warmup_best_avg_acc = avg_val_acc
            warmup_best_epoch = epoch
            torch.save(model.state_dict(), warmup_model_path)
    
    print(f"Warmup Best: Epoch {warmup_best_epoch} | Avg:{warmup_best_avg_acc*100:.2f}%")
    
    # ========== ST ITERATIONS ==========
    # Strategy: Each iteration starts from global warmup (not from previous iteration)
    # This avoids error accumulation and overfitting to pseudo-labels
    
    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{max_iterations-1}")
        
        # ===== Step 1: Generate pseudo-labels using global warmup model =====
        print(f"Generating pseudo-labels...")
        pseudo_label_model = AD_XLSR_Model_DANN(dropout=dropout).to(device)
        pseudo_label_model.load_state_dict(torch.load(warmup_model_path, map_location=device))
        
        selected_features, selected_labels, selected_session_ids, selected_confidences, pseudo_stats = \
            generate_pseudo_labels_dann(pseudo_label_model, target_train_dataset, device, confidence_threshold)
        
        del pseudo_label_model  # Free memory
        
        # Check minimum pseudo-samples
        if pseudo_stats['selected_count'] < min_pseudo_samples:
            print(f"⚠ Insufficient pseudo-samples ({pseudo_stats['selected_count']} < {min_pseudo_samples}), stopping")
            break
        
        # Create pseudo-label dataset
        pseudo_dataset = PseudoLabelDataset(
            features_list=selected_features,
            pseudo_labels_list=selected_labels,
            session_ids_list=selected_session_ids,
            confidences_list=selected_confidences
        )
        
        # Create combined source + pseudo loader
        from dataset_ST import create_st_dataloaders
        combined_train_loader = create_st_dataloaders(
            source_dataset=source_train_dataset,
            pseudo_dataset=pseudo_dataset,
            seed=seed
        )
        
        all_pseudo_stats.append(pseudo_stats)
        
        # ===== Step 2: Train with source + pseudo-labels =====
        print(f"Training with pseudo-labels...")
        
        # Initialize model from global warmup
        model = AD_XLSR_Model_DANN(dropout=dropout).to(device)
        model.load_state_dict(torch.load(warmup_model_path, map_location=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        history = {
            'epochs': [],
            'train_class_losses': [],
            'train_domain_losses': [],
            'train_accs': [],
            'source_val_accs': [],
            'target_val_accs': [],
            'avg_val_accs': [],
            'domain_accs': [],
            'lambda_domains': []
        }
        
        # Best model tracking for this iteration
        iter_best_avg_acc = 0
        iter_best_metrics = {}
        no_improve_count = 0
        
        for epoch in range(1, max_epochs_per_iter + 1):
            # Use full lambda from start (warmup already done)
            current_lambda_domain = lambda_domain
            
            # Train one epoch
            class_loss, domain_loss, train_acc, domain_acc = train_one_epoch_st_dann(
                model, combined_train_loader, target_train_loader, optimizer, device,
                epoch, max_epochs_per_iter,
                lambda_class=lambda_class,
                lambda_domain=current_lambda_domain
            )
            
            # Validate on source and target
            source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = \
                validate_dann(model, source_val_loader, device)
            
            target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = \
                validate_dann(model, target_val_loader, device)
            
            # Weighted average
            avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc
            
            # Update history
            history['epochs'].append(epoch)
            history['train_class_losses'].append(class_loss)
            history['train_domain_losses'].append(domain_loss)
            history['train_accs'].append(train_acc)
            history['source_val_accs'].append(source_val_acc)
            history['target_val_accs'].append(target_val_acc)
            history['avg_val_accs'].append(avg_val_acc)
            history['domain_accs'].append(domain_acc)
            history['lambda_domains'].append(current_lambda_domain)
            
            print(f"Epoch {epoch:3d} | Train: {train_acc:.3f} | Source Val: {source_val_acc:.3f} | "
                  f"Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f} | Domain: {domain_acc:.3f} | "
                  f"λ_dom: {current_lambda_domain:.3f}")
            
            # Check if current model is best for this iteration
            if (avg_val_acc > iter_best_avg_acc and
                source_val_acc >= min_source_acc and
                target_val_acc >= min_target_acc):
                iter_best_avg_acc = avg_val_acc
                iter_best_metrics = {
                    'iteration': iteration,
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
                    'pseudo_count': pseudo_stats['selected_count']
                }
                # Save checkpoint
                checkpoint_path = seed_dir / f'st_dann_iter_{iteration}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        
        # ===== Step 3: Record iteration results =====
        if iter_best_metrics:
            print(f"Iter {iteration} Best: Epoch {iter_best_metrics['epoch']} | Avg:{iter_best_metrics['avg_val_acc']*100:.2f}% | Src:{iter_best_metrics['source_val_acc']*100:.2f}% | Tgt:{iter_best_metrics['target_val_acc']*100:.2f}% | Pseudo:{pseudo_stats['selected_count']}")
            
            # Check global best
            if iter_best_metrics['avg_val_acc'] > global_best_avg_acc:
                global_best_avg_acc = iter_best_metrics['avg_val_acc']
                global_best_iteration = iteration
                print(f"  → New global best!")
        else:
            print(f"Iter {iteration}: No valid model (thresholds not met)")
            iter_best_metrics = {
                'iteration': iteration,
                'avg_val_acc': 0.0,
                'pseudo_count': pseudo_stats['selected_count']
            }
        
        all_metrics.append(iter_best_metrics)
        all_histories.append(history)
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    if global_best_iteration >= 0:
        print(f"Best: Iter {global_best_iteration} | Avg:{global_best_avg_acc*100:.2f}% | Src:{all_metrics[global_best_iteration]['source_val_acc']*100:.2f}% | Tgt:{all_metrics[global_best_iteration]['target_val_acc']*100:.2f}%")
    else:
        print("No valid iterations (thresholds not met)")
    print(f"{'='*70}\n")
    
    return seed, all_metrics, all_histories, all_pseudo_stats

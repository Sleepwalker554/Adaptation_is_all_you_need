"""
Self-Training Functions for Domain Adaptation

This module implements the self-training algorithm:
1. Train model on source domain
2. Generate pseudo-labels for target domain
3. Select high-confidence predictions
4. Retrain on source + pseudo-labeled target
5. Repeat until convergence or max iterations

Key differences from DANN:
- No domain adversarial training
- Uses simple AD_XLSR_Model (not DANN version)
- Iterative refinement via pseudo-labeling
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from config import LEARNING_RATE, MAX_EPOCHS, WEIGHT_DECAY, XLSR_DROPOUT, XLSR_MAX_TIME_STEPS
from model import AD_XLSR_Model
from train import train_one_epoch, validate
from dataset_ST import PseudoLabelDataset, create_st_dataloaders


# ====== Self-Training Hyperparameters ======
ST_MAX_ITERATIONS = 5           # Maximum ST iterations (rounds of pseudo-labeling)
ST_CONFIDENCE_THRESHOLD = 0.9   # Minimum confidence for pseudo-label selection
ST_MIN_PSEUDO_SAMPLES = 5       # Minimum pseudo-labeled samples required per iteration
ST_PATIENCE = 10                # Early stopping patience for each iteration


def generate_pseudo_labels(model, target_dataset, device, confidence_threshold=ST_CONFIDENCE_THRESHOLD):
    """
    Generate pseudo-labels for target domain data

    Algorithm:
    1. Run model inference on all target samples
    2. Compute softmax probabilities
    3. Select predictions with max probability >= threshold
    4. Return selected samples with hard pseudo-labels (argmax)

    Args:
        model: Trained AD_XLSR_Model
        target_dataset: FeatureDataset for target domain (unlabeled)
        device: Device
        confidence_threshold: Minimum confidence (probability) for selection

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

            # Forward pass
            logits = model(features, mask)  # (1, 2)
            probs = F.softmax(logits, dim=1)  # (1, 2)

            # Get prediction and confidence
            confidence, predicted_class = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            all_confidences.append(confidence)

            # Select if confidence exceeds threshold
            if confidence >= confidence_threshold:
                # Store original features (without padding) for later use
                original_features, _ = target_dataset[idx]
                selected_features.append(original_features)
                selected_labels.append(predicted_class)
                selected_session_ids.append(target_dataset.session_ids[idx])
                selected_confidences.append(confidence)

                if predicted_class == 0:
                    class_0_count += 1
                else:
                    class_1_count += 1
            else:
                rejected_count += 1

    # Compute statistics
    stats = {
        'total_target_samples': len(target_dataset),
        'selected_count': len(selected_labels),
        'rejected_count': rejected_count,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'avg_selected_confidence': np.mean(selected_confidences) if selected_confidences else 0.0,
        'avg_all_confidence': np.mean(all_confidences),
        'min_confidence': np.min(all_confidences),
        'max_confidence': np.max(all_confidences),
        'class_balance_ratio': class_0_count / class_1_count if class_1_count > 0 else float('inf')
    }

    print(f"\nPseudo-label Generation (threshold={confidence_threshold:.2f}):")
    print(f"  Total target samples: {stats['total_target_samples']}, Selected: {stats['selected_count']} ({stats['selected_count']/stats['total_target_samples']*100:.1f}%)")

    return selected_features, selected_labels, selected_session_ids, selected_confidences, stats


def train_st_iteration(iteration, model, source_dataset, pseudo_dataset, source_val_loader, target_val_loader,
                      device, seed, max_epochs=MAX_EPOCHS, patience=ST_PATIENCE,
                      manual_weight_control=None, manual_weight_dementia=None,
                      source_weight=0.5, target_weight=0.5):
    """
    Train model for one ST iteration on source + pseudo-labeled data

    Args:
        iteration: Current ST iteration number (0 = initial training on source only)
        model: AD_XLSR_Model (freshly initialized or loaded)
        source_dataset: FeatureDataset with real source labels
        pseudo_dataset: PseudoLabelDataset with pseudo-labels (can be empty for iteration 0)
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        device: Device
        seed: Random seed
        max_epochs: Max epochs for this iteration
        patience: Early stopping patience
        manual_weight_control: Manual weight for Control class (None = auto-compute)
        manual_weight_dementia: Manual weight for Dementia class (None = auto-compute)
        source_weight: Weight for source domain in avg accuracy (default: 0.5)
        target_weight: Weight for target domain in avg accuracy (default: 0.5)

    Returns:
        trained_model: Trained model
        best_metrics: Dict with best validation metrics
        history: Training history
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Create combined dataloader
    train_loader = create_st_dataloaders(source_dataset, pseudo_dataset, seed=seed)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training history
    history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'source_val_losses': [],
        'source_val_accs': [],
        'target_val_accs': [],
        'target_val_losses': [],
        'avg_val_accs': [],
        'source_f1s': [],
        'target_f1s': []
    }

    # Model selection: weighted average accuracy (no thresholds)
    best_avg_acc = float('-inf')
    best_metrics = {}
    patience_counter = 0
    best_epoch = 0
    best_state_dict = None

    print(f"\n{'='*60}")
    print(f"ST Iteration {iteration} - Training")
    print(f"Model Selection: Avg Acc (Source {source_weight:.1f}, Target {target_weight:.1f})")
    print(f"{'='*60}")

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device,
                                                 epoch=epoch)

        # Validate on both domains
        source_val_loss, source_val_acc, source_control_acc, source_dementia_acc, source_f1 = validate(
            model, source_val_loader, device, epoch=epoch
        )

        target_val_loss, target_val_acc, target_control_acc, target_dementia_acc, target_f1 = validate(
            model, target_val_loader, device, epoch=epoch
        )

        # Compute weighted average accuracy
        avg_val_acc = source_weight * source_val_acc + target_weight * target_val_acc

        # Record history
        history['epochs'].append(epoch)
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['source_val_losses'].append(source_val_loss)
        history['source_val_accs'].append(source_val_acc)
        history['target_val_losses'].append(target_val_loss)
        history['target_val_accs'].append(target_val_acc)
        history['avg_val_accs'].append(avg_val_acc)
        history['source_f1s'].append(source_f1)
        history['target_f1s'].append(target_f1)

        print(f"Epoch {epoch:3d} | Train: {train_acc:.3f} | "
              f"Source Val: {source_val_acc:.3f} | Target Val: {target_val_acc:.3f} | Avg: {avg_val_acc:.3f}")

        # Track best by avg accuracy (no thresholds)
        improved = False
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch
            best_metrics = {
                'iteration': iteration,
                'epoch': best_epoch,
                'avg_val_acc': avg_val_acc,
                'source_val_acc': source_val_acc,
                'source_val_loss': source_val_loss,
                'source_control_acc': source_control_acc,
                'source_dementia_acc': source_dementia_acc,
                'source_f1': source_f1,
                'target_val_acc': target_val_acc,
                'target_val_loss': target_val_loss,
                'target_control_acc': target_control_acc,
                'target_dementia_acc': target_dementia_acc,
                'target_f1': target_f1
            }
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = True

        # Early stopping based on fallback improvement (not threshold-based)
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    if best_metrics:
        print(f"\nIteration {iteration} Best Results (Epoch {best_epoch}):")
        print(f"  Avg Acc: {best_metrics['avg_val_acc']*100:.2f}%")
        print(f"  Source Acc: {best_metrics['source_val_acc']*100:.2f}% | F1: {best_metrics['source_f1']:.4f}")
        print(f"  Target Acc: {best_metrics['target_val_acc']*100:.2f}% | F1: {best_metrics['target_f1']:.4f}")

    # Load best weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_metrics, history


def train_st(seed, source_train_dataset, target_train_dataset, source_val_loader, target_val_loader,
            output_dir, device, max_iterations=ST_MAX_ITERATIONS,
            confidence_threshold=ST_CONFIDENCE_THRESHOLD,
            min_pseudo_samples=ST_MIN_PSEUDO_SAMPLES,
            manual_weight_control=None, manual_weight_dementia=None,
            warmup_epochs=None, max_epochs=MAX_EPOCHS,
            source_weight=0.5, target_weight=0.5):
    """
    Complete Self-Training pipeline

    Algorithm:
    - Iteration 0: Train on source only (warmup phase) → Generate initial pseudo-labels
    - Iteration 1-N: Train on source + pseudo → Regenerate pseudo-labels → Repeat
    - Stop when: max iterations reached OR no new pseudo-labels OR performance plateaus

    Args:
        seed: Random seed
        source_train_dataset: FeatureDataset for source training
        target_train_dataset: FeatureDataset for target training (unlabeled)
        source_val_loader: Source validation loader
        target_val_loader: Target validation loader
        output_dir: Directory to save models
        device: Device
        max_iterations: Maximum ST iterations
        confidence_threshold: Pseudo-label confidence threshold
        min_pseudo_samples: Minimum pseudo-labeled samples to continue
        manual_weight_control: Manual weight for Control class (None = auto-compute)
        manual_weight_dementia: Manual weight for Dementia class (None = auto-compute)
        warmup_epochs: Number of epochs for iteration 0 warmup (None = use max_epochs)
        max_epochs: Number of epochs for iteration 1+ (default: from config.MAX_EPOCHS)
        source_weight: Weight for source domain in avg accuracy (default: 0.5)
        target_weight: Weight for target domain in avg accuracy (default: 0.5)

    Returns:
        seed: Random seed used
        all_metrics: List of metrics for each iteration
        all_histories: List of training histories for each iteration
        all_pseudo_stats: List of pseudo-label statistics for each iteration
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_histories = []
    all_pseudo_stats = []

    # Track best model across all iterations
    best_overall_avg_acc = float('-inf')
    best_overall_iteration = -1

    print(f"\n{'='*70}")
    print(f"Self-Training: Seed {seed}")
    print(f"Model Selection: Avg Acc (Source {source_weight:.1f}, Target {target_weight:.1f})")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*70}")

    # Initialize model for iteration 0
    model = AD_XLSR_Model(dropout=XLSR_DROPOUT).to(device)

    for iteration in range(max_iterations + 1):  # 0 to max_iterations inclusive

        # ===== Determine pseudo-label dataset for this iteration =====
        if iteration == 0:
            # Iteration 0: Train on source only (no pseudo-labels yet)
            pseudo_dataset = PseudoLabelDataset([], [], [], [])
            print(f"\nIteration 0: Initial training on source domain only")
        else:
            # Iteration 1+: Use pseudo-labels from previous iteration's model
            print(f"\nIteration {iteration}: Generating pseudo-labels...")
            selected_features, selected_labels, selected_ids, selected_confs, pseudo_stats = generate_pseudo_labels(
                model, target_train_dataset, device, confidence_threshold
            )

            all_pseudo_stats.append(pseudo_stats)

            # Check stopping criteria
            if len(selected_labels) < min_pseudo_samples:
                print(f"\n⚠️  Stopping: Only {len(selected_labels)} pseudo-labels (< {min_pseudo_samples})")
                break

            # Create pseudo-label dataset
            pseudo_dataset = PseudoLabelDataset(selected_features, selected_labels, selected_ids, selected_confs)

        # ===== Train model for this iteration =====
        # Note: For iteration 1+, we reinitialize the model (complete retraining)
        if iteration > 0:
            model = AD_XLSR_Model(dropout=XLSR_DROPOUT).to(device)

        # Use warmup_epochs for iteration 0, max_epochs for subsequent iterations
        epochs_for_iteration = warmup_epochs if (iteration == 0 and warmup_epochs is not None) else max_epochs

        trained_model, metrics, history = train_st_iteration(
            iteration=iteration,
            model=model,
            source_dataset=source_train_dataset,
            pseudo_dataset=pseudo_dataset,
            source_val_loader=source_val_loader,
            target_val_loader=target_val_loader,
            device=device,
            seed=seed,  # Use same seed for reproducibility
            max_epochs=epochs_for_iteration,
            manual_weight_control=manual_weight_control,
            manual_weight_dementia=manual_weight_dementia,
            source_weight=source_weight,
            target_weight=target_weight
        )

        all_metrics.append(metrics)
        all_histories.append(history)

        # Save model for this iteration
        torch.save(trained_model.state_dict(), seed_dir / f'st_iter_{iteration}.pth')

        # Track best model across iterations (no thresholds)
        if metrics:
            current_avg_acc = metrics.get('avg_val_acc', 0)
            if current_avg_acc > best_overall_avg_acc:
                best_overall_avg_acc = current_avg_acc
                best_overall_iteration = iteration
                torch.save(trained_model.state_dict(), seed_dir / 'best_st.pth')

        # Update model for next iteration (for pseudo-label generation)
        model = trained_model

        # Check for convergence
        if iteration > 0 and metrics and all_metrics[iteration-1]:
            prev_target_acc = all_metrics[iteration-1].get('target_val_acc', 0)
            curr_target_acc = metrics.get('target_val_acc', 0)
            improvement = curr_target_acc - prev_target_acc

    # Print final summary
    print(f"\n{'='*70}")
    print(f"Self-Training Summary (Seed {seed})")
    print(f"{'='*70}")
    if best_overall_iteration >= 0:
        print(f"Best iteration: {best_overall_iteration}")
        print(f"Best avg accuracy: {best_overall_avg_acc*100:.2f}%")
    else:
        print(f"⚠️  No valid iterations found")
    print(f"\nPer-Iteration Results:")
    print(f"{'Iter':<6} {'Avg Acc':<12} {'Source Acc':<12} {'Target Acc':<12} {'Source F1':<12} {'Target F1':<12} {'Pseudo':<10}")
    print(f"{'-'*80}")
    for i, m in enumerate(all_metrics):
        if m:
            pseudo_count = all_pseudo_stats[i-1]['selected_count'] if i > 0 and i <= len(all_pseudo_stats) else 0
            avg_acc = m.get('avg_val_acc', 0)
            print(f"{i:<6} {avg_acc*100:>10.2f}% {m['source_val_acc']*100:>10.2f}% {m['target_val_acc']*100:>10.2f}% "
                  f"{m['source_f1']:>10.4f} {m['target_f1']:>10.4f} {pseudo_count:>8}")

    return seed, all_metrics, all_histories, all_pseudo_stats

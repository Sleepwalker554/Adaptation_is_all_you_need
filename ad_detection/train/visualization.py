"""
Visualization utilities for training metrics.

Simple and reusable plotting functions for training visualization.
"""

import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


def plot_training_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    train_color: str = '#27F5EE',
    val_color: str = '#B727F5',
    title_prefix: Optional[str] = None,
    save_path: Optional[Path] = None
):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        epochs: epoch list
        train_loss, val_loss: Training/validation loss
        train_acc, val_acc: Training/validation accuracy (0-1)
        train_color: Training curve color (default cyan)
        val_color: Validation curve color (default purple)
        title_prefix: Title prefix (e.g. "Seed 42")
        save_path: Save path (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    title_base = f"{title_prefix}: " if title_prefix else ""
    
    # Loss curve
    ax1.plot(epochs, train_loss, 'o-', label='Train', color=train_color, linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 's-', label='Val', color=val_color, linewidth=2, markersize=4)
    ax1.set_title(f'{title_base}Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    train_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in train_acc]
    val_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in val_acc]
    
    ax2.plot(epochs, train_acc_pct, 'o-', label='Train', color=train_color, linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc_pct, 's-', label='Val', color=val_color, linewidth=2, markersize=4)
    ax2.set_title(f'{title_base}Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_seeds_comparison(
    seeds: List[int],
    accuracies: List[float],
    bar_color: str = '#27F5EE',
    mean_color: str = '#B727F5',
    save_path: Optional[Path] = None
):
    """
    Args:
        seeds: seed list
        accuracies: Corresponding accuracy
        bar_color: Bar chart color
        mean_color: Mean line color
        save_path: Save path (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracies]
    bars = ax.bar(range(len(seeds)), acc_pct, color=bar_color, alpha=0.7, edgecolor='black')
    mean_acc = sum(acc_pct) / len(acc_pct)
    ax.axhline(y=mean_acc, color=mean_color, linestyle='--', 
               linewidth=2, label=f'Mean: {mean_acc:.2f}%')
    
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Across Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Label the values on the bars
    for bar, value in zip(bars, acc_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()




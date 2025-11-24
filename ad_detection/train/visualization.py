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
    绘制训练和验证的损失和准确率曲线
    
    Args:
        epochs: epoch列表
        train_loss, val_loss: 训练/验证损失
        train_acc, val_acc: 训练/验证准确率 (0-1之间)
        train_color: 训练曲线颜色 (默认青色)
        val_color: 验证曲线颜色 (默认紫色)
        title_prefix: 标题前缀 (如 "Seed 42")
        save_path: 保存路径 (可选)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    title_base = f"{title_prefix}: " if title_prefix else ""
    
    # 损失曲线
    ax1.plot(epochs, train_loss, 'o-', label='Train', color=train_color, linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 's-', label='Val', color=val_color, linewidth=2, markersize=4)
    ax1.set_title(f'{title_base}Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线 (转为百分比)
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
    对比不同seed的验证准确率
    
    Args:
        seeds: seed列表
        accuracies: 对应的准确率 (0-1之间)
        bar_color: 柱状图颜色
        mean_color: 平均线颜色
        save_path: 保存路径 (可选)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 转换为百分比
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracies]
    
    # 柱状图
    bars = ax.bar(range(len(seeds)), acc_pct, color=bar_color, alpha=0.7, edgecolor='black')
    
    # 平均线
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
    
    # 在柱子上标注数值
    for bar, value in zip(bars, acc_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()




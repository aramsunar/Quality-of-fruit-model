"""
Multi-task evaluation metrics and visualization utilities.

This module provides evaluation functions specifically designed for multi-task
learning with dual prediction heads (quality and fruit type).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
import pandas as pd


def evaluate_multitask_model(model, data_loader, device, num_quality_classes, num_fruit_classes):
    """
    Evaluate multi-task model and return predictions and labels for both tasks.

    Args:
        model (nn.Module): Multi-task PyTorch model
        data_loader (DataLoader): Data loader
        device (torch.device): Device to use
        num_quality_classes (int): Number of quality classes
        num_fruit_classes (int): Number of fruit type classes

    Returns:
        tuple: (quality_preds, quality_labels, quality_probs,
                fruit_preds, fruit_labels, fruit_probs)
    """
    model.eval()

    all_quality_preds = []
    all_quality_labels = []
    all_quality_probs = []

    all_fruit_preds = []
    all_fruit_labels = []
    all_fruit_probs = []

    with torch.no_grad():
        for images, quality_labels, fruit_labels in data_loader:
            images = images.to(device)

            # Forward pass
            quality_logits, fruit_logits = model(images)

            # Get probabilities
            quality_probs = torch.softmax(quality_logits, dim=1)
            fruit_probs = torch.softmax(fruit_logits, dim=1)

            all_quality_probs.append(quality_probs.cpu().numpy())
            all_fruit_probs.append(fruit_probs.cpu().numpy())

            # Get predictions
            _, quality_preds = quality_logits.max(1)
            _, fruit_preds = fruit_logits.max(1)

            all_quality_preds.append(quality_preds.cpu().numpy())
            all_quality_labels.append(quality_labels.numpy())

            all_fruit_preds.append(fruit_preds.cpu().numpy())
            all_fruit_labels.append(fruit_labels.numpy())

    # Concatenate all batches
    all_quality_preds = np.concatenate(all_quality_preds)
    all_quality_labels = np.concatenate(all_quality_labels)
    all_quality_probs = np.concatenate(all_quality_probs)

    all_fruit_preds = np.concatenate(all_fruit_preds)
    all_fruit_labels = np.concatenate(all_fruit_labels)
    all_fruit_probs = np.concatenate(all_fruit_probs)

    return (all_quality_preds, all_quality_labels, all_quality_probs,
            all_fruit_preds, all_fruit_labels, all_fruit_probs)


def calculate_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    """
    Calculate evaluation metrics (same as single-task).

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_prob (array): Prediction probabilities
        num_classes (int): Number of classes

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class

    # Calculate AUC if probabilities are provided
    if y_prob is not None and num_classes is not None:
        try:
            if num_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                metrics['auc'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
        except:
            metrics['auc'] = None

    return metrics


def print_multitask_metrics(quality_metrics, fruit_metrics, quality_classes, fruit_classes):
    """
    Print evaluation metrics for both tasks in a formatted way.

    Args:
        quality_metrics (dict): Quality classification metrics
        fruit_metrics (dict): Fruit type classification metrics
        quality_classes (list): List of quality class names
        fruit_classes (list): List of fruit type class names
    """
    print("\n" + "=" * 80)
    print("MULTI-TASK EVALUATION METRICS".center(80))
    print("=" * 80)

    # Quality Classification Metrics
    print("\n" + "-" * 80)
    print("QUALITY CLASSIFICATION".center(80))
    print("-" * 80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {quality_metrics['accuracy']:.4f}")
    print(f"  Precision: {quality_metrics['precision']:.4f}")
    print(f"  Recall:    {quality_metrics['recall']:.4f}")
    print(f"  F1-Score:  {quality_metrics['f1_score']:.4f}")
    if 'auc' in quality_metrics and quality_metrics['auc'] is not None:
        print(f"  AUC:       {quality_metrics['auc']:.4f}")

    if 'precision_per_class' in quality_metrics and quality_classes is not None:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        for i, class_name in enumerate(quality_classes):
            print(f"{class_name:<20} {quality_metrics['precision_per_class'][i]:<12.4f} "
                  f"{quality_metrics['recall_per_class'][i]:<12.4f} {quality_metrics['f1_per_class'][i]:<12.4f}")

    # Fruit Type Classification Metrics
    print("\n" + "-" * 80)
    print("FRUIT TYPE CLASSIFICATION".center(80))
    print("-" * 80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {fruit_metrics['accuracy']:.4f}")
    print(f"  Precision: {fruit_metrics['precision']:.4f}")
    print(f"  Recall:    {fruit_metrics['recall']:.4f}")
    print(f"  F1-Score:  {fruit_metrics['f1_score']:.4f}")
    if 'auc' in fruit_metrics and fruit_metrics['auc'] is not None:
        print(f"  AUC:       {fruit_metrics['auc']:.4f}")

    if 'precision_per_class' in fruit_metrics and fruit_classes is not None:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        for i, class_name in enumerate(fruit_classes):
            print(f"{class_name:<20} {fruit_metrics['precision_per_class'][i]:<12.4f} "
                  f"{fruit_metrics['recall_per_class'][i]:<12.4f} {fruit_metrics['f1_per_class'][i]:<12.4f}")

    print("\n" + "=" * 80)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title_prefix=""):
    """
    Plot confusion matrix.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the figure
        title_prefix (str): Prefix for plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{title_prefix}Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'{title_prefix}Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, num_classes, save_path=None, title_prefix=""):
    """
    Plot ROC curves for multi-class classification.

    Args:
        y_true (array): True labels
        y_prob (array): Prediction probabilities
        class_names (list): List of class names
        num_classes (int): Number of classes
        save_path (str): Path to save the figure
        title_prefix (str): Prefix for plot title
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    colors = plt.cm.get_cmap('tab10', num_classes)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title_prefix}ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")

    plt.close()


def plot_multitask_training_history(history, save_path=None):
    """
    Plot multi-task training history.

    Args:
        history (dict): Training history with both task metrics
        save_path (str): Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Quality metrics
    # Loss
    axes[0, 0].plot(history['train_quality_loss'], label='Train Quality Loss', linewidth=2)
    axes[0, 0].plot(history['val_quality_loss'], label='Val Quality Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Quality Classification Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_quality_acc'], label='Train Quality Acc', linewidth=2)
    axes[0, 1].plot(history['val_quality_acc'], label='Val Quality Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Quality Classification Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # Combined total loss
    axes[0, 2].plot(history['train_total_loss'], label='Train Total Loss', linewidth=2)
    axes[0, 2].plot(history['val_total_loss'], label='Val Total Loss', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('Loss', fontsize=11)
    axes[0, 2].set_title('Combined Total Loss', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(alpha=0.3)

    # Row 2: Fruit type metrics
    # Loss
    axes[1, 0].plot(history['train_fruit_loss'], label='Train Fruit Loss', linewidth=2)
    axes[1, 0].plot(history['val_fruit_loss'], label='Val Fruit Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].set_title('Fruit Type Classification Loss', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # Accuracy
    axes[1, 1].plot(history['train_fruit_acc'], label='Train Fruit Acc', linewidth=2)
    axes[1, 1].plot(history['val_fruit_acc'], label='Val Fruit Acc', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].set_title('Fruit Type Classification Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    # Learning rate
    axes[1, 2].plot(history['learning_rates'], linewidth=2, color='green')
    axes[1, 2].set_xlabel('Epoch', fontsize=11)
    axes[1, 2].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.close()


def save_multitask_metrics_to_csv(quality_metrics, fruit_metrics,
                                   quality_classes, fruit_classes, save_path):
    """
    Save multi-task metrics to CSV file.

    Args:
        quality_metrics (dict): Quality classification metrics
        fruit_metrics (dict): Fruit type classification metrics
        quality_classes (list): List of quality class names
        fruit_classes (list): List of fruit type class names
        save_path (str): Path to save the CSV file
    """
    # Quality overall metrics
    quality_overall_df = pd.DataFrame({
        'Task': ['Quality'] * 5,
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [
            quality_metrics['accuracy'],
            quality_metrics['precision'],
            quality_metrics['recall'],
            quality_metrics['f1_score'],
            quality_metrics.get('auc', 'N/A')
        ]
    })

    # Fruit overall metrics
    fruit_overall_df = pd.DataFrame({
        'Task': ['Fruit Type'] * 5,
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [
            fruit_metrics['accuracy'],
            fruit_metrics['precision'],
            fruit_metrics['recall'],
            fruit_metrics['f1_score'],
            fruit_metrics.get('auc', 'N/A')
        ]
    })

    # Combined overall metrics
    overall_df = pd.concat([quality_overall_df, fruit_overall_df], ignore_index=True)

    # Quality per-class metrics
    quality_per_class_df = pd.DataFrame({
        'Task': ['Quality'] * len(quality_classes),
        'Class': quality_classes,
        'Precision': quality_metrics['precision_per_class'],
        'Recall': quality_metrics['recall_per_class'],
        'F1-Score': quality_metrics['f1_per_class']
    })

    # Fruit per-class metrics
    fruit_per_class_df = pd.DataFrame({
        'Task': ['Fruit Type'] * len(fruit_classes),
        'Class': fruit_classes,
        'Precision': fruit_metrics['precision_per_class'],
        'Recall': fruit_metrics['recall_per_class'],
        'F1-Score': fruit_metrics['f1_per_class']
    })

    # Combined per-class metrics
    per_class_df = pd.concat([quality_per_class_df, fruit_per_class_df], ignore_index=True)

    # Save to CSV
    with open(save_path, 'w') as f:
        f.write("Overall Metrics\n")
        overall_df.to_csv(f, index=False)
        f.write("\n\nPer-Class Metrics\n")
        per_class_df.to_csv(f, index=False)

    print(f"Multi-task metrics saved to {save_path}")


def evaluate_and_save_multitask_results(model, data_loader, device,
                                         num_quality_classes, num_fruit_classes,
                                         quality_classes, fruit_classes,
                                         paths, split_name='val'):
    """
    Evaluate multi-task model and save all results.

    Args:
        model (nn.Module): Multi-task PyTorch model
        data_loader (DataLoader): Data loader to evaluate on
        device (torch.device): Device to use
        num_quality_classes (int): Number of quality classes
        num_fruit_classes (int): Number of fruit type classes
        quality_classes (list): List of quality class names
        fruit_classes (list): List of fruit type class names
        paths (dict): Dictionary with 'figures' and 'reports' paths
        split_name (str): Name of split ('val' or 'test')

    Returns:
        tuple: (quality_metrics, fruit_metrics)
    """
    # Evaluate model
    (quality_preds, quality_labels, quality_probs,
     fruit_preds, fruit_labels, fruit_probs) = evaluate_multitask_model(
        model, data_loader, device, num_quality_classes, num_fruit_classes
    )

    # Calculate metrics for both tasks
    quality_metrics = calculate_metrics(quality_labels, quality_preds, quality_probs, num_quality_classes)
    fruit_metrics = calculate_metrics(fruit_labels, fruit_preds, fruit_probs, num_fruit_classes)

    # Print metrics
    print_multitask_metrics(quality_metrics, fruit_metrics, quality_classes, fruit_classes)

    # Plot confusion matrices
    plot_confusion_matrix(
        y_true=quality_labels,
        y_pred=quality_preds,
        class_names=quality_classes,
        save_path=paths['figures'] / f'confusion_matrix_quality_{split_name}.png',
        title_prefix='Quality - '
    )

    plot_confusion_matrix(
        y_true=fruit_labels,
        y_pred=fruit_preds,
        class_names=fruit_classes,
        save_path=paths['figures'] / f'confusion_matrix_fruit_{split_name}.png',
        title_prefix='Fruit Type - '
    )

    # Plot ROC curves
    if num_quality_classes > 1:
        plot_roc_curves(
            y_true=quality_labels,
            y_prob=quality_probs,
            class_names=quality_classes,
            num_classes=num_quality_classes,
            save_path=paths['figures'] / f'roc_curves_quality_{split_name}.png',
            title_prefix='Quality - '
        )

    if num_fruit_classes > 1:
        plot_roc_curves(
            y_true=fruit_labels,
            y_prob=fruit_probs,
            class_names=fruit_classes,
            num_classes=num_fruit_classes,
            save_path=paths['figures'] / f'roc_curves_fruit_{split_name}.png',
            title_prefix='Fruit Type - '
        )

    # Save metrics to CSV
    save_multitask_metrics_to_csv(
        quality_metrics=quality_metrics,
        fruit_metrics=fruit_metrics,
        quality_classes=quality_classes,
        fruit_classes=fruit_classes,
        save_path=paths['reports'] / f'metrics_{split_name}.csv'
    )

    return quality_metrics, fruit_metrics

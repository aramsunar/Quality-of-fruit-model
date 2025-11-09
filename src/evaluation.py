"""
Evaluation metrics and visualization utilities.
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


def evaluate_model(model, data_loader, device, num_classes):
    """
    Evaluate model and return predictions and labels.

    Args:
        model (nn.Module): PyTorch model
        data_loader (DataLoader): Data loader
        device (torch.device): Device to use
        num_classes (int): Number of classes

    Returns:
        tuple: (predictions, labels, probabilities)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            # Get predictions
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return all_preds, all_labels, all_probs


def calculate_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    """
    Calculate evaluation metrics.

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


def print_metrics(metrics, class_names=None):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of metrics
        class_names (list): List of class names
    """
    print("\n" + "=" * 70)
    print("EVALUATION METRICS".center(70))
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"  AUC:       {metrics['auc']:.4f}")

    if 'precision_per_class' in metrics and class_names is not None:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<20} {metrics['precision_per_class'][i]:<12.4f} "
                  f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}")

    print("=" * 70)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, num_classes, save_path=None):
    """
    Plot ROC curves for multi-class classification.

    Args:
        y_true (array): True labels
        y_prob (array): Prediction probabilities
        class_names (list): List of class names
        num_classes (int): Number of classes
        save_path (str): Path to save the figure
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
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")

    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history.

    Args:
        history (dict): Training history
        save_path (str): Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    # Plot learning rate
    axes[2].plot(history['learning_rates'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.close()


def save_metrics_to_csv(metrics, class_names, save_path):
    """
    Save metrics to CSV file.

    Args:
        metrics (dict): Dictionary of metrics
        class_names (list): List of class names
        save_path (str): Path to save the CSV file
    """
    # Overall metrics
    overall_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics.get('auc', 'N/A')
        ]
    })

    # Per-class metrics
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1-Score': metrics['f1_per_class']
    })

    # Save to CSV
    with open(save_path, 'w') as f:
        f.write("Overall Metrics\n")
        overall_df.to_csv(f, index=False)
        f.write("\n\nPer-Class Metrics\n")
        per_class_df.to_csv(f, index=False)

    print(f"Metrics saved to {save_path}")


def evaluate_and_save_results(model, data_loader, device, num_classes, class_names,
                               paths, split_name='val'):
    """
    Evaluate model and save all results (metrics, plots, CSV).

    Combines the common evaluation workflow:
    - Evaluate model
    - Calculate metrics
    - Print metrics
    - Plot confusion matrix
    - Plot ROC curves (if multi-class)
    - Save metrics to CSV

    Args:
        model (nn.Module): PyTorch model
        data_loader (DataLoader): Data loader to evaluate on
        device (torch.device): Device to use
        num_classes (int): Number of classes
        class_names (list): List of class names
        paths (dict): Dictionary with 'figures' and 'reports' paths
        split_name (str): Name of split ('val' or 'test')

    Returns:
        dict: Calculated metrics
    """
    # Evaluate model
    preds, labels, probs = evaluate_model(model, data_loader, device, num_classes)

    # Calculate metrics
    metrics = calculate_metrics(labels, preds, probs, num_classes)

    # Print metrics
    print_metrics(metrics, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=labels,
        y_pred=preds,
        class_names=class_names,
        save_path=paths['figures'] / f'confusion_matrix_{split_name}.png'
    )

    # Plot ROC curves (if multi-class)
    if num_classes > 1:
        plot_roc_curves(
            y_true=labels,
            y_prob=probs,
            class_names=class_names,
            num_classes=num_classes,
            save_path=paths['figures'] / f'roc_curves_{split_name}.png'
        )

    # Save metrics to CSV
    save_metrics_to_csv(
        metrics=metrics,
        class_names=class_names,
        save_path=paths['reports'] / f'metrics_{split_name}.csv'
    )

    return metrics

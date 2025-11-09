"""
Multi-task training utilities for fruit type and quality classification.

This module provides training utilities specifically designed for multi-task
learning with dual prediction heads.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
from pathlib import Path
from utils import AverageMeter, format_time
from collections import Counter


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function combining quality and fruit type classification losses.

    This loss function computes separate cross-entropy losses for both tasks
    and combines them with configurable weights.
    """

    def __init__(self, quality_weight=1.0, fruit_weight=1.0,
                 quality_class_weights=None, fruit_class_weights=None):
        """
        Args:
            quality_weight (float): Weight for quality classification loss
            fruit_weight (float): Weight for fruit type classification loss
            quality_class_weights (Tensor): Class weights for quality classification
            fruit_class_weights (Tensor): Class weights for fruit type classification
        """
        super(MultiTaskLoss, self).__init__()

        self.quality_weight = quality_weight
        self.fruit_weight = fruit_weight

        self.quality_criterion = nn.CrossEntropyLoss(weight=quality_class_weights)
        self.fruit_criterion = nn.CrossEntropyLoss(weight=fruit_class_weights)

    def forward(self, quality_logits, fruit_logits, quality_labels, fruit_labels):
        """
        Compute multi-task loss.

        Args:
            quality_logits: Quality classification logits [batch_size, num_quality_classes]
            fruit_logits: Fruit type classification logits [batch_size, num_fruit_classes]
            quality_labels: Quality labels [batch_size]
            fruit_labels: Fruit type labels [batch_size]

        Returns:
            tuple: (total_loss, quality_loss, fruit_loss)
        """
        quality_loss = self.quality_criterion(quality_logits, quality_labels)
        fruit_loss = self.fruit_criterion(fruit_logits, fruit_labels)

        total_loss = (self.quality_weight * quality_loss +
                     self.fruit_weight * fruit_loss)

        return total_loss, quality_loss, fruit_loss


class MultiTaskTrainer:
    """Training class for multi-task learning."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, save_dir, num_epochs=50, patience=10,
                 warmup_epochs=0, grad_clip=None, mixed_precision=False):
        """
        Args:
            model (nn.Module): Multi-task PyTorch model
            train_loader (DataLoader): Training data loader (returns images, quality_labels, fruit_labels)
            val_loader (DataLoader): Validation data loader
            criterion (MultiTaskLoss): Multi-task loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device (torch.device): Device to use
            save_dir (str): Directory to save checkpoints
            num_epochs (int): Number of epochs to train
            patience (int): Early stopping patience
            warmup_epochs (int): Number of warmup epochs
            grad_clip (float): Gradient clipping value
            mixed_precision (bool): Use mixed precision training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision

        # Track best metrics for both tasks
        self.best_combined_acc = 0.0  # Average of both tasks
        self.best_quality_acc = 0.0
        self.best_fruit_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Training history
        self.history = {
            'train_total_loss': [],
            'train_quality_loss': [],
            'train_fruit_loss': [],
            'train_quality_acc': [],
            'train_fruit_acc': [],
            'val_total_loss': [],
            'val_quality_loss': [],
            'val_fruit_loss': [],
            'val_quality_acc': [],
            'val_fruit_acc': [],
            'learning_rates': []
        }

        # Store initial learning rate for warmup
        self.initial_lr = optimizer.param_groups[0]['lr']

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if mixed_precision and device.type == 'cuda' else None

    def _adjust_learning_rate_warmup(self, epoch, batch_idx, num_batches):
        """Adjust learning rate during warmup phase."""
        if epoch > self.warmup_epochs:
            return

        # Linear warmup
        total_warmup_steps = self.warmup_epochs * num_batches
        current_step = (epoch - 1) * num_batches + batch_idx
        warmup_lr = self.initial_lr * (current_step / total_warmup_steps)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        total_losses = AverageMeter()
        quality_losses = AverageMeter()
        fruit_losses = AverageMeter()
        quality_accs = AverageMeter()
        fruit_accs = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        num_batches = len(self.train_loader)

        for batch_idx, (images, quality_labels, fruit_labels) in enumerate(pbar):
            # Apply learning rate warmup
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                self._adjust_learning_rate_warmup(epoch, batch_idx, num_batches)

            images = images.to(self.device)
            quality_labels = quality_labels.to(self.device)
            fruit_labels = fruit_labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    quality_logits, fruit_logits = self.model(images)
                    total_loss, quality_loss, fruit_loss = self.criterion(
                        quality_logits, fruit_logits,
                        quality_labels, fruit_labels
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                quality_logits, fruit_logits = self.model(images)
                total_loss, quality_loss, fruit_loss = self.criterion(
                    quality_logits, fruit_logits,
                    quality_labels, fruit_labels
                )

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Calculate accuracies for both tasks
            _, quality_pred = quality_logits.max(1)
            _, fruit_pred = fruit_logits.max(1)

            quality_correct = quality_pred.eq(quality_labels).sum().item()
            fruit_correct = fruit_pred.eq(fruit_labels).sum().item()

            quality_acc = quality_correct / quality_labels.size(0)
            fruit_acc = fruit_correct / fruit_labels.size(0)

            # Update meters
            batch_size = quality_labels.size(0)
            total_losses.update(total_loss.item(), batch_size)
            quality_losses.update(quality_loss.item(), batch_size)
            fruit_losses.update(fruit_loss.item(), batch_size)
            quality_accs.update(quality_acc, batch_size)
            fruit_accs.update(fruit_acc, batch_size)

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_losses.avg:.4f}',
                'q_acc': f'{quality_accs.avg:.4f}',
                'f_acc': f'{fruit_accs.avg:.4f}',
                'lr': f'{current_lr:.6f}'
            })

        return (total_losses.avg, quality_losses.avg, fruit_losses.avg,
                quality_accs.avg, fruit_accs.avg)

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()

        total_losses = AverageMeter()
        quality_losses = AverageMeter()
        fruit_losses = AverageMeter()
        quality_accs = AverageMeter()
        fruit_accs = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]')

        with torch.no_grad():
            for images, quality_labels, fruit_labels in pbar:
                images = images.to(self.device)
                quality_labels = quality_labels.to(self.device)
                fruit_labels = fruit_labels.to(self.device)

                # Forward pass
                quality_logits, fruit_logits = self.model(images)
                total_loss, quality_loss, fruit_loss = self.criterion(
                    quality_logits, fruit_logits,
                    quality_labels, fruit_labels
                )

                # Calculate accuracies
                _, quality_pred = quality_logits.max(1)
                _, fruit_pred = fruit_logits.max(1)

                quality_correct = quality_pred.eq(quality_labels).sum().item()
                fruit_correct = fruit_pred.eq(fruit_labels).sum().item()

                quality_acc = quality_correct / quality_labels.size(0)
                fruit_acc = fruit_correct / fruit_labels.size(0)

                # Update meters
                batch_size = quality_labels.size(0)
                total_losses.update(total_loss.item(), batch_size)
                quality_losses.update(quality_loss.item(), batch_size)
                fruit_losses.update(fruit_loss.item(), batch_size)
                quality_accs.update(quality_acc, batch_size)
                fruit_accs.update(fruit_acc, batch_size)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_losses.avg:.4f}',
                    'q_acc': f'{quality_accs.avg:.4f}',
                    'f_acc': f'{fruit_accs.avg:.4f}'
                })

        return (total_losses.avg, quality_losses.avg, fruit_losses.avg,
                quality_accs.avg, fruit_accs.avg)

    def train(self):
        """Train the model for specified number of epochs."""
        print(f"\nMulti-Task Training on device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Patience: {self.patience}")
        if self.warmup_epochs > 0:
            print(f"Warmup epochs: {self.warmup_epochs}")
        if self.grad_clip is not None:
            print(f"Gradient clipping: {self.grad_clip}")
        if self.mixed_precision:
            print(f"Mixed precision: {'Enabled' if self.scaler is not None else 'Not available (requires CUDA)'}")
        print(f"Loss weights - Quality: {self.criterion.quality_weight}, Fruit: {self.criterion.fruit_weight}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_results = self.train_epoch(epoch)
            train_total_loss, train_quality_loss, train_fruit_loss, train_quality_acc, train_fruit_acc = train_results

            # Validate
            val_results = self.validate(epoch)
            val_total_loss, val_quality_loss, val_fruit_loss, val_quality_acc, val_fruit_acc = val_results

            # Update learning rate (skip scheduler during warmup)
            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_total_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_total_loss'].append(train_total_loss)
            self.history['train_quality_loss'].append(train_quality_loss)
            self.history['train_fruit_loss'].append(train_fruit_loss)
            self.history['train_quality_acc'].append(train_quality_acc)
            self.history['train_fruit_acc'].append(train_fruit_acc)
            self.history['val_total_loss'].append(val_total_loss)
            self.history['val_quality_loss'].append(val_quality_loss)
            self.history['val_fruit_loss'].append(val_fruit_loss)
            self.history['val_quality_acc'].append(val_quality_acc)
            self.history['val_fruit_acc'].append(val_fruit_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            print(f"  Train - Total Loss: {train_total_loss:.4f} | Quality Acc: {train_quality_acc:.4f} | Fruit Acc: {train_fruit_acc:.4f}")
            print(f"  Val   - Total Loss: {val_total_loss:.4f} | Quality Acc: {val_quality_acc:.4f} | Fruit Acc: {val_fruit_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Calculate combined accuracy (average of both tasks)
            val_combined_acc = (val_quality_acc + val_fruit_acc) / 2

            # Save best model based on combined accuracy
            if val_combined_acc > self.best_combined_acc:
                self.best_combined_acc = val_combined_acc
                self.best_quality_acc = val_quality_acc
                self.best_fruit_acc = val_fruit_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth', epoch, val_quality_acc, val_fruit_acc)
                print(f"  New best model saved! (Combined Acc: {val_combined_acc:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement ({self.epochs_without_improvement}/{self.patience})")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                break

            print("-" * 70)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(training_time)}")
        print(f"Best combined accuracy: {self.best_combined_acc:.4f} (Epoch {self.best_epoch})")
        print(f"  Quality Acc: {self.best_quality_acc:.4f}")
        print(f"  Fruit Acc: {self.best_fruit_acc:.4f}")

        # Save final model
        self.save_checkpoint('final_model.pth', epoch, val_quality_acc, val_fruit_acc)

        return self.history

    def save_checkpoint(self, filename, epoch, val_quality_acc, val_fruit_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_quality_acc': val_quality_acc,
            'val_fruit_acc': val_fruit_acc,
            'val_combined_acc': (val_quality_acc + val_fruit_acc) / 2,
            'history': self.history
        }
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint.get('val_quality_acc', 0.0), checkpoint.get('val_fruit_acc', 0.0)


def create_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4):
    """
    Create an optimizer (same as single-task).

    Args:
        model (nn.Module): PyTorch model
        optimizer_name (str): Name of optimizer ('adam', 'sgd', 'adamw')
        lr (float): Learning rate
        weight_decay (float): Weight decay

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name='plateau', **kwargs):
    """
    Create a learning rate scheduler (same as single-task).

    Args:
        optimizer: Optimizer
        scheduler_name (str): Name of scheduler ('plateau', 'cosine', 'step')
        **kwargs: Additional arguments for the scheduler

    Returns:
        Learning rate scheduler
    """
    if scheduler_name.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 50)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def compute_multitask_class_weights(train_loader, num_quality_classes, num_fruit_classes, device):
    """
    Compute class weights for both tasks in multi-task learning.

    Args:
        train_loader: Training data loader (returns images, quality_labels, fruit_labels)
        num_quality_classes: Number of quality classes
        num_fruit_classes: Number of fruit type classes
        device: Device to put weights on

    Returns:
        tuple: (quality_weights, fruit_weights)
    """
    print("Computing class weights for both tasks...")

    # Count samples per class for both tasks
    quality_counts = Counter()
    fruit_counts = Counter()

    for _, quality_labels, fruit_labels in train_loader:
        quality_counts.update(quality_labels.numpy())
        fruit_counts.update(fruit_labels.numpy())

    # Quality class weights
    quality_count_array = np.array([quality_counts[i] for i in range(num_quality_classes)])
    total_quality = quality_count_array.sum()
    quality_weights = total_quality / (num_quality_classes * quality_count_array)
    quality_weights = quality_weights / quality_weights.sum() * num_quality_classes
    quality_weights_tensor = torch.FloatTensor(quality_weights).to(device)

    # Fruit class weights
    fruit_count_array = np.array([fruit_counts[i] for i in range(num_fruit_classes)])
    total_fruit = fruit_count_array.sum()
    fruit_weights = total_fruit / (num_fruit_classes * fruit_count_array)
    fruit_weights = fruit_weights / fruit_weights.sum() * num_fruit_classes
    fruit_weights_tensor = torch.FloatTensor(fruit_weights).to(device)

    print(f"\nQuality class counts: {quality_count_array}")
    print(f"Quality class weights: {quality_weights}")
    print(f"\nFruit class counts: {fruit_count_array}")
    print(f"Fruit class weights: {fruit_weights}")

    return quality_weights_tensor, fruit_weights_tensor

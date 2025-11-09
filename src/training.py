"""
Training utilities and loops.
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


class Trainer:
    """Training class for managing the training process."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, save_dir, num_epochs=50, patience=10,
                 warmup_epochs=0, grad_clip=None, mixed_precision=False):
        """
        Args:
            model (nn.Module): PyTorch model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device (torch.device): Device to use
            save_dir (str): Directory to save checkpoints
            num_epochs (int): Number of epochs to train
            patience (int): Early stopping patience
            warmup_epochs (int): Number of warmup epochs (default: 0)
            grad_clip (float): Gradient clipping value (default: None)
            mixed_precision (bool): Use mixed precision training (default: False)
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

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
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

        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        num_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(pbar):
            # Apply learning rate warmup
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                self._adjust_learning_rate_warmup(epoch, batch_idx, num_batches)

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            acc = correct / labels.size(0)

            # Update meters
            losses.update(loss.item(), labels.size(0))
            accs.update(acc, labels.size(0))

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.4f}',
                'lr': f'{current_lr:.6f}'
            })

        return losses.avg, accs.avg

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()

        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]')

        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                acc = correct / labels.size(0)

                # Update meters
                losses.update(loss.item(), labels.size(0))
                accs.update(acc, labels.size(0))

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc': f'{accs.avg:.4f}'
                })

        return losses.avg, accs.avg

    def train(self):
        """Train the model for specified number of epochs."""
        print(f"\nTraining on device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Patience: {self.patience}")
        if self.warmup_epochs > 0:
            print(f"Warmup epochs: {self.warmup_epochs}")
        if self.grad_clip is not None:
            print(f"Gradient clipping: {self.grad_clip}")
        if self.mixed_precision:
            print(f"Mixed precision: {'Enabled' if self.scaler is not None else 'Not available (requires CUDA)'}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Update learning rate (skip scheduler during warmup)
            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
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
        print(f"Best validation accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")

        # Save final model
        self.save_checkpoint('final_model.pth', epoch, val_acc)

        return self.history

    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
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
        return checkpoint['epoch'], checkpoint['val_acc']


def create_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4):
    """
    Create an optimizer.

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
    Create a learning rate scheduler.

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


def compute_class_weights(train_loader, num_classes, device):
    """
    Compute class weights for handling imbalanced datasets.

    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to put weights on

    Returns:
        torch.Tensor: Class weights
    """
    from collections import Counter

    print("Computing class weights...")

    # Count samples per class
    class_counts = Counter()
    for _, labels in train_loader:
        class_counts.update(labels.numpy())

    # Convert to array
    counts = np.array([class_counts[i] for i in range(num_classes)])
    total_samples = counts.sum()

    # Compute weights: inverse frequency
    weights = total_samples / (num_classes * counts)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    weights_tensor = torch.FloatTensor(weights).to(device)

    print(f"Class counts: {counts}")
    print(f"Class weights: {weights}")

    return weights_tensor

"""
Data loading and preprocessing utilities.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class FruitDataset(Dataset):
    """Custom dataset for fruit images."""

    def __init__(self, root_dir, transform=None, grayscale=False):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            grayscale (bool): Convert images to grayscale
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        if self.grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224, grayscale=False, augment=False):
    """
    Get data transforms for training and validation.

    Args:
        img_size (int): Size to resize images to
        grayscale (bool): Whether to use grayscale images
        augment (bool): Whether to apply data augmentation

    Returns:
        tuple: (train_transform, val_transform)
    """
    # Normalization values
    if grayscale:
        mean = [0.5]
        std = [0.5]
    else:
        # ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Training transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def create_data_loaders(data_dir, img_size=224, batch_size=32, grayscale=False,
                        augment=False, num_workers=4):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir (str): Root directory containing train/val/test folders
        img_size (int): Size to resize images to
        batch_size (int): Batch size for data loaders
        grayscale (bool): Whether to use grayscale images
        augment (bool): Whether to apply data augmentation
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    # Get transforms
    train_transform, val_transform = get_transforms(img_size, grayscale, augment)

    # Create datasets
    train_dataset = FruitDataset(train_dir, transform=train_transform, grayscale=grayscale)
    val_dataset = FruitDataset(val_dir, transform=val_transform, grayscale=grayscale)

    test_loader = None
    if test_dir.exists():
        test_dataset = FruitDataset(test_dir, transform=val_transform, grayscale=grayscale)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names


def get_data_statistics(data_loader, device):
    """
    Calculate mean and std of dataset.

    Args:
        data_loader (DataLoader): Data loader
        device (torch.device): Device to use

    Returns:
        tuple: (mean, std)
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in data_loader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std


def analyze_dataset(data_dir):
    """
    Analyze the dataset structure and provide statistics.

    Args:
        data_dir (str): Root directory containing the dataset

    Returns:
        dict: Dictionary containing dataset statistics
    """
    data_dir = Path(data_dir)
    stats = {
        'train': {},
        'val': {},
        'test': {}
    }

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        class_counts = {}

        for class_name in classes:
            class_dir = split_dir / class_name
            count = len(list(class_dir.glob('*.[jp][pn][g]')))
            class_counts[class_name] = count

        stats[split] = {
            'num_classes': len(classes),
            'classes': classes,
            'class_counts': class_counts,
            'total_images': sum(class_counts.values())
        }

    return stats

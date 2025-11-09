"""
Multi-task data loading utilities for fruit type and quality classification.

This module extends the standard data loader to provide both fruit type
and quality labels for multi-task learning.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class MultiTaskFruitDataset(Dataset):
    """Custom dataset for fruit images with dual labels (fruit type + quality)."""

    def __init__(self, root_dir, transform=None, grayscale=False):
        """
        Args:
            root_dir (str): Directory with all the images organized by quality class
            transform (callable, optional): Optional transform to be applied on a sample
            grayscale (bool): Convert images to grayscale
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale

        # Quality classes from folder structure
        self.quality_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.quality_to_idx = {cls_name: i for i, cls_name in enumerate(self.quality_classes)}

        # Extract fruit types from filenames
        self.fruit_types = self._extract_fruit_types()
        self.fruit_to_idx = {fruit: i for i, fruit in enumerate(sorted(self.fruit_types))}

        # Build samples list with dual labels
        self.samples = []
        for quality_name in self.quality_classes:
            class_dir = self.root_dir / quality_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    fruit_type = self._extract_fruit_type_from_filename(img_path.name)
                    if fruit_type in self.fruit_to_idx:  # Ensure valid fruit type
                        quality_label = self.quality_to_idx[quality_name]
                        fruit_label = self.fruit_to_idx[fruit_type]
                        self.samples.append((str(img_path), quality_label, fruit_label))

    def _extract_fruit_type_from_filename(self, filename):
        """
        Extract fruit type from filename.

        Filenames follow pattern: FruitType_ImageName.ext
        Examples: BananaDB_Image1.png, PeachQ_Image23.png

        Args:
            filename (str): Image filename

        Returns:
            str: Fruit type prefix
        """
        # Split by underscore and take first part as fruit type
        parts = filename.split('_')
        if len(parts) > 0:
            return parts[0]
        return "Unknown"

    def _extract_fruit_types(self):
        """
        Scan all images and extract unique fruit types from filenames.

        Returns:
            set: Set of unique fruit type names
        """
        fruit_types = set()
        for quality_name in self.quality_classes:
            class_dir = self.root_dir / quality_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    fruit_type = self._extract_fruit_type_from_filename(img_path.name)
                    fruit_types.add(fruit_type)

        # Remove any invalid entries
        fruit_types.discard("Unknown")
        fruit_types.discard("")

        return fruit_types

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get item with dual labels.

        Returns:
            tuple: (image, quality_label, fruit_label)
        """
        img_path, quality_label, fruit_label = self.samples[idx]

        if self.grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, quality_label, fruit_label


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


def create_multitask_data_loaders(data_dir, img_size=224, batch_size=32, grayscale=False,
                                   augment=False, num_workers=4):
    """
    Create multi-task data loaders for training, validation, and testing.

    Args:
        data_dir (str): Root directory containing train/val/test folders
        img_size (int): Size to resize images to
        batch_size (int): Batch size for data loaders
        grayscale (bool): Whether to use grayscale images
        augment (bool): Whether to apply data augmentation
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader, quality_classes, fruit_types)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    # Get transforms
    train_transform, val_transform = get_transforms(img_size, grayscale, augment)

    # Create datasets
    train_dataset = MultiTaskFruitDataset(train_dir, transform=train_transform, grayscale=grayscale)
    val_dataset = MultiTaskFruitDataset(val_dir, transform=val_transform, grayscale=grayscale)

    test_loader = None
    if test_dir.exists():
        test_dataset = MultiTaskFruitDataset(test_dir, transform=val_transform, grayscale=grayscale)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    quality_classes = train_dataset.quality_classes
    fruit_types = sorted(train_dataset.fruit_types)

    return train_loader, val_loader, test_loader, quality_classes, fruit_types


def analyze_multitask_dataset(data_dir):
    """
    Analyze the multi-task dataset structure and provide statistics.

    Args:
        data_dir (str): Root directory containing the dataset

    Returns:
        dict: Dictionary containing dataset statistics for both tasks
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

        # Quality class statistics
        quality_classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        quality_counts = {}

        # Fruit type statistics
        fruit_type_counts = {}

        # Combined statistics
        combined_counts = {}  # (fruit_type, quality) pairs

        for quality_name in quality_classes:
            class_dir = split_dir / quality_name
            quality_count = 0

            for img_path in class_dir.glob('*.[jp][pn][g]'):
                quality_count += 1

                # Extract fruit type
                fruit_type = img_path.name.split('_')[0] if '_' in img_path.name else "Unknown"

                # Count fruit types
                fruit_type_counts[fruit_type] = fruit_type_counts.get(fruit_type, 0) + 1

                # Count combinations
                combo_key = (fruit_type, quality_name)
                combined_counts[combo_key] = combined_counts.get(combo_key, 0) + 1

            quality_counts[quality_name] = quality_count

        stats[split] = {
            'num_quality_classes': len(quality_classes),
            'quality_classes': quality_classes,
            'quality_counts': quality_counts,
            'num_fruit_types': len([f for f in fruit_type_counts.keys() if f != "Unknown"]),
            'fruit_types': sorted([f for f in fruit_type_counts.keys() if f != "Unknown"]),
            'fruit_type_counts': {k: v for k, v in fruit_type_counts.items() if k != "Unknown"},
            'combined_counts': combined_counts,
            'total_images': sum(quality_counts.values())
        }

    return stats

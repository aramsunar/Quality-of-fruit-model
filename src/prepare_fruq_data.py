"""
Data Preparation Script for FruQ-multi Dataset

This script organizes the FruQ-multi dataset into a standardized train/val/test structure
suitable for training CNN models. It handles:
- Combining all 11 fruit/vegetable types
- Normalizing inconsistent class names
- Creating stratified train/val/test splits
- Handling missing classes (e.g., StrawberryQ missing "Good")
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import argparse
from typing import Dict, List, Tuple

# For reproducibility
random.seed(42)


def normalize_class_name(class_name: str) -> str:
    """
    Normalize inconsistent class names across different fruit types.

    Args:
        class_name: Original class name (e.g., "Good", "Fresh", "mild", "Mild")

    Returns:
        Normalized class name ("Good", "Mild", or "Rotten")
    """
    class_name_lower = class_name.lower()

    if class_name_lower in ['good', 'fresh']:
        return 'Good'
    elif class_name_lower == 'mild':
        return 'Mild'
    elif class_name_lower == 'rotten':
        return 'Rotten'
    else:
        raise ValueError(f"Unknown class name: {class_name}")


def collect_images(source_dir: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Collect all images from FruQ-multi dataset and organize by class.

    Args:
        source_dir: Path to FruQ-multi directory

    Returns:
        Dictionary mapping class names to list of (image_path, fruit_type) tuples
    """
    images_by_class = defaultdict(list)

    # Iterate through all fruit/vegetable directories
    for fruit_dir in sorted(source_dir.iterdir()):
        if not fruit_dir.is_dir():
            continue

        fruit_name = fruit_dir.name
        print(f"Processing {fruit_name}...")

        # Iterate through quality classes
        for class_dir in fruit_dir.iterdir():
            if not class_dir.is_dir():
                continue

            try:
                normalized_class = normalize_class_name(class_dir.name)
            except ValueError as e:
                print(f"  Warning: {e} in {fruit_name}")
                continue

            # Collect all image files
            image_count = 0
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images_by_class[normalized_class].append((img_path, fruit_name))
                    image_count += 1

            print(f"  {class_dir.name} -> {normalized_class}: {image_count} images")

    return images_by_class


def split_data(images_by_class: Dict[str, List[Tuple[Path, str]]],
               train_ratio: float = 0.7,
               val_ratio: float = 0.2,
               test_ratio: float = 0.1) -> Dict[str, Dict[str, List[Tuple[Path, str]]]]:
    """
    Split images into train/val/test sets with stratification.

    Args:
        images_by_class: Dictionary mapping class names to image lists
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing

    Returns:
        Nested dictionary: {split: {class: [(path, fruit_type), ...]}}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }

    print("\nSplitting data...")
    for class_name, images in images_by_class.items():
        # Shuffle images
        images_copy = images.copy()
        random.shuffle(images_copy)

        total = len(images_copy)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits['train'][class_name] = images_copy[:train_end]
        splits['val'][class_name] = images_copy[train_end:val_end]
        splits['test'][class_name] = images_copy[val_end:]

        print(f"  {class_name}: {len(splits['train'][class_name])} train, "
              f"{len(splits['val'][class_name])} val, {len(splits['test'][class_name])} test")

    return splits


def copy_images(splits: Dict[str, Dict[str, List[Tuple[Path, str]]]],
                output_dir: Path,
                use_symlinks: bool = False) -> None:
    """
    Copy or symlink images to the new directory structure.

    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Destination directory
        use_symlinks: If True, create symlinks instead of copying files
    """
    print("\nCopying images to new structure...")

    total_copied = 0
    for split_name, classes in splits.items():
        for class_name, images in classes.items():
            # Create destination directory
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy or symlink images
            for img_path, fruit_name in images:
                # Create unique filename: fruit_originalname
                new_name = f"{fruit_name}_{img_path.name}"
                dest_path = dest_dir / new_name

                if use_symlinks:
                    # Create symlink (relative path)
                    if not dest_path.exists():
                        dest_path.symlink_to(img_path)
                else:
                    # Copy file
                    shutil.copy2(img_path, dest_path)

                total_copied += 1

                if total_copied % 1000 == 0:
                    print(f"  Processed {total_copied} images...")

    print(f"  Total images copied: {total_copied}")


def print_statistics(splits: Dict[str, Dict[str, List[Tuple[Path, str]]]]) -> None:
    """
    Print detailed statistics about the dataset splits.

    Args:
        splits: Dictionary with train/val/test splits
    """
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    # Overall statistics
    total_images = sum(len(images) for split in splits.values()
                      for images in split.values())
    print(f"\nTotal Images: {total_images}")
    print(f"Number of Classes: {len(splits['train'].keys())}")
    print(f"Classes: {', '.join(sorted(splits['train'].keys()))}")

    # Per-split statistics
    for split_name in ['train', 'val', 'test']:
        split_total = sum(len(images) for images in splits[split_name].values())
        percentage = (split_total / total_images) * 100
        print(f"\n{split_name.upper()} SET: {split_total} images ({percentage:.1f}%)")

        for class_name in sorted(splits[split_name].keys()):
            count = len(splits[split_name][class_name])
            class_percentage = (count / split_total) * 100
            print(f"  {class_name:10s}: {count:5d} images ({class_percentage:5.1f}%)")

    # Class distribution analysis
    print("\n" + "-"*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-"*70)

    for class_name in sorted(splits['train'].keys()):
        total_class = sum(len(splits[split][class_name]) for split in ['train', 'val', 'test'])
        percentage = (total_class / total_images) * 100
        print(f"{class_name:10s}: {total_class:5d} images ({percentage:5.1f}% of total)")

    # Check for class imbalance
    class_counts = [sum(len(splits[split][class_name]) for split in ['train', 'val', 'test'])
                   for class_name in splits['train'].keys()]

    if class_counts:
        max_count = max(class_counts)
        min_count = min(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3.0:
            print("  ⚠️  WARNING: Significant class imbalance detected!")
            print("  Consider using class weights or sampling strategies during training.")

    print("="*70 + "\n")


def save_statistics_report(splits: Dict[str, Dict[str, List[Tuple[Path, str]]]],
                           output_dir: Path) -> None:
    """
    Save dataset statistics to a text file.

    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Directory to save the report
    """
    report_path = output_dir / 'dataset_statistics.txt'

    with open(report_path, 'w') as f:
        f.write("FruQ-multi Dataset Statistics Report\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        total_images = sum(len(images) for split in splits.values()
                          for images in split.values())
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Number of Classes: {len(splits['train'].keys())}\n")
        f.write(f"Classes: {', '.join(sorted(splits['train'].keys()))}\n\n")

        # Per-split statistics
        for split_name in ['train', 'val', 'test']:
            split_total = sum(len(images) for images in splits[split_name].values())
            percentage = (split_total / total_images) * 100
            f.write(f"{split_name.upper()} SET: {split_total} images ({percentage:.1f}%)\n")

            for class_name in sorted(splits[split_name].keys()):
                count = len(splits[split_name][class_name])
                class_percentage = (count / split_total) * 100
                f.write(f"  {class_name:10s}: {count:5d} images ({class_percentage:5.1f}%)\n")
            f.write("\n")

    print(f"Statistics report saved to: {report_path}")


def main():
    """Main function to prepare FruQ-multi dataset."""
    parser = argparse.ArgumentParser(description='Prepare FruQ-multi dataset for training')
    parser.add_argument('--source', type=str,
                       default='../data/raw/FruQ-multi',
                       help='Path to FruQ-multi source directory')
    parser.add_argument('--output', type=str,
                       default='../data/processed/FruQ-combined',
                       help='Path to output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--symlinks', action='store_true',
                       help='Use symlinks instead of copying files')

    args = parser.parse_args()

    # Convert to absolute paths
    source_dir = Path(__file__).parent / args.source
    source_dir = source_dir.resolve()
    output_dir = Path(__file__).parent / args.output
    output_dir = output_dir.resolve()

    print("="*70)
    print("FruQ-multi Dataset Preparation")
    print("="*70)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: {args.train_ratio:.1f}/{args.val_ratio:.1f}/{args.test_ratio:.1f}")
    print(f"Use symlinks: {args.symlinks}")
    print("="*70 + "\n")

    # Check if source directory exists
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    # Check if output directory exists
    if output_dir.exists():
        response = input(f"\nOutput directory already exists: {output_dir}\nDelete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(output_dir)
            print("Deleted existing output directory.")
        else:
            print("Aborted.")
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect all images
    print("\nStep 1: Collecting images from source directory...")
    images_by_class = collect_images(source_dir)

    # Step 2: Split data
    print("\nStep 2: Splitting data into train/val/test sets...")
    splits = split_data(images_by_class, args.train_ratio, args.val_ratio, args.test_ratio)

    # Step 3: Copy images to new structure
    print("\nStep 3: Organizing images...")
    copy_images(splits, output_dir, use_symlinks=args.symlinks)

    # Step 4: Print and save statistics
    print_statistics(splits)
    save_statistics_report(splits, output_dir)

    print("✓ Dataset preparation completed successfully!")
    print(f"\nYou can now use this dataset with scenario1_baseline.py:")
    print(f"  python src/scenario1_baseline.py")


if __name__ == "__main__":
    main()

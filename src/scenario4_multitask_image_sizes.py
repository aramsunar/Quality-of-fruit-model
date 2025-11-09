"""
Scenario 4 (Multi-Task): Various Image Sizes

This scenario tests multiple input image resolutions (64x64, 128x128, 224x224, 299x299)
for multi-task learning to find the optimal balance between computational cost and model
accuracy for both fruit type and quality classification. Smaller images train faster but
may lose important details, while larger images preserve more information but require
more computation.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import set_seed, get_device, create_directories, print_section, format_time, save_config_to_json
from multitask_data_loader import create_multitask_data_loaders, analyze_multitask_dataset
from multitask_model import get_multitask_model, print_multitask_model_summary
from multitask_training import (
    MultiTaskLoss, MultiTaskTrainer, create_optimizer, create_scheduler,
    compute_multitask_class_weights
)
from multitask_evaluation import (
    evaluate_and_save_multitask_results, plot_multitask_training_history
)


class Scenario4MultitaskConfig:
    """Configuration for Scenario 4 Multi-Task."""

    # Scenario info
    SCENARIO_NAME = "scenario4_multitask_image_sizes"
    SCENARIO_DESCRIPTION = "Multi-task testing various image sizes"

    # Data parameters
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "FruQ-combined"
    IMAGE_SIZES = [64, 128, 224, 299]  # Different image sizes to test
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    GRAYSCALE = False
    AUGMENT = False

    # Model parameters
    MODEL_NAME = "simple"  # Options: 'simple'
    INPUT_CHANNELS = 3  # RGB
    NUM_QUALITY_CLASSES = 3  # Good, Mild, Rotten
    NUM_FRUIT_CLASSES = 11  # Number of fruit types

    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "adam"  # Options: 'adam', 'sgd', 'adamw'
    SCHEDULER = "plateau"  # Options: 'plateau', 'cosine', 'step'
    PATIENCE = 10

    # Advanced training parameters
    WARMUP_EPOCHS = 5  # Learning rate warmup
    GRAD_CLIP = 1.0  # Gradient clipping value (None to disable)
    MIXED_PRECISION = True  # Use mixed precision training (requires CUDA)

    # Multi-task loss weights
    QUALITY_LOSS_WEIGHT = 1.0  # Weight for quality classification loss
    FRUIT_LOSS_WEIGHT = 1.0  # Weight for fruit type classification loss

    # Class imbalance handling
    USE_CLASS_WEIGHTS = True  # Automatically compute and use class weights

    # Other
    SEED = 42


def run_single_size_experiment(img_size, config, quality_classes, fruit_types,
                                num_quality_classes, num_fruit_classes):
    """
    Run multi-task experiment for a single image size.

    Args:
        img_size (int): Image size to test
        config: Configuration object
        quality_classes (list): List of quality class names
        fruit_types (list): List of fruit type names
        num_quality_classes (int): Number of quality classes
        num_fruit_classes (int): Number of fruit type classes

    Returns:
        dict: Results for this image size
    """
    print_section(f"TESTING IMAGE SIZE: {img_size}x{img_size}", 50)

    # Set random seed
    set_seed(config.SEED)

    # Get device
    device = get_device()

    # Create directories
    base_dir = Path(__file__).parent.parent
    size_name = f"{config.SCENARIO_NAME}_size{img_size}"
    paths = create_directories(base_dir, size_name)

    # Save configuration for this size
    size_config = config.__dict__.copy() if hasattr(config, '__dict__') else {}
    size_config['CURRENT_IMG_SIZE'] = img_size
    save_config_to_json(config, paths['reports'] / f'config_size{img_size}.json')

    # Create data loaders
    print(f"Loading data with image size {img_size}x{img_size}...")
    train_loader, val_loader, test_loader, _, _ = create_multitask_data_loaders(
        data_dir=config.DATA_DIR,
        img_size=img_size,
        batch_size=config.BATCH_SIZE,
        grayscale=config.GRAYSCALE,
        augment=config.AUGMENT,
        num_workers=config.NUM_WORKERS
    )

    # Create multi-task model
    print(f"Creating multi-task model for {img_size}x{img_size} input...")
    model = get_multitask_model(
        model_name=config.MODEL_NAME,
        num_quality_classes=num_quality_classes,
        num_fruit_classes=num_fruit_classes,
        input_channels=config.INPUT_CHANNELS,
        img_size=img_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create loss function with optional class weights
    if config.USE_CLASS_WEIGHTS:
        quality_weights, fruit_weights = compute_multitask_class_weights(
            train_loader, num_quality_classes, num_fruit_classes, device
        )
        criterion = MultiTaskLoss(
            quality_weight=config.QUALITY_LOSS_WEIGHT,
            fruit_weight=config.FRUIT_LOSS_WEIGHT,
            quality_class_weights=quality_weights,
            fruit_class_weights=fruit_weights
        )
        print("Using weighted multi-task loss to handle class imbalance")
    else:
        criterion = MultiTaskLoss(
            quality_weight=config.QUALITY_LOSS_WEIGHT,
            fruit_weight=config.FRUIT_LOSS_WEIGHT
        )
        print("Using standard multi-task loss")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config.OPTIMIZER,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=config.SCHEDULER,
        T_max=config.NUM_EPOCHS
    )

    # Create multi-task trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=paths['models'],
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        warmup_epochs=config.WARMUP_EPOCHS,
        grad_clip=config.GRAD_CLIP,
        mixed_precision=config.MIXED_PRECISION
    )

    # Train model
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time

    # Plot training history
    plot_multitask_training_history(
        history=history,
        save_path=paths['figures'] / 'training_history.png'
    )

    # Load best model
    trainer.load_checkpoint('best_model.pth')

    # Evaluate on validation set
    val_quality_metrics, val_fruit_metrics = evaluate_and_save_multitask_results(
        model=model,
        data_loader=val_loader,
        device=device,
        num_quality_classes=num_quality_classes,
        num_fruit_classes=num_fruit_classes,
        quality_classes=quality_classes,
        fruit_classes=fruit_types,
        paths=paths,
        split_name='val'
    )

    # Evaluate on test set if available
    test_quality_metrics = None
    test_fruit_metrics = None
    if test_loader:
        test_quality_metrics, test_fruit_metrics = evaluate_and_save_multitask_results(
            model=model,
            data_loader=test_loader,
            device=device,
            num_quality_classes=num_quality_classes,
            num_fruit_classes=num_fruit_classes,
            quality_classes=quality_classes,
            fruit_classes=fruit_types,
            paths=paths,
            split_name='test'
        )

    # Calculate combined accuracies
    val_combined_acc = (val_quality_metrics['accuracy'] + val_fruit_metrics['accuracy']) / 2
    test_combined_acc = None
    if test_quality_metrics and test_fruit_metrics:
        test_combined_acc = (test_quality_metrics['accuracy'] + test_fruit_metrics['accuracy']) / 2

    return {
        'img_size': img_size,
        'total_params': total_params,
        'training_time': training_time,
        'val_quality_metrics': val_quality_metrics,
        'val_fruit_metrics': val_fruit_metrics,
        'val_combined_acc': val_combined_acc,
        'test_quality_metrics': test_quality_metrics,
        'test_fruit_metrics': test_fruit_metrics,
        'test_combined_acc': test_combined_acc,
        'best_epoch': trainer.best_epoch,
        'history': history
    }


def run_scenario4_multitask(config=None):
    """
    Run Scenario 4 (Multi-Task): Testing various image sizes.

    Args:
        config: Configuration object (default: Scenario4MultitaskConfig)

    Returns:
        dict: Results for all image sizes
    """
    if config is None:
        config = Scenario4MultitaskConfig()

    print_section("SCENARIO 4 (MULTI-TASK): VARIOUS IMAGE SIZES")

    # Check if data directory exists
    if not config.DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {config.DATA_DIR}")
        print("Please place your dataset in the data/processed/ directory.")
        return None

    # Analyze dataset
    print_section("DATASET ANALYSIS", 50)
    stats = analyze_multitask_dataset(config.DATA_DIR)
    print(f"Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split].get('total_images', 0) > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats[split]['total_images']}")
            print(f"  Quality classes: {stats[split]['num_quality_classes']}")
            print(f"  Fruit types: {stats[split]['num_fruit_types']}")

    # Get class names and number of classes
    quality_classes = stats['train']['quality_classes']
    fruit_types = stats['train']['fruit_types']
    num_quality_classes = len(quality_classes)
    num_fruit_classes = len(fruit_types)

    print(f"\nImage sizes to test: {config.IMAGE_SIZES}")
    print(f"Total experiments: {len(config.IMAGE_SIZES)}")
    print(f"Quality classes: {num_quality_classes}")
    print(f"Fruit types: {num_fruit_classes}")

    # Run experiments for each image size
    all_results = []

    for img_size in config.IMAGE_SIZES:
        result = run_single_size_experiment(
            img_size, config, quality_classes, fruit_types,
            num_quality_classes, num_fruit_classes
        )
        all_results.append(result)

        print(f"\nResults for {img_size}x{img_size}:")
        print(f"  Training time: {format_time(result['training_time'])}")
        print(f"  Best epoch: {result['best_epoch']}")
        print(f"  Val Quality Accuracy: {result['val_quality_metrics']['accuracy']:.4f}")
        print(f"  Val Fruit Type Accuracy: {result['val_fruit_metrics']['accuracy']:.4f}")
        print(f"  Val Combined Accuracy: {result['val_combined_acc']:.4f}")
        if result['test_quality_metrics'] and result['test_fruit_metrics']:
            print(f"  Test Quality Accuracy: {result['test_quality_metrics']['accuracy']:.4f}")
            print(f"  Test Fruit Type Accuracy: {result['test_fruit_metrics']['accuracy']:.4f}")
            print(f"  Test Combined Accuracy: {result['test_combined_acc']:.4f}")
        print("-" * 70)

    # Create comparison summary
    print_section("COMPARISON SUMMARY", 50)

    comparison_data = []
    for result in all_results:
        row = {
            'Image Size': f"{result['img_size']}x{result['img_size']}",
            'Parameters': f"{result['total_params']:,}",
            'Training Time': format_time(result['training_time']),
            'Best Epoch': result['best_epoch'],
            'Val Quality Acc': f"{result['val_quality_metrics']['accuracy']:.4f}",
            'Val Fruit Acc': f"{result['val_fruit_metrics']['accuracy']:.4f}",
            'Val Combined Acc': f"{result['val_combined_acc']:.4f}",
            'Val Quality F1': f"{result['val_quality_metrics']['f1_score']:.4f}",
            'Val Fruit F1': f"{result['val_fruit_metrics']['f1_score']:.4f}",
        }
        if result['test_quality_metrics'] and result['test_fruit_metrics']:
            row['Test Quality Acc'] = f"{result['test_quality_metrics']['accuracy']:.4f}"
            row['Test Fruit Acc'] = f"{result['test_fruit_metrics']['accuracy']:.4f}"
            row['Test Combined Acc'] = f"{result['test_combined_acc']:.4f}"
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison to CSV
    base_dir = Path(__file__).parent.parent
    comparison_path = base_dir / 'reports' / config.SCENARIO_NAME / 'size_comparison.csv'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")

    # Find best configuration based on combined accuracy
    best_val_combined_acc = max(r['val_combined_acc'] for r in all_results)
    best_result = next(r for r in all_results if r['val_combined_acc'] == best_val_combined_acc)

    print(f"\nBest Configuration (Based on Combined Validation Accuracy):")
    print(f"  Image Size: {best_result['img_size']}x{best_result['img_size']}")
    print(f"  Validation Quality Accuracy: {best_result['val_quality_metrics']['accuracy']:.4f}")
    print(f"  Validation Fruit Type Accuracy: {best_result['val_fruit_metrics']['accuracy']:.4f}")
    print(f"  Validation Combined Accuracy: {best_result['val_combined_acc']:.4f}")
    if best_result['test_quality_metrics'] and best_result['test_fruit_metrics']:
        print(f"  Test Quality Accuracy: {best_result['test_quality_metrics']['accuracy']:.4f}")
        print(f"  Test Fruit Type Accuracy: {best_result['test_fruit_metrics']['accuracy']:.4f}")
        print(f"  Test Combined Accuracy: {best_result['test_combined_acc']:.4f}")
    print(f"  Training Time: {format_time(best_result['training_time'])}")

    # Additional analysis: Best for each task
    best_quality = max(all_results, key=lambda r: r['val_quality_metrics']['accuracy'])
    best_fruit = max(all_results, key=lambda r: r['val_fruit_metrics']['accuracy'])
    fastest = min(all_results, key=lambda r: r['training_time'])

    print("\nAdditional Analysis:")
    print(f"  Best for Quality Classification: {best_quality['img_size']}x{best_quality['img_size']} "
          f"({best_quality['val_quality_metrics']['accuracy']:.4f})")
    print(f"  Best for Fruit Type Classification: {best_fruit['img_size']}x{best_fruit['img_size']} "
          f"({best_fruit['val_fruit_metrics']['accuracy']:.4f})")
    print(f"  Fastest Training: {fastest['img_size']}x{fastest['img_size']} "
          f"({format_time(fastest['training_time'])})")

    print_section("SCENARIO 4 (MULTI-TASK) COMPLETED", 50)

    return {
        'scenario_name': config.SCENARIO_NAME,
        'scenario_description': config.SCENARIO_DESCRIPTION,
        'config': config,
        'all_results': all_results,
        'comparison_df': comparison_df,
        'best_result': best_result,
        'best_quality': best_quality,
        'best_fruit': best_fruit,
        'fastest': fastest,
        'quality_classes': quality_classes,
        'fruit_types': fruit_types
    }


if __name__ == "__main__":
    results = run_scenario4_multitask()
    if results:
        print("\nScenario 4 (Multi-Task) completed successfully!")
        print(f"\nBest overall configuration: {results['best_result']['img_size']}x{results['best_result']['img_size']}")
        print(f"Combined validation accuracy: {results['best_result']['val_combined_acc']:.4f}")
        print("Check the reports directory for detailed comparisons.")

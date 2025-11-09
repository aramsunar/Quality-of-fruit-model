"""
Scenario 4: Various Image Sizes

This scenario tests multiple input image resolutions (64x64, 128x128, 224x224, 299x299)
to find the optimal balance between computational cost and model accuracy. Smaller images
train faster but may lose important details, while larger images preserve more information
but require more computation.
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
from data_loader import create_data_loaders, analyze_dataset
from model import get_model
from training import Trainer, create_optimizer, create_scheduler, compute_class_weights
from evaluation import evaluate_and_save_results, plot_training_history


class Scenario4Config:
    """Configuration for Scenario 4."""

    # Scenario info
    SCENARIO_NAME = "scenario4_image_sizes"
    SCENARIO_DESCRIPTION = "Testing various image sizes"

    # Data parameters
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "FruQ-combined"
    IMAGE_SIZES = [64, 128, 224, 299]  # Different image sizes to test
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    GRAYSCALE = False
    AUGMENT = False

    # Model parameters
    MODEL_NAME = "simple"  # Options: 'simple', 'deep', 'light'
    INPUT_CHANNELS = 3  # RGB

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

    # Class imbalance handling
    USE_CLASS_WEIGHTS = True  # Automatically compute and use class weights

    # Other
    SEED = 42


def run_single_size_experiment(img_size, config, class_names, num_classes):
    """
    Run experiment for a single image size.

    Args:
        img_size (int): Image size to test
        config: Configuration object
        class_names (list): List of class names
        num_classes (int): Number of classes

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

    # Save configuration
    save_config_to_json(config, paths['reports'] / f'config_size{img_size}.json')

    # Create data loaders
    print(f"Loading data with image size {img_size}x{img_size}...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        data_dir=config.DATA_DIR,
        img_size=img_size,
        batch_size=config.BATCH_SIZE,
        grayscale=config.GRAYSCALE,
        augment=config.AUGMENT,
        num_workers=config.NUM_WORKERS
    )

    # Create model
    print(f"Creating model for {img_size}x{img_size} input...")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        input_channels=config.INPUT_CHANNELS,
        img_size=img_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create loss function with optional class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_loader, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss to handle class imbalance")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

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

    # Create trainer
    trainer = Trainer(
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
    plot_training_history(
        history=history,
        save_path=paths['figures'] / 'training_history.png'
    )

    # Load best model
    trainer.load_checkpoint('best_model.pth')

    # Evaluate on validation set
    val_metrics = evaluate_and_save_results(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        paths=paths,
        split_name='val'
    )

    # Evaluate on test set if available
    test_metrics = None
    if test_loader:
        test_metrics = evaluate_and_save_results(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            paths=paths,
            split_name='test'
        )

    return {
        'img_size': img_size,
        'total_params': total_params,
        'training_time': training_time,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_epoch': trainer.best_epoch,
        'history': history
    }


def run_scenario4(config=None):
    """
    Run Scenario 4: Testing various image sizes.

    Args:
        config: Configuration object (default: Scenario4Config)

    Returns:
        dict: Results for all image sizes
    """
    if config is None:
        config = Scenario4Config()

    print_section("SCENARIO 4: VARIOUS IMAGE SIZES")

    # Check if data directory exists
    if not config.DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {config.DATA_DIR}")
        print("Please place your dataset in the data/raw/ directory.")
        return None

    # Analyze dataset
    print_section("DATASET ANALYSIS", 50)
    stats = analyze_dataset(config.DATA_DIR)
    print(f"Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        if stats[split]:
            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats[split]['total_images']}")
            print(f"  Number of classes: {stats[split]['num_classes']}")

    # Get class names and number of classes
    class_names = stats['train']['classes']
    num_classes = len(class_names)

    print(f"\nImage sizes to test: {config.IMAGE_SIZES}")
    print(f"Total experiments: {len(config.IMAGE_SIZES)}")

    # Run experiments for each image size
    all_results = []

    for img_size in config.IMAGE_SIZES:
        result = run_single_size_experiment(img_size, config, class_names, num_classes)
        all_results.append(result)

        print(f"\nResults for {img_size}x{img_size}:")
        print(f"  Training time: {format_time(result['training_time'])}")
        print(f"  Best epoch: {result['best_epoch']}")
        print(f"  Val Accuracy: {result['val_metrics']['accuracy']:.4f}")
        if result['test_metrics']:
            print(f"  Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
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
            'Val Accuracy': f"{result['val_metrics']['accuracy']:.4f}",
            'Val F1-Score': f"{result['val_metrics']['f1_score']:.4f}",
        }
        if result['test_metrics']:
            row['Test Accuracy'] = f"{result['test_metrics']['accuracy']:.4f}"
            row['Test F1-Score'] = f"{result['test_metrics']['f1_score']:.4f}"
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison to CSV
    base_dir = Path(__file__).parent.parent
    comparison_path = base_dir / 'reports' / config.SCENARIO_NAME / 'size_comparison.csv'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")

    # Find best configuration
    best_val_acc = max(r['val_metrics']['accuracy'] for r in all_results)
    best_result = next(r for r in all_results if r['val_metrics']['accuracy'] == best_val_acc)

    print(f"\nBest Configuration:")
    print(f"  Image Size: {best_result['img_size']}x{best_result['img_size']}")
    print(f"  Validation Accuracy: {best_result['val_metrics']['accuracy']:.4f}")
    if best_result['test_metrics']:
        print(f"  Test Accuracy: {best_result['test_metrics']['accuracy']:.4f}")
    print(f"  Training Time: {format_time(best_result['training_time'])}")

    print_section("SCENARIO 4 COMPLETED", 50)

    return {
        'scenario_name': config.SCENARIO_NAME,
        'scenario_description': config.SCENARIO_DESCRIPTION,
        'config': config,
        'all_results': all_results,
        'comparison_df': comparison_df,
        'best_result': best_result,
        'class_names': class_names
    }


if __name__ == "__main__":
    results = run_scenario4()
    if results:
        print("\nScenario 4 completed successfully!")
        print("Check the reports directory for detailed comparisons.")

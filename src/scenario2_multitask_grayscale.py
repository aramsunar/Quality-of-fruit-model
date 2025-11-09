"""
Scenario 2 (Multi-Task): Grayscale Images

This scenario converts all images to grayscale to assess whether color information
is critical for multi-task fruit type and quality assessment. Comparing results
with Scenario 1 will reveal the importance of color features for both tasks.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import set_seed, get_device, create_directories, print_section, save_config_to_json
from multitask_data_loader import create_multitask_data_loaders, analyze_multitask_dataset
from multitask_model import get_multitask_model, print_multitask_model_summary
from multitask_training import (
    MultiTaskLoss, MultiTaskTrainer, create_optimizer, create_scheduler,
    compute_multitask_class_weights
)
from multitask_evaluation import (
    evaluate_and_save_multitask_results, plot_multitask_training_history
)


class Scenario2MultitaskConfig:
    """Configuration for Scenario 2 Multi-Task."""

    # Scenario info
    SCENARIO_NAME = "scenario2_multitask_grayscale"
    SCENARIO_DESCRIPTION = "Multi-task with grayscale images to test color dependency"

    # Data parameters
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "FruQ-combined"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    GRAYSCALE = True  # Convert to grayscale
    AUGMENT = False

    # Model parameters
    MODEL_NAME = "simple"  # Options: 'simple'
    INPUT_CHANNELS = 1  # Grayscale
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


def run_scenario2_multitask(config=None):
    """
    Run Scenario 2 (Multi-Task): Grayscale images.

    Args:
        config: Configuration object (default: Scenario2MultitaskConfig)

    Returns:
        dict: Results including metrics and history
    """
    if config is None:
        config = Scenario2MultitaskConfig()

    print_section("SCENARIO 2 (MULTI-TASK): GRAYSCALE IMAGES")

    # Set random seed for reproducibility
    set_seed(config.SEED)

    # Get device
    device = get_device()
    print(f"Using device: {device}\n")

    # Create directories for saving results
    base_dir = Path(__file__).parent.parent
    paths = create_directories(base_dir, config.SCENARIO_NAME)
    print(f"Model save directory: {paths['models']}")
    print(f"Reports save directory: {paths['reports']}")
    print(f"Figures save directory: {paths['figures']}\n")

    # Save configuration
    save_config_to_json(config, paths['reports'] / 'config.json')

    # Analyze dataset
    print_section("DATASET ANALYSIS", 50)
    if not config.DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {config.DATA_DIR}")
        print("Please place your dataset in the data/processed/ directory.")
        return None

    stats = analyze_multitask_dataset(config.DATA_DIR)
    print(f"Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        if stats[split] and stats[split].get('total_images', 0) > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats[split]['total_images']}")
            print(f"  Quality classes: {stats[split]['num_quality_classes']} - {', '.join(stats[split]['quality_classes'])}")
            print(f"  Fruit types: {stats[split]['num_fruit_types']} - {', '.join(stats[split]['fruit_types'])}")

    # Create data loaders
    print_section("LOADING DATA", 50)
    print("Note: Converting all images to grayscale")
    train_loader, val_loader, test_loader, quality_classes, fruit_types = create_multitask_data_loaders(
        data_dir=config.DATA_DIR,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        grayscale=config.GRAYSCALE,
        augment=config.AUGMENT,
        num_workers=config.NUM_WORKERS
    )

    num_quality_classes = len(quality_classes)
    num_fruit_classes = len(fruit_types)

    print(f"Quality classes ({num_quality_classes}): {', '.join(quality_classes)}")
    print(f"Fruit types ({num_fruit_classes}): {', '.join(fruit_types)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")

    # Create multi-task model
    print_section("CREATING MULTI-TASK MODEL", 50)
    model = get_multitask_model(
        model_name=config.MODEL_NAME,
        num_quality_classes=num_quality_classes,
        num_fruit_classes=num_fruit_classes,
        input_channels=config.INPUT_CHANNELS,
        img_size=config.IMG_SIZE
    )
    model = model.to(device)

    print(f"Input channels: {config.INPUT_CHANNELS} (Grayscale)")
    print_multitask_model_summary(model, input_size=(config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))

    # Create loss function with optional class weights
    print_section("MULTI-TASK LOSS FUNCTION", 50)
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
        print("\nUsing weighted multi-task loss to handle class imbalance")
    else:
        criterion = MultiTaskLoss(
            quality_weight=config.QUALITY_LOSS_WEIGHT,
            fruit_weight=config.FRUIT_LOSS_WEIGHT
        )
        print("\nUsing standard multi-task loss")

    print(f"Quality loss weight: {config.QUALITY_LOSS_WEIGHT}")
    print(f"Fruit type loss weight: {config.FRUIT_LOSS_WEIGHT}")

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
    print_section("TRAINING MULTI-TASK MODEL", 50)
    history = trainer.train()

    # Plot training history
    plot_multitask_training_history(
        history=history,
        save_path=paths['figures'] / 'training_history.png'
    )

    # Load best model for evaluation
    print_section("EVALUATION", 50)
    trainer.load_checkpoint('best_model.pth')

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
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
        print("\nTest Set Evaluation:")
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

    print_section("SCENARIO 2 (MULTI-TASK) COMPLETED", 50)

    # Return results
    results = {
        'scenario_name': config.SCENARIO_NAME,
        'scenario_description': config.SCENARIO_DESCRIPTION,
        'config': config,
        'history': history,
        'val_quality_metrics': val_quality_metrics,
        'val_fruit_metrics': val_fruit_metrics,
        'test_quality_metrics': test_quality_metrics if test_loader else None,
        'test_fruit_metrics': test_fruit_metrics if test_loader else None,
        'quality_classes': quality_classes,
        'fruit_types': fruit_types
    }

    return results


if __name__ == "__main__":
    results = run_scenario2_multitask()
    if results:
        print("\nScenario 2 (Multi-Task) completed successfully!")
        print(f"Best validation quality accuracy: {results['val_quality_metrics']['accuracy']:.4f}")
        print(f"Best validation fruit type accuracy: {results['val_fruit_metrics']['accuracy']:.4f}")
        print(f"Combined accuracy: {(results['val_quality_metrics']['accuracy'] + results['val_fruit_metrics']['accuracy']) / 2:.4f}")
        print("\nCompare this with Scenario 1 (Multi-Task) to assess the importance of color information for both tasks.")

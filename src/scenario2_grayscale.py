"""
Scenario 2: Grayscale Images

This scenario converts all images to grayscale to assess whether color information
is critical for fruit quality assessment. Comparing results with Scenario 1 will
reveal the importance of color features in the classification task.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import set_seed, get_device, create_directories, print_section, save_config_to_json
from data_loader import create_data_loaders, analyze_dataset
from model import get_model
from training import Trainer, create_optimizer, create_scheduler, compute_class_weights
from evaluation import evaluate_and_save_results, plot_training_history


class Scenario2Config:
    """Configuration for Scenario 2."""

    # Scenario info
    SCENARIO_NAME = "scenario2_grayscale"
    SCENARIO_DESCRIPTION = "Grayscale images to test color dependency"

    # Data parameters
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "FruQ-combined"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    GRAYSCALE = True  # Convert to grayscale
    AUGMENT = False

    # Model parameters
    MODEL_NAME = "simple"  # Options: 'simple', 'deep', 'light'
    INPUT_CHANNELS = 1  # Grayscale

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


def run_scenario2(config=None):
    """
    Run Scenario 2: Grayscale images.

    Args:
        config: Configuration object (default: Scenario2Config)

    Returns:
        dict: Results including metrics and history
    """
    if config is None:
        config = Scenario2Config()

    print_section("SCENARIO 2: GRAYSCALE IMAGES")

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
        print("Please place your dataset in the data/raw/ directory.")
        return None

    stats = analyze_dataset(config.DATA_DIR)
    print(f"Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        if stats[split]:
            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats[split]['total_images']}")
            print(f"  Number of classes: {stats[split]['num_classes']}")
            print(f"  Classes: {', '.join(stats[split]['classes'])}")

    # Create data loaders
    print_section("LOADING DATA", 50)
    print("Note: Converting all images to grayscale")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=config.DATA_DIR,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        grayscale=config.GRAYSCALE,
        augment=config.AUGMENT,
        num_workers=config.NUM_WORKERS
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {', '.join(class_names)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")

    # Create model
    print_section("CREATING MODEL", 50)
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        input_channels=config.INPUT_CHANNELS,
        img_size=config.IMG_SIZE
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Input channels: {config.INPUT_CHANNELS} (Grayscale)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function with optional class weights
    print_section("LOSS FUNCTION", 50)
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
    print_section("TRAINING MODEL", 50)
    history = trainer.train()

    # Plot training history
    plot_training_history(
        history=history,
        save_path=paths['figures'] / 'training_history.png'
    )

    # Load best model for evaluation
    print_section("EVALUATION", 50)
    trainer.load_checkpoint('best_model.pth')

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
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
        print("\nTest Set Evaluation:")
        test_metrics = evaluate_and_save_results(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            paths=paths,
            split_name='test'
        )

    print_section("SCENARIO 2 COMPLETED", 50)

    # Return results
    results = {
        'scenario_name': config.SCENARIO_NAME,
        'scenario_description': config.SCENARIO_DESCRIPTION,
        'config': config,
        'history': history,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics if test_loader else None,
        'class_names': class_names
    }

    return results


if __name__ == "__main__":
    results = run_scenario2()
    if results:
        print("\nScenario 2 completed successfully!")
        print(f"Best validation accuracy: {results['val_metrics']['accuracy']:.4f}")
        print("\nCompare this with Scenario 1 to assess the importance of color information.")

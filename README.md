# Fruit Identification and Quality Assessment using Deep Learning

## Project Overview

This project develops deep learning models using Convolutional Neural Networks (CNN) to identify fruit types and assess their quality from images. The project includes both single-task models (quality classification only) and multi-task models (simultaneous fruit type and quality classification). Models are evaluated using comprehensive metrics including accuracy, precision, recall, F1-score, ROC curves, and AUC scores to ensure robust performance.

## Objectives

- Develop CNN-based deep learning models for fruit classification and quality assessment
- Implement both single-task (quality only) and multi-task (fruit type + quality) learning approaches
- Evaluate model performance using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC Curves
  - AUC (Area Under Curve)
- Analyze the impact of different input data characteristics on model performance
- Compare single-task vs multi-task learning performance

## Project Structure

```
fruit-identification-and-quality/
├── venv/                    # Virtual environment
├── data/                    # Dataset storage
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed data
│   └── README.md            # Data documentation
├── src/                     # Source code
│   ├── main.py              # Main entry point for running scenarios
│   │
│   ├── Single-Task Scenarios (Quality Classification Only)
│   ├── scenario1_baseline.py          # Normal colored images
│   ├── scenario2_grayscale.py         # Grayscale images
│   ├── scenario3_augmented.py         # Augmented colored images
│   ├── scenario4_image_sizes.py       # Various image sizes
│   │
│   ├── Multi-Task Scenarios (Fruit Type + Quality Classification)
│   ├── scenario1_multitask.py                # Normal colored images
│   ├── scenario2_multitask_grayscale.py      # Grayscale images
│   ├── scenario3_multitask_augmented.py      # Augmented colored images
│   ├── scenario4_multitask_image_sizes.py    # Various image sizes
│   │
│   ├── Single-Task Components
│   ├── data_loader.py       # Data loading utilities
│   ├── model.py             # CNN model architectures
│   ├── training.py          # Training loops and utilities
│   ├── evaluation.py        # Evaluation metrics and visualization
│   │
│   ├── Multi-Task Components
│   ├── multitask_data_loader.py       # Multi-task data loading (dual labels)
│   ├── multitask_model.py             # Multi-task CNN architectures
│   ├── multitask_training.py          # Multi-task training utilities
│   ├── multitask_evaluation.py        # Multi-task evaluation metrics
│   │
│   └── utils.py             # Shared helper functions
├── results/                 # Saved model checkpoints and outputs
├── reports/                 # Generated reports and visualizations
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Experimental Scenarios

This project investigates different scenarios to analyze how input data characteristics affect model performance. Each scenario is implemented in both **single-task** (quality classification only) and **multi-task** (fruit type + quality classification) variants.

### Single-Task Scenarios (Quality Classification Only)

| Scenario           | Description                                                        | Status | Key Findings |
| ------------------ | ------------------------------------------------------------------ | ------ | ------------ |
| **1. Baseline**    | Normal colored images with standard preprocessing                  | Ready  | -            |
| **2. Grayscale**   | Convert colored images to grayscale to test color dependency       | Ready  | -            |
| **3. Augmented**   | Apply data augmentation techniques to colored images               | Ready  | -            |
| **4. Image Sizes** | Test multiple image resolutions (64x64, 128x128, 224x224, 299x299) | Ready  | -            |

### Multi-Task Scenarios (Fruit Type + Quality Classification)

| Scenario                      | Description                                                 | Status | Key Findings |
| ----------------------------- | ----------------------------------------------------------- | ------ | ------------ |
| **1. Multi-Task Baseline**    | Normal colored images, dual classification heads            | Ready  | -            |
| **2. Multi-Task Grayscale**   | Grayscale images, test color dependency for both tasks      | Ready  | -            |
| **3. Multi-Task Augmented**   | Data augmentation for improved generalization on both tasks | Ready  | -            |
| **4. Multi-Task Image Sizes** | Test image resolutions for both tasks simultaneously        | Ready  | -            |

### Scenario Details

#### Single-Task Scenarios

**Scenario 1: Baseline (Normal Colored Images)**

- Establish baseline performance using standard colored images
- Standard preprocessing: resize, normalization
- Provides benchmark for all other scenarios
- Output: Quality classification (3 classes: Good, Mild, Rotten)

**Scenario 2: Grayscale Images**

- Convert all images to grayscale to assess color dependency
- Determine if color information is critical for fruit quality assessment
- Compare performance metrics against baseline
- Output: Quality classification

**Scenario 3: Data Augmentation**

- Apply augmentation techniques: rotation, flipping, brightness/contrast adjustment
- Evaluate if augmentation improves model generalization
- Analyze impact on overfitting
- Output: Quality classification

**Scenario 4: Various Image Sizes**

- Test multiple input resolutions to find optimal size-performance tradeoff
- Resolutions tested: 64x64, 128x128
- Balance between computational cost and accuracy
- Output: Quality classification

#### Multi-Task Scenarios

**Scenario 1: Multi-Task Baseline (Normal Colored Images)**

- Establish baseline for multi-task learning using standard colored images
- Shared convolutional backbone with dual classification heads
- Standard preprocessing: resize, normalization
- Output: Fruit type (11 classes) + Quality (3 classes)

**Scenario 2: Multi-Task Grayscale**

- Convert all images to grayscale to assess color dependency for both tasks
- Determine if color is critical for fruit type identification vs quality assessment
- Compare impact on both tasks
- Output: Fruit type + Quality

**Scenario 3: Multi-Task Data Augmentation**

- Apply augmentation techniques to improve generalization for both tasks
- Evaluate if augmentation benefits both tasks equally
- Analyze task-specific overfitting patterns
- Output: Fruit type + Quality

**Scenario 4: Multi-Task Various Image Sizes**

- Test multiple input resolutions for dual-head architecture
- Find optimal size for both fruit type and quality classification
- Analyze if different tasks benefit from different resolutions
- Resolutions tested: 64x64, 128x128
- Output: Fruit type + Quality

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Create and activate virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup

Place your fruit dataset in the `data/processed/FruQ-combined/` directory. The expected structure is:

```
data/processed/FruQ-combined/
├── train/
│   ├── Good/
│   ├── Mild/
│   └── Rotten/
├── val/
│   ├── Good/
│   ├── Mild/
│   └── Rotten/
└── test/
    ├── Good/
    ├── Mild/
    └── Rotten/
```

**Important for Multi-Task Learning**:
The multi-task models extract fruit type information from the filename prefix. Image files should follow the naming convention:

```
FruitType_ImageName.png
```

Examples:

- `BananaDB_Image001.png`
- `PeachQ_Image042.png`
- `tomatoQ_Image123.png`

The multi-task data loader automatically parses these filenames to create dual labels:

- Quality label: from folder name (Good/Mild/Rotten)
- Fruit type label: from filename prefix (BananaDB/PeachQ/etc.)

## Usage

### Running Single-Task Scenarios (Quality Classification Only)

Execute scenarios using the main entry point or run files directly:

```bash
# Run specific scenario
python src/main.py --scenario 1  # Baseline
python src/main.py --scenario 2  # Grayscale
python src/main.py --scenario 3  # Augmented
python src/main.py --scenario 4  # Image sizes

# Or run individual scenario files directly
python src/scenario1_baseline.py
python src/scenario2_grayscale.py
python src/scenario3_augmented.py
python src/scenario4_image_sizes.py
```

### Running Multi-Task Scenarios (Fruit Type + Quality Classification)

Run multi-task scenarios directly:

```bash
# Multi-task scenarios
python src/scenario1_multitask.py                # Baseline
python src/scenario2_multitask_grayscale.py      # Grayscale
python src/scenario3_multitask_augmented.py      # Augmented
python src/scenario4_multitask_image_sizes.py    # Various image sizes
```

## Evaluation Metrics

### Single-Task Models (Quality Classification)

The models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC Curves**: True Positive Rate vs False Positive Rate
- **AUC**: Area under the ROC curve

### Multi-Task Models (Fruit Type + Quality Classification)

Multi-task models are evaluated separately for each task:

- **Quality Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC, AUC
- **Fruit Type Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC, AUC
- **Combined Metrics**: Average performance across both tasks
- **Per-Task Visualization**: Separate confusion matrices and ROC curves for each task

## Multi-Task Learning Architecture

The multi-task models use a shared convolutional backbone with dual classification heads:

### Architecture Overview

```
Input Image (RGB or Grayscale)
    ↓
Shared Convolutional Backbone
├── Conv Block 1 (32 filters)
├── Conv Block 2 (64 filters)
├── Conv Block 3 (128 filters)
└── Conv Block 4 (256 filters)
    ↓
Shared Feature Extraction (512 units)
    ↓
    ├─→ Quality Classification Head
    │   ├── FC Layer (256 units)
    │   └── Output (3 classes: Good, Mild, Rotten)
    │
    └─→ Fruit Type Classification Head
        ├── FC Layer (256 units)
        └── Output (11 classes: fruit types)
```

### Key Features

- **Shared Learning**: Both tasks learn from the same low-level features
- **Task-Specific Heads**: Separate fully connected layers for each task
- **Configurable Loss Weights**: Balance quality vs fruit type importance
- **Dual Metrics Tracking**: Monitor both tasks during training
- **Combined Optimization**: Best model selected based on average accuracy

## Technology Stack

- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Computer Vision**: OpenCV, Pillow, torchvision
- **Machine Learning**: scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Report Generation

### Single-Task Results

Results and visualizations are automatically saved in the `results/` directory:

- Confusion matrices
- ROC curves
- Training/validation loss and accuracy curves
- Performance comparison tables
- Metrics CSV files

### Multi-Task Results

Multi-task scenarios generate comprehensive reports for both tasks:

- Separate confusion matrices for quality and fruit type classification
- Separate ROC curves for both tasks
- Multi-panel training history (showing both tasks)
- Combined metrics CSV with per-task breakdown
- Model architecture summaries with parameter counts per head

## Development Status

This is an active research project. The README will be updated throughout development with:

- Experimental results
- Key findings
- Model performance comparisons
- Best practices identified
- Lessons learned

## Contributors

This project is developed as part of a the ITRI626 module focused on practical application of CNN architectures for image classification tasks with the aim to apply the theory that was learned in class to the application and implementation of real world examples.

---

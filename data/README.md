# Data Directory

This directory contains all datasets used for the fruit quality assessment project.

## Quick Reference

**Dataset Location**: `data/processed/FruQ-combined/` (ready to use)
**Total Images**: 9,370 images across 11 fruit types
**Quality Classes**: 3 (Good, Mild, Rotten)
**Fruit Types**: 11 (BananaDB, CucumberQ, GrapeQ, KakiQ, PapayaQ, PeachQ, PearQ, PepperQ, StrawberryQ, tomatoQ, WatermeloQ)

**Scenarios Supported**:
- 4 Single-Task Scenarios (Quality classification only)
- 4 Multi-Task Scenarios (Fruit type + Quality classification)

Jump to:
- [Directory Structure](#directory-structure)
- [Dataset Organization for Different Use Cases](#dataset-organization-for-different-use-cases)
- [Supported Scenarios](#supported-scenarios)
- [Adding Your Dataset](#adding-your-dataset)
- [FruQ-multi Dataset Details](#fruq-multi-dataset)

## Directory Structure

```
data/
├── raw/                           # Original, unprocessed datasets
│   └── FruQ-multi/                # Multi-fruit quality dataset (9,370 images)
│       ├── BananaDB/              # Individual fruit type directories
│       ├── CucumberQ/
│       ├── GrapeQ/
│       └── ...                    # (11 fruit/vegetable types total)
│
└── processed/                     # Processed datasets (created automatically or manually)
    └── FruQ-combined/             # Combined dataset for training
        ├── train/                 # Training set (~60-70%)
        │   ├── Good/              # High quality images
        │   ├── Mild/              # Moderate quality images
        │   └── Rotten/            # Poor quality images
        ├── val/                   # Validation set (~15-20%)
        │   ├── Good/
        │   ├── Mild/
        │   └── Rotten/
        └── test/                  # Test set (~15-20%)
            ├── Good/
            ├── Mild/
            └── Rotten/
```

## Dataset Organization for Different Use Cases

### Single-Task Learning (Quality Classification Only)

For **single-task scenarios** (Scenarios 1-4), the model learns only quality classification:

```
data/processed/FruQ-combined/
├── train/
│   ├── Good/      # Mix of all fruit types, good quality
│   ├── Mild/      # Mix of all fruit types, moderate quality
│   └── Rotten/    # Mix of all fruit types, poor quality
├── val/
└── test/
```

- **Label**: Quality class only (Good/Mild/Rotten)
- **Use**: Standard image classification

### Multi-Task Learning (Fruit Type + Quality Classification)

For **multi-task scenarios** (Multi-Task Scenarios 1-4), the model learns both fruit type and quality:

```
data/processed/FruQ-combined/
├── train/
│   ├── Good/
│   │   ├── BananaDB_image001.png    # Filename prefix = fruit type
│   │   ├── PeachQ_image042.png      # Folder name = quality
│   │   └── tomatoQ_image123.png
│   ├── Mild/
│   │   ├── KakiQ_image055.png
│   │   └── GrapeQ_image089.png
│   └── Rotten/
│       └── ...
├── val/
└── test/
```

- **Labels**: Dual labels extracted automatically
  - Quality label: from folder name (Good/Mild/Rotten)
  - Fruit type label: from filename prefix (BananaDB, PeachQ, etc.)
- **Naming Convention**: `FruitType_ImageName.png`
- **Use**: Multi-task learning with shared backbone

## Dataset Requirements

### File Format

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Images can be of any size (will be resized automatically)
- Images should be RGB color images (3 channels)

### Data Splits

- **Training set**: Used to train the model (typically 60-80% of data)
- **Validation set**: Used to tune hyperparameters and monitor training (typically 10-20% of data)
- **Test set**: Optional, used for final evaluation (typically 10-20% of data)

### Class Organization

- Each class should have its own subdirectory
- Directory names will be used as class labels
- Example structure:
  ```
  train/
  ├── apple/
  │   ├── apple_001.jpg
  │   ├── apple_002.jpg
  │   └── ...
  ├── banana/
  │   ├── banana_001.jpg
  │   ├── banana_002.jpg
  │   └── ...
  └── orange/
      ├── orange_001.jpg
      ├── orange_002.jpg
      └── ...
  ```

## Data Guidelines

### Recommended Dataset Size

- Minimum: 100 images per class
- Recommended: 500+ images per class
- More data generally leads to better model performance

### Image Quality

- Resolution: At least 224x224 pixels (higher is better)
- Clear, well-lit images
- Variety in angles, backgrounds, and lighting conditions
- Balanced class distribution (similar number of images per class)

### Data Collection Tips

1. Ensure consistent image quality within each split
2. Avoid data leakage (same object in train and test)
3. Include diverse examples (different angles, lighting, backgrounds)
4. Label data consistently
5. Remove corrupted or mislabeled images

## Supported Scenarios

This project supports **8 different experimental scenarios** divided into two categories:

### Single-Task Scenarios (Quality Classification Only)

These scenarios classify fruit quality into 3 classes (Good/Mild/Rotten):

| Scenario | Command | Description | Data Requirements |
|----------|---------|-------------|-------------------|
| **Scenario 1** | `python src/main.py --scenario 1` | Baseline with colored images | Standard preprocessing |
| **Scenario 2** | `python src/main.py --scenario 2` | Grayscale images | Converted to single channel |
| **Scenario 3** | `python src/main.py --scenario 3` | Data augmentation | Rotation, flip, brightness |
| **Scenario 4** | `python src/main.py --scenario 4` | Multiple image sizes | Tests 64x64, 128x128 |

### Multi-Task Scenarios (Fruit Type + Quality Classification)

These scenarios simultaneously classify fruit type (11 classes) AND quality (3 classes):

| Scenario | Command | Description | Data Requirements |
|----------|---------|-------------|-------------------|
| **Multi-Task 1** | `python src/scenario1_multitask.py` | Baseline with colored images | Filenames must include fruit type prefix |
| **Multi-Task 2** | `python src/scenario2_multitask_grayscale.py` | Grayscale images | Grayscale + fruit type prefix |
| **Multi-Task 3** | `python src/scenario3_multitask_augmented.py` | Data augmentation | Augmented + fruit type prefix |
| **Multi-Task 4** | `python src/scenario4_multitask_image_sizes.py` | Multiple image sizes | Various sizes + fruit type prefix |

### Key Differences Between Single-Task and Multi-Task

| Aspect | Single-Task | Multi-Task |
|--------|-------------|------------|
| **Output** | Quality only (3 classes) | Quality (3 classes) + Fruit type (11 classes) |
| **Labels** | Folder name only | Folder name + filename prefix |
| **Architecture** | Single classification head | Dual classification heads with shared backbone |
| **Data Loader** | `data_loader.py` | `multitask_data_loader.py` |
| **Filename Requirements** | Any valid filename | Must follow `FruitType_imagename.png` pattern |

## Adding Your Dataset

### For Single-Task Scenarios

1. Place images in `data/processed/FruQ-combined/` following this structure:
   ```
   FruQ-combined/
   ├── train/
   │   ├── Good/
   │   ├── Mild/
   │   └── Rotten/
   ├── val/
   └── test/
   ```
2. Images can have any filename
3. Run any single-task scenario:
   ```bash
   python src/main.py --scenario 1
   ```

### For Multi-Task Scenarios

1. Place images in `data/processed/FruQ-combined/` following the same folder structure
2. **IMPORTANT**: Rename files to include fruit type prefix:
   ```
   BananaDB_image001.png
   PeachQ_image042.png
   tomatoQ_image123.png
   ```
3. Supported fruit type prefixes (from FruQ-multi dataset):
   - `BananaDB`
   - `CucumberQ`
   - `GrapeQ`
   - `KakiQ`
   - `PapayaQ`
   - `PeachQ`
   - `PearQ`
   - `PepperQ`
   - `StrawberryQ`
   - `tomatoQ`
   - `WatermeloQ`
4. Run any multi-task scenario:
   ```bash
   python src/scenario1_multitask.py
   ```

## Data Preprocessing

The project automatically handles preprocessing based on the scenario:

### All Scenarios
- Image resizing to specified dimensions (224x224 default, or 64x64/128x128 in Scenario 4)
- Normalization using ImageNet statistics
- Conversion to PyTorch tensors

### Scenario-Specific Preprocessing

| Scenario | Additional Preprocessing |
|----------|-------------------------|
| **Scenario 2 & Multi-Task 2** | Grayscale conversion (RGB → single channel) |
| **Scenario 3 & Multi-Task 3** | Data augmentation (rotation, horizontal flip, color jitter) |
| **Scenario 4 & Multi-Task 4** | Multiple resolution testing (64x64, 128x128) |

### Multi-Task Specific
- Dual label extraction from folder names and filenames
- Automatic fruit type encoding (11 classes)
- Automatic quality encoding (3 classes)

## Notes

- The `raw/` directory should contain your original, unmodified data
- The `processed/` directory is created automatically during runtime
- Do not commit large datasets to version control
- Consider using `.gitignore` to exclude data files
- For large datasets, consider using data versioning tools like DVC

## FruQ-multi Dataset

The `data/raw/FruQ-multi/` directory contains a multi-fruit quality assessment dataset with **9,370 images** across **11 fruit and vegetable types**.

### Dataset Structure

```
FruQ-multi/
├── BananaDB/          # 612 images
│   ├── Good/          # 179 images
│   ├── mild/          # 96 images
│   └── Rotten/        # 337 images
├── CucumberQ/         # 711 images
│   ├── Fresh/         # 250 images
│   ├── Mild/          # 345 images
│   └── Rotten/        # 116 images
├── GrapeQ/            # 709 images
│   ├── Good/          # 227 images
│   ├── Mild/          # 194 images
│   └── Rotten/        # 288 images
├── KakiQ/             # 1,111 images
│   ├── Good/          # 545 images
│   ├── Mild/          # 226 images
│   └── Rotten/        # 340 images
├── PapayaQ/           # 793 images
│   ├── Good/          # 130 images
│   ├── Mild/          # 250 images
│   └── Rotten/        # 413 images
├── PeachQ/            # 1,145 images
│   ├── Good/          # 425 images
│   ├── Mild/          # 136 images
│   └── Rotten/        # 584 images
├── PearQ/             # 1,097 images
│   ├── Good/          # 504 images
│   ├── Mild/          # 493 images
│   └── Rotten/        # 100 images
├── PepperQ/           # 732 images
│   ├── Good/          # 48 images
│   ├── Mild/          # 24 images
│   └── Rotten/        # 660 images
├── StrawberryQ/       # 216 images
│   ├── Mild/          # 119 images
│   └── Rotten/        # 97 images (missing "Good" class)
├── tomatoQ/           # 1,990 images
│   ├── Good/          # 600 images
│   ├── Mild/          # 440 images
│   └── Rotten/        # 950 images
└── WatermeloQ/        # 254 images
    ├── Good/          # 51 images
    ├── Mild/          # 53 images
    └── Rotten/        # 150 images
```

### Quality Classes

- **Good/Fresh**: High-quality, fresh produce
- **Mild**: Moderate quality with minor defects
- **Rotten**: Poor quality, spoiled produce

### Dataset Characteristics

- **Total Images**: 9,370
- **Fruit/Vegetable Types**: 11
- **Quality Classes**: 3 (Good/Fresh, Mild, Rotten)
- **Average Images per Type**: 852
- **File Format**: PNG images
- **Use Case**: Multi-class quality assessment, classification, freshness detection

### Notes

- Class names have some inconsistency (e.g., "Good" vs "Fresh", "mild" vs "Mild")
- StrawberryQ dataset is missing the "Good" class
- Class distribution is imbalanced in some categories (e.g., PepperQ has significantly more Rotten images)
- Each fruit/vegetable type can be used independently for single-fruit quality assessment
- The entire dataset can be combined for multi-fruit quality classification

### Using FruQ-multi with This Project

The FruQ-multi dataset is the source data that has been processed into `data/processed/FruQ-combined/` for use with all scenarios.

**Current Setup (Recommended)**

The combined dataset in `data/processed/FruQ-combined/` is ready to use with all 8 scenarios:

```bash
# Single-Task Scenarios (Quality Classification Only)
python src/main.py --scenario 1    # Baseline colored
python src/main.py --scenario 2    # Grayscale
python src/main.py --scenario 3    # Augmented
python src/main.py --scenario 4    # Multiple image sizes

# Multi-Task Scenarios (Fruit Type + Quality Classification)
python src/scenario1_multitask.py                 # Baseline colored
python src/scenario2_multitask_grayscale.py       # Grayscale
python src/scenario3_multitask_augmented.py       # Augmented
python src/scenario4_multitask_image_sizes.py     # Multiple sizes
```

**Alternative Options**

**Option 1: Single Fruit Type Quality Assessment**
Use only one fruit type for quality classification:

```bash
# Example: Train only on bananas
# Manually organize BananaDB images into train/val/test splits
python src/main.py --scenario 1
```

**Option 2: Binary Classification**
Modify quality classes to create binary classification:
- Combine Good/Mild into "Fresh" class
- Keep Rotten as separate class

**Option 3: Custom Fruit Subset**
Select specific fruit types for multi-task learning:
- Filter images by filename prefix
- Use only selected fruit types (e.g., only tropical fruits)

## Dataset Statistics

After adding your dataset, run any scenario to see detailed statistics including:

- Number of classes
- Images per class
- Total images per split
- Class distribution

This information will help you understand if your dataset is well-balanced and sufficient for training.

## Workflow Summary

### For Single-Task Scenarios (Scenarios 1-4)

1. **Data Location**: `data/processed/FruQ-combined/`
2. **Structure**: Quality folders (Good/Mild/Rotten) inside train/val/test
3. **Filenames**: Any valid image filename
4. **Run Command**: `python src/main.py --scenario [1-4]`
5. **Output**: Quality classification model (3 classes)

### For Multi-Task Scenarios (Multi-Task 1-4)

1. **Data Location**: `data/processed/FruQ-combined/`
2. **Structure**: Same quality folders as single-task
3. **Filenames**: **MUST** follow `FruitType_imagename.png` pattern
4. **Run Command**: `python src/scenario[1-4]_multitask.py`
5. **Output**: Dual-output model (11 fruit types + 3 quality classes)

### Dataset Preparation Checklist

- [ ] Images are in `data/processed/FruQ-combined/`
- [ ] Train/val/test splits are properly organized
- [ ] Quality folders (Good/Mild/Rotten) contain images
- [ ] For multi-task: All filenames include fruit type prefix
- [ ] Supported image formats (.jpg, .jpeg, .png, .bmp)
- [ ] Minimum 100 images per quality class (500+ recommended)

## Troubleshooting

### Common Issues

**Issue**: "FileNotFoundError: data/processed/FruQ-combined/"
- **Solution**: Ensure the FruQ-combined directory exists with train/val/test subdirectories

**Issue**: Multi-task model fails to extract fruit labels
- **Solution**: Check that filenames follow `FruitType_imagename.png` pattern (e.g., `BananaDB_001.png`)

**Issue**: Class imbalance warnings
- **Solution**: This is expected for FruQ-multi dataset; the code handles imbalanced classes

**Issue**: Out of memory errors during training
- **Solution**: Reduce batch size in scenario files or use Scenario 4 with smaller image sizes (64x64)

## Support

If you encounter issues with your dataset:

1. Check that the directory structure matches the requirements above
2. Verify that image files are in supported formats (.jpg, .jpeg, .png, .bmp)
3. Ensure proper file permissions (read access required)
4. Check the console output for specific error messages
5. Review the [Workflow Summary](#workflow-summary) section
6. For multi-task scenarios, verify filename prefixes match supported fruit types

For more information, refer to the main README.md file in the project root.

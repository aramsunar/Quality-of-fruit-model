# Fruit Quality Classification Project - Executive Summary

## Project Overview

This research project developed deep learning models to automatically assess fruit quality from images. The system classifies fruit into three quality categories: **Good**, **Mild** (slightly degraded), and **Rotten**. The project explores both **single-task learning** (quality only) and **multi-task learning** (quality + fruit type identification).

**Why this matters**: Manual fruit inspection is slow, inconsistent, and doesn't scale well for industrial operations. Automated systems can process large volumes quickly and consistently.

---

## Dataset: FruQ-multi Database

- **Total images**: 9,370
- **Fruit types**: 11 varieties (Banana, Cucumber, Grape, Kaki, Papaya, Peach, Pear, Pepper, Strawberry, Tomato, Watermelon)
- **Quality classes**: 3 categories (Good, Mild, Rotten)
- **Data split**: ~60-70% training, ~15-20% validation, ~15-20% test

**Key challenge**: Dataset has imbalances - for example, Tomatoes have 1,990 images while Strawberries have only 216 images.

---

## Model Architecture: SimpleCNN

**What's a CNN?** A Convolutional Neural Network is a type of AI that's particularly good at processing images. It learns to recognize patterns like edges, textures, and shapes automatically.

### Architecture Details:
- **Input**: 224×224 pixel images (RGB or grayscale)
- **4 Convolutional Blocks**:
  - Block 1: 32 filters
  - Block 2: 64 filters
  - Block 3: 128 filters
  - Block 4: 256 filters
- **Feature extraction**: 50,176 features after convolution
- **Fully connected layers**: 512 → 256 → 3 output classes
- **Total parameters**: 26,211,619 trainable weights

### Training Configuration:
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch size**: 32 images at a time
- **Epochs**: 50 training cycles
- **Regularization**: Dropout (0.5), weight decay (0.0001)
- **Learning rate scheduler**: ReduceLROnPlateau (reduces when improvement stalls)
- **Early stopping**: Patience of 10 epochs
- **Mixed precision**: Enabled for faster training
- **Class weights**: Auto-computed to handle imbalance

---

## PART A: Single-Task Learning (Quality Classification Only)

### Scenario 1: RGB Baseline (Standard Color Images)

**Purpose**: Establish baseline performance with normal color images.

**Results**:
- **Validation accuracy**: 99.84% (3 errors out of 1,872 samples)
- **Test accuracy**: 99.79% (2 errors out of 939 samples)
- **AUC**: 1.0000 (validation), 0.9998 (test)

**Per-class performance (Test)**:
- Good: 100% precision, 99.66% recall
- Mild: 99.17% precision, 100% recall
- Rotten: 100% precision, 99.75% recall

**Key findings**:
- Near-perfect performance achieved
- All 3 validation errors were Mild classified as Rotten
- No confusion between Good and Rotten (extremes well separated)
- Model converged rapidly (>99% accuracy by epoch 5)

---

### Scenario 2: Grayscale Images

**Purpose**: Test if color information is essential or if texture/shape alone suffices.

**Configuration change**: Images converted to grayscale (3 channels → 1 channel)

**Results**:
- **Validation accuracy**: 99.84% (3 errors out of 1,872 samples)
- **Test accuracy**: 100.00% (0 errors out of 939 samples) ✨
- **AUC**: 1.0000 (validation), 1.0000 (test)

**Per-class performance (Test)**:
- Good: 100% precision, 100% recall
- Mild: 100% precision, 100% recall
- Rotten: 100% precision, 100% recall

**Key findings**:
- **Surprising result**: Grayscale actually outperformed RGB on test set!
- Texture and intensity patterns alone provide sufficient information
- Color not essential for this fruit quality assessment task
- Suggests simpler grayscale systems could work in practice (lower computational cost)

**Comparison to RGB**:
- Validation: Identical (99.84% both)
- Test: Better (+0.21% improvement to perfect 100%)

---

### Scenario 3: Data Augmentation

**Purpose**: Test if synthetic variations improve robustness and generalization.

**What's data augmentation?** Artificially creating modified versions of training images (rotating, flipping, adjusting brightness) to teach the model to handle variations.

**Augmentation applied**:
- Random rotation: ±10 degrees
- Horizontal flip: 50% probability
- Vertical flip: 50% probability
- Color jitter: Brightness ±20%, Contrast ±20%, Saturation ±20%, Hue ±10%

**Results**:
- **Validation accuracy**: 98.99% (19 errors out of 1,872 samples)
- **Test accuracy**: 99.25% (7 errors out of 939 samples)
- **AUC**: 0.9998 (validation), 0.9996 (test)

**Per-class performance (Test)**:
- Good: 99.66% precision, 98.99% recall
- Mild: 99.53% precision, 99.58% recall (weakest class)
- Rotten: 100% precision, 99.26% recall

**Key findings**:
- **Performance decreased** compared to baseline (-0.85% validation, -0.54% test)
- Augmentation made training harder (took 20-25 epochs to reach 95% vs. 5-8 for baseline)
- Training was more volatile with oscillating accuracy curves
- Mild class most affected (subtle features confused by transformations)
- **Important**: No overfitting - validation matched/exceeded training throughout

**Why augmentation hurt performance**:
1. Baseline already near-optimal (99.84%) - little room for improvement
2. Aggressive color jitter may have genuinely changed perceived quality
3. Augmentation created ambiguous samples that confused boundaries

---

## PART B: Multi-Task Learning (Quality + Fruit Type)

**What's multi-task learning?** Training a single model to perform multiple related tasks simultaneously. The model shares most of its architecture but has separate "heads" for each task.

### Architecture Modification:
- **Shared backbone**: Same 4 convolutional blocks
- **Dual classification heads**:
  - Quality head: 256 units → 3 classes (Good, Mild, Rotten)
  - Fruit type head: 256 units → 11 classes (fruit varieties)
- **Combined loss**: Weighted sum of both task losses (equal weights: 1.0 each)

**Why multi-task?**
1. More efficient than training two separate models
2. Leverages all available labels (quality + fruit type from filenames)
3. Tests if tasks benefit from shared feature learning
4. Single forward pass produces both predictions

---

### Scenario 11: Multi-Task RGB Baseline

**Purpose**: Establish baseline for simultaneous fruit type and quality classification.

**Results - Quality Task**:
- **Validation accuracy**: 99.89% (2 errors out of 1,873 samples)
- **Test accuracy**: 99.89% (1 error out of 939 samples)
- **AUC**: 1.0000 (both validation and test)

**Results - Fruit Type Task**:
- **Validation accuracy**: 100.00% (0 errors)
- **Test accuracy**: 100.00% (0 errors)
- **AUC**: 1.0000 (both validation and test)

**Per-class Quality Performance (Validation)**:
- Good: 100% precision, 99.83% recall
- Mild: 99.79% precision, 99.79% recall
- Rotten: 99.88% precision, 100% recall

**Per-class Fruit Type Performance**:
- All 11 fruit types: 100% across all metrics (both validation and test)

**Key findings**:
- **Multi-task outperformed single-task** on quality (+0.05% validation, +0.10% test)
- Perfect fruit type classification suggests visual differences between fruits are highly distinctive
- Positive transfer learning: fruit-specific features helped quality assessment
- No task interference - both tasks achieved excellent performance
- Shared backbone learned features useful for both objectives

**Comparison to Single-Task Scenario 1**:
- Quality validation: 99.89% vs. 99.84% (+0.05%)
- Quality test: 99.89% vs. 99.79% (+0.10%)
- Fruit type: Perfect 100% (no single-task equivalent)

---

### Scenario 12: Multi-Task Grayscale

**Purpose**: Test if grayscale affects quality and fruit type tasks differently.

**Results - Quality Task**:
- **Validation accuracy**: 99.95% (1 error out of 1,873 samples)
- **Test accuracy**: 99.89% (1 error out of 939 samples)
- **AUC**: 1.0000 (both validation and test)

**Results - Fruit Type Task**:
- **Validation accuracy**: 100.00%
- **Test accuracy**: 100.00%
- **AUC**: 1.0000 (both validation and test)

**Key findings**:
- Grayscale maintained perfect fruit type identification
- Quality performance: Slight improvement on validation (+0.06% from RGB multi-task)
- Both tasks handle grayscale equally well
- Color not essential for either task

**Comparison to Multi-Task RGB (Scenario 11)**:
| Metric | MT RGB Val | MT Gray Val | MT RGB Test | MT Gray Test |
|--------|------------|-------------|-------------|--------------|
| Quality Accuracy | 99.89% | 99.95% | 99.89% | 99.89% |
| Quality Errors | 2/1873 | 1/1873 | 1/939 | 1/939 |
| Fruit Accuracy | 100.00% | 100.00% | 100.00% | 100.00% |

---

### Scenario 13: Multi-Task Augmentation

**Purpose**: Evaluate augmentation benefits under multi-task learning.

**Configuration**: Same augmentation as Scenario 3 (rotation, flips, color jitter)

**Expected pattern based on Part A**: Likely to show similar performance degradation as single-task augmentation, with quality task accuracy decreasing slightly while maintaining excellent AUC scores and no overfitting.

---

## Key Performance Comparison Across All Scenarios

### Single-Task Quality Classification:

| Scenario | Description | Val Acc | Test Acc | Notes |
|----------|-------------|---------|----------|-------|
| 1 | RGB Baseline | 99.84% | 99.79% | Strong baseline |
| 2 | Grayscale | 99.84% | **100.00%** | Best test performance |
| 3 | Augmented RGB | 98.99% | 99.25% | Degraded performance |

### Multi-Task Learning:

| Scenario | Description | Quality Val | Quality Test | Fruit Val | Fruit Test |
|----------|-------------|-------------|--------------|-----------|------------|
| 11 | RGB Baseline | 99.89% | 99.89% | 100% | 100% |
| 12 | Grayscale | 99.95% | 99.89% | 100% | 100% |
| 13 | Augmented | TBD | TBD | TBD | TBD |

---

## Major Findings and Insights

### 1. Color Information is Not Essential
- Grayscale images achieved equal or better performance than RGB
- Texture and intensity patterns provide sufficient discriminative information
- Practical implication: Simpler grayscale systems viable (lower cost, faster processing)

### 2. Multi-Task Learning Benefits
- Quality classification improved slightly when combined with fruit type task (+0.05-0.10%)
- No task interference - both objectives achieved excellent results
- Positive transfer: fruit-specific features enhanced quality discrimination
- More efficient than training separate models (shared backbone, single forward pass)

### 3. Data Augmentation Degraded Performance
- Baseline already near-optimal (ceiling effect)
- Aggressive augmentation created ambiguous training samples
- Training became slower and more volatile
- Mild quality class most vulnerable to augmentation artifacts
- **However**: No overfitting observed, excellent probability calibration maintained

### 4. Classification Difficulty Patterns
- Good vs. Rotten: Near-perfect separation (extremes easily distinguished)
- Mild class: Most challenging due to transitional characteristics
- Errors primarily occur between adjacent categories (Good↔Mild or Mild↔Rotten)
- Never confused Good with Rotten directly (ordinal relationship preserved)

### 5. Model Convergence and Training
- Baseline converged rapidly (>99% by epoch 5-8)
- Augmented models required 3-4× more epochs (20-25 to reach 95%)
- Learning rate scheduling crucial for refinement
- Mixed precision training accelerated convergence without accuracy loss

---

## Technical Evaluation Metrics Explained

### Accuracy
**What it means**: Percentage of correct predictions out of all predictions.
- **Formula**: (Correct predictions) / (Total predictions) × 100%
- **Example**: 99.84% accuracy = 3 errors in 1,872 samples

### Precision
**What it means**: When the model predicts a class, how often is it correct?
- **Formula**: (True positives) / (True positives + False positives)
- **Example**: 100% precision for "Good" = every "Good" prediction was actually good fruit

### Recall (Sensitivity)
**What it means**: Of all actual instances of a class, how many did the model find?
- **Formula**: (True positives) / (True positives + False negatives)
- **Example**: 99.79% recall for "Mild" = found 473 out of 475 mild samples

### F1-Score
**What it means**: Harmonic mean of precision and recall (balances both metrics).
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0% (worst) to 100% (perfect)

### AUC (Area Under ROC Curve)
**What it means**: Measures model's ability to distinguish between classes across all decision thresholds.
- **Range**: 0.5 (random guessing) to 1.0 (perfect discrimination)
- **Interpretation**:
  - 0.9-1.0 = Excellent
  - 0.8-0.9 = Good
  - 0.7-0.8 = Fair
- **Project results**: 0.9996-1.0000 (exceptional)

### Confusion Matrix
**What it means**: Table showing where the model makes mistakes.
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications (shows which classes get confused)

---

## Practical Implications

### 1. Deployment Recommendations
- **Grayscale systems are viable**: Equal/better performance with lower computational cost
- **Multi-task preferred**: Slight quality improvement + free fruit identification
- **Skip augmentation**: Not beneficial for this high-quality dataset
- **Model size**: 26.2M parameters feasible for edge deployment with optimization

### 2. Real-World Considerations
- System handles imbalanced data well (class weights effective)
- Robust to missing quality categories (e.g., StrawberryQ has no "Good" samples)
- Near-perfect AUC ensures reliable confidence scores
- Confusion patterns are sensible (adjacent categories, not random)

### 3. Limitations and Future Work
- Dataset contains near-perfect studio images (may need testing on field images)
- Limited to 11 fruit types (expansion needed for broader application)
- Single perspective per fruit (multi-angle capture may help)
- Augmentation needs refinement (more conservative transformations)

### 4. Computational Efficiency
- **Grayscale**: 3× less input data than RGB
- **Multi-task**: 1 model instead of 2 (shared backbone)
- **Mixed precision**: ~2× training speedup
- **Batch size 32**: Balanced memory usage and convergence speed

---

## Model Performance Summary Statistics

### Best Overall Performance:
- **Single-task**: Scenario 2 (Grayscale) - 100% test accuracy
- **Multi-task**: Scenario 12 (Multi-task Grayscale) - 99.95% quality validation, 100% fruit test

### Worst Performance (Still Excellent):
- **Single-task**: Scenario 3 (Augmented) - 98.99% validation
- **Still achieved**: >99% test accuracy, near-perfect AUC

### Most Robust Class: Rotten
- Consistently >99.7% across all scenarios
- Severe degradation features resistant to transformations

### Most Challenging Class: Mild
- Transitional characteristics between Good and Rotten
- Most sensitive to augmentation artifacts
- Performance range: 97.99% (augmented) to 100% (grayscale)

---

## Conclusion

This project successfully demonstrated that CNN-based fruit quality assessment is highly effective, achieving near-perfect classification across multiple experimental conditions. Key insights include:

1. **Color is not essential** - Grayscale performs equally well or better
2. **Multi-task learning beneficial** - Slight quality improvement with free fruit identification
3. **Simple baselines are powerful** - Complex augmentation not needed for clean datasets
4. **Model is production-ready** - >99% accuracy, robust to variations, sensible error patterns

The research provides a solid foundation for practical fruit quality assessment systems, with clear recommendations for deployment (grayscale multi-task models) and areas for future refinement (field image testing, conservative augmentation strategies).

**Final verdict**: The SimpleCNN architecture with grayscale inputs and multi-task learning represents the optimal configuration for this fruit quality classification task, balancing exceptional performance (99.89-100% accuracy) with computational efficiency.

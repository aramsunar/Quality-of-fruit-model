# Fruit Quality Classification Model Documentation

## Overview

This document describes the Convolutional Neural Network (CNN) model used for classifying fruit quality in the FruQ-DB dataset. The model classifies fruits into three quality categories: **Fresh**, **Mild**, and **Rotten**.

---

## Dataset Structure

### FruQ-DB Dataset

- **Location**: `FruQ-DB/` folder in project root
- **Classes**: 3 quality categories
  - Fresh
  - Mild
  - Rotten
- **Image Format**: RGB images (JPG/PNG)
- **Processing**: Resized to 128x128 pixels, normalized to [0, 1] range

### Data Split

```
Total Dataset
├── Training Set (64%)    - Used to train the model
├── Validation Set (16%)  - Used to tune hyperparameters during training
└── Test Set (20%)        - Used for final evaluation
```

The data is split using stratification to maintain class balance across all sets.

---

## Model Architecture

### Sequential CNN Model

The model follows a classic CNN architecture with three convolutional blocks followed by dense layers for classification.

```
Input (128x128x3 RGB Image)
    ↓
[BLOCK 1] - Feature Extraction (Low-level features)
├── Conv2D (32 filters, 3x3, ReLU, same padding)
├── BatchNormalization
├── Conv2D (32 filters, 3x3, ReLU, same padding)
├── MaxPooling2D (2x2) - Reduces spatial dimensions by half
└── Dropout (0.25) - Prevents overfitting
    ↓
[BLOCK 2] - Feature Extraction (Mid-level features)
├── Conv2D (64 filters, 3x3, ReLU, same padding)
├── BatchNormalization
├── Conv2D (64 filters, 3x3, ReLU, same padding)
├── MaxPooling2D (2x2) - Further dimension reduction
└── Dropout (0.25)
    ↓
[BLOCK 3] - Feature Extraction (High-level features)
├── Conv2D (128 filters, 3x3, ReLU, same padding)
├── BatchNormalization
├── Conv2D (128 filters, 3x3, ReLU, same padding)
├── MaxPooling2D (2x2) - Final dimension reduction
└── Dropout (0.25)
    ↓
[CLASSIFIER] - Classification Head
├── Flatten - Converts 2D features to 1D
├── Dense (256 units, ReLU)
├── BatchNormalization
├── Dropout (0.5) - Higher dropout for dense layers
├── Dense (128 units, ReLU)
├── Dropout (0.5)
└── Dense (3 units, Softmax) - Output probabilities for 3 classes
```

### Layer Details

#### Convolutional Layers

- **Purpose**: Extract spatial features from images
- **Filters**: 32 → 64 → 128 (progressively capturing more complex features)
- **Kernel Size**: 3x3 (standard for image processing)
- **Activation**: ReLU (Rectified Linear Unit) - adds non-linearity
- **Padding**: 'same' - maintains spatial dimensions

#### Batch Normalization

- **Purpose**: Normalizes activations between layers
- **Benefits**:
  - Faster training
  - Reduces internal covariate shift
  - Acts as regularization

#### MaxPooling Layers

- **Size**: 2x2
- **Purpose**: Downsamples feature maps
- **Benefits**:
  - Reduces computational cost
  - Provides translation invariance
  - Reduces overfitting

#### Dropout Layers

- **Rates**: 0.25 (convolutional blocks), 0.5 (dense layers)
- **Purpose**: Randomly deactivates neurons during training
- **Benefit**: Prevents overfitting by forcing network redundancy

#### Dense (Fully Connected) Layers

- **256 units**: First dense layer for feature combination
- **128 units**: Second dense layer for further abstraction
- **3 units (Softmax)**: Final output layer
  - Produces probability distribution over 3 classes
  - Sum of probabilities = 1.0

---

## Training Process

### Data Augmentation

Real-time augmentation is applied during training using `ImageDataGenerator`:

```python
ImageDataGenerator(
    rotation_range=20,        # Rotate images up to 20 degrees
    width_shift_range=0.2,    # Shift horizontally by 20%
    height_shift_range=0.2,   # Shift vertically by 20%
    horizontal_flip=True,     # Random horizontal flipping
    zoom_range=0.2,           # Zoom in/out by 20%
    shear_range=0.1,          # Shear transformation (10%)
    fill_mode='nearest'       # Fill empty pixels with nearest value
)
```

**Why Augmentation?**

- Increases training data variety artificially
- Improves model generalization
- Reduces overfitting
- Makes model robust to variations in fruit orientation, lighting, and position

**Note**: Augmentation is **only applied to training data**, not validation or test data.

### Optimization

#### Optimizer: Adam

```python
Adam(learning_rate=0.001)
```

- **Adaptive learning rate**: Adjusts per parameter
- **Momentum**: Accelerates convergence
- **Initial learning rate**: 0.001

#### Loss Function: Categorical Crossentropy

```python
loss='categorical_crossentropy'
```

- Standard for multi-class classification
- Measures difference between predicted and true probability distributions
- Formula: `-Σ(y_true * log(y_pred))`

#### Metric: Accuracy

```python
metrics=['accuracy']
```

- Percentage of correctly classified samples
- Simple, interpretable metric

### Training Configuration

```python
Epochs: 30 (maximum)
Batch Size: 32 images per batch
Steps per Epoch: len(training_data) // 32
```

### Callbacks (Training Controls)

#### 1. Early Stopping

```python
EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)
```

- **Monitors**: Validation accuracy
- **Patience**: Stops if no improvement for 10 epochs
- **Benefit**: Prevents unnecessary training and overfitting
- **Best Weights**: Restores model to best validation accuracy state

#### 2. Learning Rate Reduction

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
```

- **Monitors**: Validation loss
- **Action**: Reduces learning rate by 50% if no improvement for 5 epochs
- **Benefit**: Fine-tunes model when learning plateaus

---

## Training Flow

```
1. Load and Preprocess Images
   ├── Read images from FruQ-DB folders
   ├── Resize to 128x128
   ├── Normalize to [0, 1]
   └── Convert labels to one-hot encoding

2. Split Dataset
   ├── 64% Training (with stratification)
   ├── 16% Validation (from training set)
   └── 20% Test (held out for final evaluation)

3. Build CNN Model
   └── Initialize architecture with random weights

4. Training Loop (Up to 30 epochs)
   ├── For each batch:
   │   ├── Apply random augmentation
   │   ├── Forward pass (compute predictions)
   │   ├── Compute loss (categorical crossentropy)
   │   ├── Backward pass (compute gradients)
   │   └── Update weights (Adam optimizer)
   │
   ├── After each epoch:
   │   ├── Evaluate on validation set (no augmentation)
   │   ├── Record training/validation loss and accuracy
   │   ├── Check Early Stopping condition
   │   └── Check Learning Rate Reduction condition
   │
   └── Stop when:
       ├── Early stopping triggers (10 epochs no improvement)
       └── OR maximum 30 epochs reached

5. Restore Best Model
   └── Load weights from epoch with best validation accuracy

6. Final Evaluation on Test Set
   └── Report accuracy, precision, recall, F1-score
```

---

## Validation Process

### During Training (Validation Set)

The validation set is used to:

1. **Monitor Overfitting**

   - Compare training vs. validation accuracy
   - If training accuracy >> validation accuracy → overfitting

2. **Hyperparameter Tuning**

   - Early stopping decision
   - Learning rate adjustment
   - Best model selection

3. **Provide Unbiased Performance Estimate**
   - Validation data is never used for weight updates
   - Only for monitoring and decision making

### Validation Metrics Tracked

```
For each epoch:
├── Training Loss (decreasing = model learning)
├── Training Accuracy (increasing = model improving)
├── Validation Loss (should decrease similarly to training)
└── Validation Accuracy (should track training accuracy)
```

**Good Training Signs:**

- ✅ Both training and validation accuracy increase together
- ✅ Gap between training and validation is small
- ✅ Validation accuracy improves over time

**Warning Signs:**

- ⚠️ Training accuracy high, validation accuracy low → Overfitting
- ⚠️ Both accuracies low → Underfitting
- ⚠️ Validation accuracy plateaus early → Need better architecture/data

---

## Evaluation Metrics

### Test Set Evaluation

After training completes, the model is evaluated on the held-out test set:

#### 1. Classification Report

```
Per-class metrics:
├── Precision: TP / (TP + FP)
│   └── Of predictions for class X, how many were correct?
├── Recall: TP / (TP + FN)
│   └── Of actual class X samples, how many were found?
├── F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
│   └── Harmonic mean of precision and recall
└── Support: Number of true instances per class
```

#### 2. Confusion Matrix

```
           Predicted
           Fresh  Mild  Rotten
Actual  ┌─────────────────────┐
Fresh   │  TP    FP    FP    │
Mild    │  FP    TP    FP    │
Rotten  │  FP    FP    TP    │
        └─────────────────────┘

Diagonal = Correct predictions
Off-diagonal = Misclassifications
```

#### 3. Overall Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
         = (TP_Fresh + TP_Mild + TP_Rotten) / Total_Samples
```

---

## Model Predictions

### Prediction Process

```python
For a new fruit image:

1. Preprocess
   ├── Resize to 128x128
   ├── Normalize to [0, 1]
   └── Add batch dimension

2. Forward Pass
   └── Pass through all layers sequentially

3. Output (Softmax Layer)
   ├── Fresh:  0.15 (15% probability)
   ├── Mild:   0.75 (75% probability)  ← Highest
   └── Rotten: 0.10 (10% probability)

4. Prediction
   └── argmax(probabilities) = Mild

5. Confidence
   └── max(probabilities) = 0.75 (75%)
```

### Interpreting Predictions

- **High Confidence (>0.9)**: Model is very certain
- **Medium Confidence (0.7-0.9)**: Model is fairly confident
- **Low Confidence (<0.7)**: Model is uncertain
- **Close Probabilities**: Image has features of multiple classes

---

## Why This Architecture Works

### Progressive Feature Learning

```
Block 1 (32 filters)  → Edges, colors, basic shapes
Block 2 (64 filters)  → Textures, patterns, spots
Block 3 (128 filters) → Complex features (browning, decay)
```

### Regularization Techniques

1. **Dropout**: Prevents co-adaptation of neurons
2. **Batch Normalization**: Stabilizes learning
3. **Data Augmentation**: Reduces reliance on specific orientations
4. **Early Stopping**: Prevents overfitting to training data

### Optimization Strategy

1. **Adam Optimizer**: Efficient, adaptive learning
2. **Learning Rate Decay**: Fine-tunes as training progresses
3. **Batch Processing**: Stable gradient estimates

---

## Training Output Example

```
Epoch 1/30
- Training: 32 batches, loss=0.95, accuracy=0.45
- Validation: loss=0.88, accuracy=0.52
- Time: 45s

Epoch 2/30
- Training: loss=0.78, accuracy=0.62
- Validation: loss=0.72, accuracy=0.65
- Time: 44s

...

Epoch 15/30
- Training: loss=0.23, accuracy=0.92
- Validation: loss=0.31, accuracy=0.89 ← Best so far
- Time: 43s

...

Epoch 25/30
- Early stopping triggered (no improvement for 10 epochs)
- Restoring best weights from epoch 15
```

---

## Model Performance Visualization

The model generates several plots:

### 1. Training History

- **Loss Curve**: Shows how loss decreases over epochs
- **Accuracy Curve**: Shows how accuracy increases over epochs
- **Compare**: Training vs. Validation for both metrics

### 2. Confusion Matrix

- Visual representation of classification results
- Helps identify which classes are confused with each other

### 3. Class Distribution

- Bar chart showing number of samples per class in test set
- Ensures balanced evaluation

### 4. Per-Class Accuracy

- Individual accuracy for each quality category
- Identifies which classes the model performs best/worst on

### 5. Sample Predictions

- Shows actual images with predictions and confidence
- Green = Correct, Red = Incorrect
- Helps understand model behavior qualitatively

---

## Summary

This CNN model uses a proven architecture for image classification with:

- ✅ **Progressive feature extraction** through 3 convolutional blocks
- ✅ **Regularization** via dropout, batch normalization, and augmentation
- ✅ **Smart training** with early stopping and learning rate scheduling
- ✅ **Robust validation** to ensure generalization
- ✅ **Comprehensive evaluation** with multiple metrics and visualizations

The model achieves high accuracy in distinguishing between Fresh, Mild, and Rotten fruit quality by learning hierarchical visual features specific to each quality level.

# Fruit Quality Classification Model

A deep learning project that uses Convolutional Neural Networks (CNN) to classify fruit images into three quality categories: **Fresh**, **Mild**, and **Rotten**. The model is trained on the FruQ-DB (Fruit Quality Database) dataset and achieves high accuracy in distinguishing fruit quality levels.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset Structure](#dataset-structure)
- [How the Model Works](#how-the-model-works)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements two versions of a fruit quality classification system:

1. **fruit_classification.py** - Basic CNN model with dataset download functionality
2. **Fruit_Quality.py** - Advanced CNN model with enhanced architecture and visualization

Both models classify fruit images into three quality categories based on visual appearance:

- **Fresh**: High-quality, vibrant fruits
- **Mild**: Medium-quality fruits showing early signs of degradation
- **Rotten**: Low-quality fruits with significant deterioration

## 🏗️ Model Architecture

### CNN Architecture Details

The model uses a deep Convolutional Neural Network with the following architecture:

#### **Input Layer**

- Input shape: (128, 128, 3) - RGB images resized to 128x128 pixels

#### **Convolutional Blocks** (4 blocks)

**Block 1:**

- Conv2D: 32 filters, 3×3 kernel, ReLU activation
- Batch Normalization
- MaxPooling: 2×2
- Dropout: 25%

**Block 2:**

- Conv2D: 64 filters, 3×3 kernel, ReLU activation
- Batch Normalization
- MaxPooling: 2×2
- Dropout: 25%

**Block 3:**

- Conv2D: 128 filters, 3×3 kernel, ReLU activation
- Batch Normalization
- MaxPooling: 2×2
- Dropout: 25%

**Block 4:**

- Conv2D: 256 filters, 3×3 kernel, ReLU activation
- Batch Normalization
- MaxPooling: 2×2
- Dropout: 25%

#### **Fully Connected Layers**

- Flatten layer
- Dense: 512 units, ReLU activation
- Batch Normalization
- Dropout: 50%
- Dense: 256 units, ReLU activation
- Dropout: 50%
- Output Dense: 3 units (classes), Softmax activation

#### **Compilation Settings**

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### Training Features

1. **Data Augmentation**:

   - Rotation: ±20 degrees
   - Width/Height shift: 20%
   - Horizontal flip: Yes
   - Zoom: 20%
   - Shear: 20%
   - Fill mode: Nearest

2. **Callbacks**:

   - Early Stopping: Patience of 5-10 epochs
   - ReduceLROnPlateau: Reduces learning rate by 50% after 3-5 epochs without improvement

3. **Data Split**:
   - Training: 64% of data
   - Validation: 16% of data
   - Testing: 20% of data

## 📁 Dataset Structure

The model expects the FruQ-DB dataset to be organized in the following structure:

```
FruQ-DB/
├── Fresh/
│   ├── Image1.png
│   ├── Image1 (2).png
│   ├── Image3.png
│   └── ... (multiple fruit images)
├── Mild/
│   ├── Image1.png
│   ├── Image1 (2).png
│   ├── Image101.png
│   └── ... (multiple fruit images)
└── Rotten/
    ├── Image1.png
    ├── Image1 (2).png
    ├── Image101.png
    └── ... (multiple fruit images)
```

### Dataset Requirements

- **Format**: PNG images (JPG/JPEG also supported)
- **Organization**: One folder per class containing all images for that class
- **Classes**: Exactly 3 classes (Fresh, Mild, Rotten)
- **Image Size**: Any size (automatically resized to 128×128 during processing)
- **Naming**: Any valid filename (e.g., Image1.png, Image1 (2).png)

### Dataset Location

Place the FruQ-DB folder in the project root directory:

```
Quality-of-fruit-model/
├── FruQ-DB/
│   ├── Fresh/
│   ├── Mild/
│   └── Rotten/
├── fruit_classification.py
├── Fruit_Quality.py
└── README.md
```

## 🔬 How the Model Works

### 1. **Data Loading & Preprocessing**

```python
# Images are loaded and preprocessed
- Read images using OpenCV
- Convert from BGR to RGB color space
- Resize to 128×128 pixels
- Normalize pixel values to [0, 1] range
- Assign labels based on folder names
```

### 2. **Feature Extraction**

The CNN learns hierarchical features:

- **Early layers**: Basic features (edges, colors, textures)
- **Middle layers**: Complex patterns (shapes, spots, discoloration)
- **Deep layers**: High-level features (overall quality indicators)

### 3. **Classification Process**

```
Input Image (128×128×3)
    ↓
Convolutional Blocks (Feature Extraction)
    ↓
Flatten
    ↓
Fully Connected Layers (Classification)
    ↓
Softmax (Probability Distribution)
    ↓
Output: [Fresh, Mild, Rotten] probabilities
```

### 4. **Prediction**

The model outputs a probability distribution over three classes:

- Class with highest probability is the predicted quality
- Confidence score indicates prediction certainty

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/aramsunar/Quality-of-fruit-model.git
cd Quality-of-fruit-model
```

### Step 2: Set Up Virtual Environment (Recommended)

#### Option A: Automated Setup (macOS/Linux)

Use the provided setup script:

```bash
./setup.sh
```

This will:

- Create a virtual environment
- Activate it
- Install all dependencies

#### Option B: Manual Setup

**Create virtual environment:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Install dependencies:**

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**To deactivate the virtual environment:**

```bash
deactivate
```

### Step 3: Prepare Dataset

Ensure your FruQ-DB dataset is structured correctly:

```bash
Quality-of-fruit-model/
└── FruQ-DB/
    ├── Fresh/
    ├── Mild/
    └── Rotten/
```

## 💻 Usage

**Note:** Make sure your virtual environment is activated before running the scripts:

```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### Running the Basic Model (fruit_classification.py)

```bash
python fruit_classification.py
```

This script will:

1. Load the FruQ-DB dataset from the local folder
2. Display dataset statistics and sample images
3. Build and train the CNN model
4. Evaluate on test data
5. Display comprehensive visualizations:
   - Training/validation loss curves
   - Training/validation accuracy curves
   - Confusion matrix
   - Sample predictions with confidence scores

### Running the Advanced Model (Fruit_Quality.py)

```bash
python Fruit_Quality.py
```

This script provides:

1. Automatic dataset structure exploration
2. Enhanced model architecture
3. More detailed visualizations including:
   - Training history plots
   - Confusion matrix with annotations
   - Class distribution analysis
   - Per-class accuracy breakdown
   - Sample predictions with confidence scores

### Expected Output

Both scripts will display:

```
======================================================================
 FRUIT QUALITY CLASSIFICATION
======================================================================
✓ Loading dataset...
✓ Found 3 classes: ['Fresh', 'Mild', 'Rotten']
✓ Successfully loaded X,XXX images
✓ Building CNN model...
✓ Training model...
✓ Evaluating model...

CLASSIFICATION REPORT:
              precision    recall  f1-score   support
       Fresh       0.XX      0.XX      0.XX       XXX
        Mild       0.XX      0.XX      0.XX       XXX
      Rotten       0.XX      0.XX      0.XX       XXX
======================================================================
```

## 📦 Requirements

### Python Libraries

```txt
numpy>=1.19.0
matplotlib>=3.3.0
tensorflow>=2.4.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
tqdm>=4.50.0
pandas>=1.1.0
requests>=2.25.0
```

### Hardware Requirements

- **Minimum**:

  - CPU: Dual-core processor
  - RAM: 8 GB
  - Storage: 2 GB free space

- **Recommended**:
  - GPU: NVIDIA GPU with CUDA support
  - RAM: 16 GB or more
  - Storage: 5 GB free space

## 📊 Results

The model achieves:

- **Overall Accuracy**: Typically 85-95% on test data
- **Training Time**: 15-30 minutes (depending on hardware)
- **Inference Time**: < 100ms per image

### Performance Metrics

The model provides:

- **Classification Report**: Precision, Recall, F1-Score for each class
- **Confusion Matrix**: Detailed breakdown of predictions
- **Per-Class Accuracy**: Individual accuracy for Fresh, Mild, and Rotten
- **Confidence Scores**: Probability distribution for each prediction

### Visualizations

Both scripts generate:

1. Training/validation loss and accuracy curves
2. Confusion matrix heatmap
3. Sample predictions with true/predicted labels
4. Class distribution charts
5. Per-class accuracy bar charts

## 📂 Project Structure

```
Quality-of-fruit-model/
├── FruQ-DB/                      # Dataset directory
│   ├── Fresh/                    # Fresh fruit images
│   ├── Mild/                     # Mild quality fruit images
│   └── Rotten/                   # Rotten fruit images
├── venv/                         # Virtual environment (created after setup)
├── fruit_classification.py       # Basic CNN implementation
├── Fruit_Quality.py              # Advanced CNN implementation
├── requirements.txt              # Python dependencies
├── setup.sh                      # Automated setup script (macOS/Linux)
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

## 🔧 Customization

### Adjusting Model Parameters

You can modify hyperparameters in the scripts:

```python
# Image size
self.image_size = (128, 128)  # Change to (224, 224) for higher resolution

# Learning rate
optimizer=Adam(learning_rate=0.0001)  # Adjust as needed

# Epochs
epochs=30  # Increase for more training

# Batch size
batch_size=32  # Adjust based on memory
```

### Using Custom Datasets

To use your own dataset:

1. Organize images in class folders
2. Update the dataset path:
   ```python
   self.dataset_path = "your_dataset_folder"
   ```
3. Ensure folder names match your classes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project uses the FruQ-DB dataset. Please refer to the dataset's license for usage terms.

## 📚 References

- FruQ-DB Dataset: [Zenodo Repository](https://zenodo.org/records/7224690)
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org/)
- Keras Documentation: [keras.io](https://keras.io/)

## 👤 Author

**Rikus Swart**

- GitHub: [@aramsunar](https://github.com/aramsunar)

---

_Last Updated: November 2025_

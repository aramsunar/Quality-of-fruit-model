import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import cv2
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

print("=" * 70)
print(" FRUIT QUALITY CLASSIFICATION - FruQ-DB DATASET")
print("=" * 70)

class FruitQualityClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        self.image_size = (128, 128)
        self.dataset_path = "fruq_dataset"
    
    def explore_dataset_structure(self):
        """Explore the actual structure of FruQ-DB dataset"""
        print(" Exploring dataset structure...")
        
        base_path = Path(self.dataset_path)
        
        # Look for the main dataset folder
        possible_paths = [
            base_path / "FruQ-DB",
            base_path / "FruQ-DB-main",
            base_path / "dataset",
            base_path
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                print(f" Found dataset at: {path}")
                break
        
        if dataset_path is None:
            print(" Dataset not found. Please ensure FruQ-DB is extracted in 'fruq_dataset' folder")
            return None
        
        # Explore the structure
        all_items = list(dataset_path.rglob("*"))
        folders = [item for item in all_items if item.is_dir()]
        files = [item for item in all_items if item.is_file()]
        
        print(f" Total folders: {len(folders)}")
        print(f" Total files: {len(files)}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in files if f.suffix.lower() in image_extensions]
        
        print(f" Image files found: {len(image_files)}")
        
        # Group by parent folder to understand class structure
        class_images = {}
        for img_path in image_files:
            class_name = img_path.parent.name
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(img_path)
        
        print("\n Dataset structure:")
        for class_name, images in class_images.items():
            print(f" {class_name}: {len(images)} images")
        
        return dataset_path, class_images
    
    def load_images_from_structure(self, class_images):
        """Load images based on the discovered structure"""
        print("\n Loading images...")
        
        images = []
        labels = []
        
        # Create proper class mapping based on folder names
        class_mapping = {}
        current_class_id = 0
        
        # Map each unique folder name to a unique class ID
        for class_folder in sorted(class_images.keys()):
            if class_folder not in class_mapping:
                class_mapping[class_folder] = current_class_id
                current_class_id += 1
        
        # Update class names
        self.class_names = list(class_mapping.keys())
        self.num_classes = len(self.class_names)
        
        print(f" Class mapping:")
        for class_name, class_id in class_mapping.items():
            print(f"   {class_name} -> Class {class_id}")
        
        # Load images with proper class mapping
        for class_folder, image_paths in class_images.items():
            class_id = class_mapping[class_folder]
            
            print(f" Loading {len(image_paths)} images from {class_folder} -> Class {class_id}")
            
            for img_path in tqdm(image_paths):
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(class_id)
                    
                except Exception as e:
                    continue
        
        print(f"\n Successfully loaded {len(images)} images")
        print(f" Number of classes: {self.num_classes}")
        print(f" Classes: {self.class_names}")
        
        return np.array(images), np.array(labels)
    
    def load_real_dataset(self):
        """Load the actual FruQ-DB dataset"""
        print(" Loading FruQ-DB dataset...")
        
        # First explore the structure
        result = self.explore_dataset_structure()
        if result is None:
            return None, None
        
        dataset_path, class_images = result
        
        # Load images based on structure
        images, labels = self.load_images_from_structure(class_images)
        
        if len(images) == 0:
            print(" No images loaded! Using synthetic data...")
            return self.create_synthetic_dataset()
        
        return images, labels
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset as fallback"""
        print(" Creating synthetic fruit quality dataset...")
        
        n_samples = 1000
        images = []
        labels = []
        
        # Colors for different quality levels
        fresh_color = [0.2, 0.8, 0.2]    # Green
        mid_color = [0.8, 0.8, 0.2]      # Yellow
        rotten_color = [0.6, 0.2, 0.2]   # Brown-red
        
        colors = [fresh_color, mid_color, rotten_color]
        self.class_names = ['Fresh', 'Mid', 'Rotten']
        self.num_classes = 3
        
        for i in range(n_samples):
            class_id = i % self.num_classes
            img = self.create_quality_fruit_image(class_id, colors[class_id])
            images.append(img)
            labels.append(class_id)
        
        print(f" Created {len(images)} synthetic images")
        return np.array(images), np.array(labels)
    
    def create_quality_fruit_image(self, class_id, base_color):
        """Create synthetic fruit images with quality variations"""
        img = np.zeros((self.image_size[0], self.image_size[1], 3))
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        
        # Create fruit shape with quality variations
        for x in range(self.image_size[0]):
            for y in range(self.image_size[1]):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if distance < 25:  # Fruit body
                    # Add quality-specific variations
                    if class_id == 0:  # Fresh - smooth and vibrant
                        variation = np.random.normal(0, 0.05, 3)
                    elif class_id == 1:  # Mid - some spots
                        variation = np.random.normal(0, 0.1, 3)
                        if np.random.random() < 0.1:  # Add spots
                            variation += [0.3, 0.3, 0.1]
                    else:  # Rotten - more variation and spots
                        variation = np.random.normal(0, 0.15, 3)
                        if np.random.random() < 0.2:  # More spots
                            variation += [0.4, 0.2, 0.1]
                    
                    img[y, x] = np.clip(base_color + variation, 0, 1)
        
        return img
    
    def build_improved_model(self):
        """Build a better CNN model for fruit quality classification"""
        print(" Building improved CNN model...")
        
        self.model = Sequential([
            # First block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Classifier
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(" Model built successfully!")
        print(f" Model configured for {self.num_classes} classes: {self.class_names}")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30):
        """Train the model with proper validation"""
        print(" Training model...")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ],
            verbose=1
        )
        
        print(" Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print(" Evaluating model...")
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print(" CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print(f"\n OVERALL TEST ACCURACY: {accuracy:.4f}")
        
        return accuracy, y_true, y_pred, y_pred_proba
    
    def plot_enhanced_results(self, X_test, y_true, y_pred, y_pred_proba, accuracy):
        """Create enhanced visualizations"""
        print(" Creating enhanced visualizations...")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Training history
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(y_true, y_pred)
        im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(self.num_classes), self.class_names, rotation=45)
        plt.yticks(range(self.num_classes), self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot 3: Class distribution
        plt.subplot(2, 3, 4)
        unique, counts = np.unique(y_true, return_counts=True)
        plt.bar(self.class_names, counts, color=['green', 'orange', 'red'])
        plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Plot 4: Accuracy by class
        plt.subplot(2, 3, 5)
        class_accuracy = []
        for i in range(self.num_classes):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
                class_accuracy.append(class_acc)
            else:
                class_accuracy.append(0)
        
        plt.bar(self.class_names, class_accuracy, color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Accuracy by Class', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add values on bars
        for i, v in enumerate(class_accuracy):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Plot sample predictions separately
        self.plot_sample_predictions(X_test, y_true, y_pred, y_pred_proba, accuracy)
    
    def plot_sample_predictions(self, X_test, y_true, y_pred, y_pred_proba, accuracy):
        """Plot sample predictions with confidence"""
        print(" Displaying sample predictions...")
        
        # Select random samples (try to get at least 2 from each class)
        indices = []
        for class_id in range(self.num_classes):
            class_indices = np.where(y_true == class_id)[0]
            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, min(4, len(class_indices)), replace=False)
                indices.extend(selected)
        
        # If we don't have enough, add more random ones
        if len(indices) < 12:
            remaining = 12 - len(indices)
            additional_indices = np.random.choice(len(X_test), remaining, replace=False)
            indices.extend(additional_indices)
        
        plt.figure(figsize=(20, 15))
        
        for i, idx in enumerate(indices):
            plt.subplot(3, 4, i + 1)
            plt.imshow(X_test[idx])
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx])
            
            if y_true[idx] == y_pred[idx]:
                color = 'green'
                result = "CORRECT"
            else:
                color = 'red'
                result = "WRONG"
            
            plt.title(f'True: {true_label}\nPred: {pred_label}\n{result}\nConf: {confidence:.2f}', 
                     color=color, fontsize=10, fontweight='bold')
            plt.axis('off')
        
        plt.suptitle(f'Sample Predictions - Overall Accuracy: {accuracy:.2%}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete classification pipeline"""
        print(" Starting Fruit Quality Classification Pipeline...")
        
        # Load dataset
        images, labels = self.load_real_dataset()
        
        if images is None or len(images) == 0:
            print(" Failed to load dataset. Using synthetic data...")
            images, labels = self.create_synthetic_dataset()
        
        # Prepare data
        X = images
        y = to_categorical(labels, self.num_classes)
        
        print(f"\n Data shapes:")
        print(f" X: {X.shape}")
        print(f" y: {y.shape}")
        print(f" Labels range: {np.min(labels)} to {np.max(labels)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, 
            stratify=np.argmax(y_train, axis=1)
        )
        
        print(f"\n Data Summary:")
        print(f" Training set: {X_train.shape[0]} images")
        print(f" Validation set: {X_val.shape[0]} images")
        print(f" Test set: {X_test.shape[0]} images")
        print(f" Number of classes: {self.num_classes}")
        print(f" Classes: {self.class_names}")
        
        # Build and train model
        self.build_improved_model()
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate
        accuracy, y_true, y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # Visualize results
        self.plot_enhanced_results(X_test, y_true, y_pred, y_pred_proba, accuracy)
        
        return accuracy

def main():
    """Main execution function"""
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        classifier = FruitQualityClassifier()
        accuracy = classifier.run_complete_pipeline()
        
        # Final summary
        print("\n" + "="*70)
        print("FRUIT QUALITY CLASSIFICATION COMPLETED!")
        print("="*70)
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Number of Classes: {classifier.num_classes}")
        print(f"Classes: {', '.join(classifier.class_names)}")
        print("\n All tasks completed:")
        print("   • Dataset exploration and loading ✓")
        print("   • Data preprocessing and augmentation ✓")
        print("   • CNN model training with early stopping ✓")
        print("   • Comprehensive model evaluation ✓")
        print("   • Detailed visualizations ✓")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
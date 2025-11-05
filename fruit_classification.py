import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import cv2
import zipfile
import requests
from tqdm import tqdm
import pandas as pd

print("=" * 70)
print(" FRUIT CLASSIFICATION WITH REAL DATASET")
print("=" * 70)

class FruitClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        self.image_size = (128, 128)  # Increased size for better quality
        self.dataset_path = "fruq_dataset"
    
    def download_and_extract_dataset(self):
        """Download and extract the FruQ-DB dataset"""
        print(" Downloading FruQ-DB dataset...")
        
        url = "https://zenodo.org/records/7224690/files/FruQ-DB.zip?download=1"
        zip_path = "FruQ-DB.zip"
        
        # Download the file
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            
            print(" Download completed!")
            
            # Extract the zip file
            print(" Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)
            
            print(" Extraction completed!")
            
            # Remove zip file to save space
            os.remove(zip_path)
            
        except Exception as e:
            print(f" Download failed: {e}")
            print(" Please download manually from: https://zenodo.org/records/7224690/files/FruQ-DB.zip?download=1")
            print(" And extract to 'fruq_dataset' folder")
            return False
        
        return True
    
    def load_real_dataset(self):
        """Load the actual FruQ-DB dataset"""
        print(" Loading real fruit dataset...")
        
        # Define the main dataset path
        main_path = os.path.join(self.dataset_path, "FruQ-DB")
        
        if not os.path.exists(main_path):
            print(" Dataset not found. Please check the extraction path.")
            return None, None
        
        images = []
        labels = []
        
        # Get all class folders
        class_folders = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]
        self.class_names = sorted(class_folders)
        self.num_classes = len(self.class_names)
        
        print(f" Found {self.num_classes} classes: {self.class_names}")
        
        # Load images from each class folder
        for class_id, class_name in enumerate(self.class_names):
            class_path = os.path.join(main_path, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f" Loading {len(image_files)} images from {class_name}...")
            
            for image_file in tqdm(image_files):
                try:
                    img_path = os.path.join(class_path, image_file)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img = img.astype('float32') / 255.0  # Normalize to [0,1]
                    
                    images.append(img)
                    labels.append(class_id)
                    
                except Exception as e:
                    print(f" Error loading {image_file}: {e}")
                    continue
        
        print(f" Successfully loaded {len(images)} images")
        return np.array(images), np.array(labels)
    
    def explore_dataset(self, images, labels):
        """Explore and display dataset statistics"""
        print("\n Dataset Statistics:")
        print(f" Total images: {len(images)}")
        print(f" Image shape: {images[0].shape}")
        print(f" Number of classes: {self.num_classes}")
        
        # Count images per class
        unique, counts = np.unique(labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            print(f" {self.class_names[class_id]}: {count} images")
        
        # Display sample images
        self.display_sample_images(images, labels)
    
    def display_sample_images(self, images, labels):
        """Display sample images from each class"""
        print("\n Displaying sample images from each class...")
        
        plt.figure(figsize=(15, 8))
        samples_per_class = 3
        
        for class_id in range(self.num_classes):
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) > 0:
                sample_indices = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
                
                for i, idx in enumerate(sample_indices):
                    plt.subplot(self.num_classes, samples_per_class, class_id * samples_per_class + i + 1)
                    plt.imshow(images[idx])
                    plt.title(f'{self.class_names[class_id]}')
                    plt.axis('off')
        
        plt.suptitle('Sample Images from FruQ-DB Dataset', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def build_improved_model(self):
        """Build an improved CNN model for real images"""
        print(" Building improved CNN model...")
        
        self.model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Classifier
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(" Improved model built successfully!")
        self.model.summary()
        return self.model
    
    def create_data_augmentation(self):
        """Create data augmentation for training"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model with data augmentation"""
        print(" Training model with data augmentation...")
        
        datagen = self.create_data_augmentation()
        
        # Calculate steps per epoch
        batch_size = 32
        steps_per_epoch = len(X_train) // batch_size
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=30,  # Increased epochs for better learning
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        print(" Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        print(" Evaluating model...")
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print(f" Test Accuracy: {accuracy:.4f}")
        
        return accuracy, y_true, y_pred, y_pred_proba
    
    def plot_results(self, X_test, y_true, y_pred, y_pred_proba, accuracy):
        """Create comprehensive visualizations"""
        print(" Creating visualizations...")
        
        # Plot 1: Training history
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Confusion Matrix
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks(range(self.num_classes), self.class_names, rotation=45)
        plt.yticks(range(self.num_classes), self.class_names)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, str(cm[i, j]), 
                        ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Sample predictions
        self.plot_sample_predictions(X_test, y_true, y_pred, y_pred_proba, accuracy)
    
    def plot_sample_predictions(self, X_test, y_true, y_pred, y_pred_proba, accuracy):
        """Plot sample predictions"""
        print(" Showing sample predictions...")
        
        indices = np.random.choice(len(X_test), 12, replace=False)
        
        plt.figure(figsize=(20, 15))
        
        for i, idx in enumerate(indices):
            plt.subplot(3, 4, i + 1)
            plt.imshow(X_test[idx])
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx])
            
            if y_true[idx] == y_pred[idx]:
                color = 'green'
                result = "✓ CORRECT"
            else:
                color = 'red'
                result = "✗ WRONG"
            
            plt.title(f'True: {true_label}\nPred: {pred_label}\n{result}\nConf: {confidence:.2f}', 
                     color=color, fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'Sample Predictions (Overall Accuracy: {accuracy:.2%})', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def run_complete_project(self):
        """Run the complete project with real dataset"""
        print(" Starting fruit classification with REAL dataset...")
        
        # Step 1: Download and load dataset
        if not os.path.exists(self.dataset_path):
            self.download_and_extract_dataset()
        
        images, labels = self.load_real_dataset()
        
        if images is None or len(images) == 0:
            print(" Failed to load dataset. Using synthetic data as fallback.")
            return self._fallback_synthetic()
        
        # Step 2: Explore dataset
        self.explore_dataset(images, labels)
        
        # Step 3: Prepare data
        X = images.astype('float32')
        y = to_categorical(labels, self.num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        print(f"\n Data splits:")
        print(f" Training: {X_train.shape[0]} images")
        print(f" Validation: {X_val.shape[0]} images") 
        print(f" Test: {X_test.shape[0]} images")
        
        # Step 4: Build and train improved model
        self.build_improved_model()
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Step 5: Evaluate
        accuracy, y_true, y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # Step 6: Visualize results
        self.plot_results(X_test, y_true, y_pred, y_pred_proba, accuracy)
        
        return accuracy
    
    def _fallback_synthetic(self):
        """Fallback to synthetic data if real dataset fails"""
        print(" Using synthetic data as fallback...")
        # You can keep your original synthetic data creation here
        # ... (your original synthetic data code)
        pass

def main():
    """Main function"""
    try:
        classifier = FruitClassifier()
        accuracy = classifier.run_complete_project()
        
        print("\n" + "="*70)
        print(" PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f" FINAL TEST ACCURACY: {accuracy:.4f}")
        print(f" NUMBER OF CLASSES: {classifier.num_classes}")
        print(f" CLASSES: {', '.join(classifier.class_names)}")
        print("\n All evaluation metrics completed:")
        print("   • Real dataset loading ✓")
        print("   • Data exploration ✓")
        print("   • Improved CNN model ✓")
        print("   • Data augmentation ✓")
        print("   • Accuracy ✓")
        print("   • Precision ✓") 
        print("   • Recall ✓")
        print("   • F1-Score ✓")
        print("   • Confusion Matrix ✓")
        print("   • Training History ✓")
        print("="*70)
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
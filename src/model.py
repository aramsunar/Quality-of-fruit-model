"""
CNN model architectures for fruit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN architecture for fruit classification."""

    def __init__(self, num_classes, input_channels=3, img_size=224):
        """
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            img_size (int): Input image size
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Calculate the size after convolutions and pooling
        self.feature_size = (img_size // 16) * (img_size // 16) * 256

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(-1, self.feature_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class DeepCNN(nn.Module):
    """Deeper CNN architecture with residual-like connections."""

    def __init__(self, num_classes, input_channels=3, img_size=224):
        """
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            img_size (int): Input image size
        """
        super(DeepCNN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Convolutional blocks
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 2)
        self.conv4 = self._make_layer(256, 512, 2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a layer with multiple convolutional blocks."""
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Convolutional blocks
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.dropout(x)
        x = self.fc(x)

        return x


class LightCNN(nn.Module):
    """Lightweight CNN for quick experimentation."""

    def __init__(self, num_classes, input_channels=3, img_size=224):
        """
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            img_size (int): Input image size
        """
        super(LightCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions and pooling
        self.feature_size = (img_size // 8) * (img_size // 8) * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.feature_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_model(model_name, num_classes, input_channels=3, img_size=224):
    """
    Get a model by name.

    Args:
        model_name (str): Name of the model ('simple', 'deep', 'light')
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels
        img_size (int): Input image size

    Returns:
        nn.Module: PyTorch model
    """
    models = {
        'simple': SimpleCNN,
        'deep': DeepCNN,
        'light': LightCNN
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

    return models[model_name](num_classes, input_channels, img_size)


def print_model_summary(model, input_size=(3, 224, 224)):
    """
    Print a summary of the model architecture.

    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size (C, H, W)
    """
    from torchinfo import summary
    try:
        summary(model, input_size=(1, *input_size))
    except:
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

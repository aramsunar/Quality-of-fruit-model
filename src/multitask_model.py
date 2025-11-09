"""
Multi-task CNN model architectures for fruit type and quality classification.

This module provides CNN architectures with dual prediction heads for
simultaneous fruit type and quality classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskSimpleCNN(nn.Module):
    """
    Multi-task Simple CNN architecture with shared backbone and dual heads.

    The model uses a shared convolutional backbone to extract features, then
    splits into two task-specific heads for quality and fruit type classification.
    """

    def __init__(self, num_quality_classes=3, num_fruit_classes=11,
                 input_channels=3, img_size=224):
        """
        Args:
            num_quality_classes (int): Number of quality classes (default: 3)
            num_fruit_classes (int): Number of fruit type classes (default: 11)
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            img_size (int): Input image size
        """
        super(MultiTaskSimpleCNN, self).__init__()

        self.num_quality_classes = num_quality_classes
        self.num_fruit_classes = num_fruit_classes

        # Shared convolutional backbone
        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Calculate the size after convolutions and pooling
        self.feature_size = (img_size // 16) * (img_size // 16) * 256

        # Shared feature layer
        self.shared_fc = nn.Linear(self.feature_size, 512)

        # Quality classification head
        self.quality_fc1 = nn.Linear(512, 256)
        self.quality_fc2 = nn.Linear(256, num_quality_classes)

        # Fruit type classification head
        self.fruit_fc1 = nn.Linear(512, 256)
        self.fruit_fc2 = nn.Linear(256, num_fruit_classes)

    def forward(self, x):
        """
        Forward pass with dual outputs.

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            tuple: (quality_logits, fruit_type_logits)
                - quality_logits: [batch_size, num_quality_classes]
                - fruit_type_logits: [batch_size, num_fruit_classes]
        """
        # Shared convolutional backbone
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

        # Shared feature extraction
        shared_features = F.relu(self.shared_fc(x))
        shared_features = self.dropout(shared_features)

        # Quality classification head
        quality_x = F.relu(self.quality_fc1(shared_features))
        quality_x = self.dropout(quality_x)
        quality_logits = self.quality_fc2(quality_x)

        # Fruit type classification head
        fruit_x = F.relu(self.fruit_fc1(shared_features))
        fruit_x = self.dropout(fruit_x)
        fruit_logits = self.fruit_fc2(fruit_x)

        return quality_logits, fruit_logits

    def get_shared_features(self, x):
        """
        Extract shared features without classification.

        Useful for visualization and analysis.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Shared feature representation
        """
        # Shared convolutional backbone
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(-1, self.feature_size)

        # Shared feature extraction
        shared_features = F.relu(self.shared_fc(x))

        return shared_features


def get_multitask_model(model_name='simple', num_quality_classes=3, num_fruit_classes=11,
                        input_channels=3, img_size=224):
    """
    Get a multi-task model by name.

    Args:
        model_name (str): Name of the model ('simple' - more options can be added)
        num_quality_classes (int): Number of quality classes
        num_fruit_classes (int): Number of fruit type classes
        input_channels (int): Number of input channels
        img_size (int): Input image size

    Returns:
        nn.Module: PyTorch multi-task model
    """
    models = {
        'simple': MultiTaskSimpleCNN,
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

    return models[model_name](
        num_quality_classes=num_quality_classes,
        num_fruit_classes=num_fruit_classes,
        input_channels=input_channels,
        img_size=img_size
    )


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters by component
    if isinstance(model, MultiTaskSimpleCNN):
        # Backbone parameters
        backbone_params = sum(p.numel() for name, p in model.named_parameters()
                            if 'conv' in name or 'bn' in name or 'shared_fc' in name)
        # Quality head parameters
        quality_params = sum(p.numel() for name, p in model.named_parameters()
                           if 'quality_fc' in name)
        # Fruit head parameters
        fruit_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'fruit_fc' in name)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': backbone_params,
            'quality_head': quality_params,
            'fruit_head': fruit_params
        }

    return {
        'total': total_params,
        'trainable': trainable_params
    }


def print_multitask_model_summary(model, input_size=(3, 224, 224)):
    """
    Print a summary of the multi-task model architecture.

    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size (C, H, W)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Quality classes: {model.num_quality_classes}")
    print(f"Fruit type classes: {model.num_fruit_classes}")

    param_counts = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")

    if 'backbone' in param_counts:
        print(f"  Backbone (shared): {param_counts['backbone']:,}")
        print(f"  Quality head: {param_counts['quality_head']:,}")
        print(f"  Fruit type head: {param_counts['fruit_head']:,}")

    # Test forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            quality_out, fruit_out = model(dummy_input)
        print(f"\nOutput shapes:")
        print(f"  Quality logits: {quality_out.shape}")
        print(f"  Fruit type logits: {fruit_out.shape}")
    except Exception as e:
        print(f"\nCould not run test forward pass: {e}")

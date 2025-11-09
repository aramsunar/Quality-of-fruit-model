"""
Utility functions for the fruit quality assessment project.
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_directories(base_dir, scenario_name):
    """
    Create necessary directories for saving models and results.

    Args:
        base_dir (str): Base directory path
        scenario_name (str): Name of the scenario

    Returns:
        dict: Dictionary containing paths to created directories
    """
    paths = {
        'models': Path(base_dir) / 'models' / scenario_name,
        'reports': Path(base_dir) / 'reports' / scenario_name,
        'figures': Path(base_dir) / 'reports' / scenario_name / 'figures'
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_config(config, save_path):
    """
    Save configuration to JSON file.

    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_timestamp():
    """
    Get current timestamp as string.

    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_time(seconds):
    """
    Format time in seconds to human-readable format.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_section(title, width=70):
    """
    Print a formatted section header.

    Args:
        title (str): Section title
        width (int): Width of the header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_subsection(title, width=70):
    """
    Print a formatted subsection header.

    Args:
        title (str): Subsection title
        width (int): Width of the header
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_config_to_json(config, save_path):
    """
    Save configuration object to JSON file.

    Extracts all uppercase attributes from a config object,
    converts Path objects to strings, and saves to JSON.

    Args:
        config: Configuration object (class with uppercase attributes)
        save_path: Path to save JSON file
    """
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and attr.isupper():
            value = getattr(config, attr)
            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)
            config_dict[attr] = value

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"Configuration saved to: {save_path}")

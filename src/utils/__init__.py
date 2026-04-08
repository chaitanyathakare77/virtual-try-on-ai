"""Utility Functions"""

import torch
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device"""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_checkpoint_dir(checkpoint_dir: str) -> Path:
    """Create checkpoint directory"""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Get path to latest checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    checkpoints = sorted(checkpoint_path.glob("*.pt"))
    if not checkpoints:
        return None
    
    return str(checkpoints[-1])


def log_metrics(metrics: Dict[str, float], step: int, logger=None):
    """Log metrics"""
    log_str = f"Step {step}: "
    for key, value in metrics.items():
        log_str += f"{key}={value:.4f}, "
    
    if logger:
        logger.info(log_str)
    else:
        print(log_str)

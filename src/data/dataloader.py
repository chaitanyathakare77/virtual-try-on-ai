"""Data Loading Utilities"""

import torch
from torch.utils.data import DataLoader
from .dataset import VirtualTryOnDataset
from typing import Tuple


def create_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    split: str = "train",
    image_size: int = 512,
    augment: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create dataloader for Virtual Try-On dataset
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        split: "train", "val", or "test"
        image_size: Size of images
        augment: Whether to apply augmentations
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    dataset = VirtualTryOnDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        augment=augment,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=split == "train",
    )
    
    return dataloader


def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create all dataloaders (train, val, test)
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Size of images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="train",
        image_size=image_size,
        augment=True,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="val",
        image_size=image_size,
        augment=False,
        shuffle=False,
    )
    
    test_loader = create_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="test",
        image_size=image_size,
        augment=False,
        shuffle=False,
    )
    
    return train_loader, val_loader, test_loader

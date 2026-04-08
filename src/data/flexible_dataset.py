"""Flexible Dataset Loader for Unorganized Images"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random


class FlexibleVirtualTryOnDataset(Dataset):
    """
    Flexible Virtual Try-On Dataset - works with unorganized images
    
    Usage:
    ------
    # Simple - all images in one folder
    dataset = FlexibleVirtualTryOnDataset(
        data_root="path/to/all_images",
        split="train"
    )
    
    # The loader will automatically pair images randomly
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
        num_samples: Optional[int] = None,
    ):
        """
        Initialize flexible dataset
        
        Args:
            data_root: Root directory containing all images (unorganized is fine)
            split: "train", "val", or "test"
            image_size: Size of images (square)
            augment: Whether to apply augmentations
            num_samples: Total number of samples (default: auto-detect)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Find all image files
        self.image_paths = self._find_all_images()
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_root}")
        
        print(f"Found {len(self.image_paths)} images in {data_root}")
        
        # Set number of samples
        if num_samples is None:
            num_samples = len(self.image_paths)
        self.num_samples = num_samples
        
        # Split dataset
        total_samples = len(self.image_paths)
        train_end = int(0.85 * total_samples)
        val_end = int(0.95 * total_samples)
        
        if split == "train":
            self.indices = list(range(0, train_end))
        elif split == "val":
            self.indices = list(range(train_end, val_end))
        else:  # test
            self.indices = list(range(val_end, total_samples))
        
        print(f"Dataset split: {split} with {len(self.indices)} samples")
        
        # Augmentations
        self.transform = self._get_transforms()
    
    def _find_all_images(self) -> List[Path]:
        """Find all image files in directory (recursive)"""
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if Path(file).suffix.lower() in valid_formats:
                    images.append(Path(root) / file)
        
        return sorted(images)
    
    def _get_transforms(self):
        """Get image transforms"""
        if self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(p=0.2),
                A.GlassBlur(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get sample - pairs images randomly from all available images
        
        Returns:
            dict with keys:
                - person_image: First random image
                - garment_image: Second random image
                - try_on_image: Third random image (synthetic ground truth)
                - person_path: Path to first image
                - garment_path: Path to second image
        """
        # Get primary image index
        actual_idx = self.indices[idx]
        person_path = self.image_paths[actual_idx]
        
        # Randomly select two other images
        garment_idx = random.randint(0, len(self.image_paths) - 1)
        try_on_idx = random.randint(0, len(self.image_paths) - 1)
        
        garment_path = self.image_paths[garment_idx]
        try_on_path = self.image_paths[try_on_idx]
        
        # Load images
        try:
            person_img = Image.open(person_path).convert("RGB")
            garment_img = Image.open(garment_path).convert("RGB")
            try_on_img = Image.open(try_on_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {e}")
            # Return a random sample if error
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transforms
        person_np = self.transform(image=np.array(person_img))["image"]
        garment_np = self.transform(image=np.array(garment_img))["image"]
        try_on_np = self.transform(image=np.array(try_on_img))["image"]
        
        return {
            "person_image": person_np,
            "garment_image": garment_np,
            "try_on_image": try_on_np,
            "person_path": str(person_path),
            "garment_path": str(garment_path),
        }


class SimpleImageDataset(Dataset):
    """
    Ultra-simple dataset for pairing ANY 43k images
    Automatically handles unorganized folder structure
    """
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        """
        Initialize simple dataset
        
        Args:
            data_root: Root directory with images (any structure)
            image_size: Output image size
            augment: Apply augmentations
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        
        # Find all images
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.images = []
        
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if Path(file).suffix.lower() in valid_formats:
                    self.images.append(Path(root) / file)
        
        self.images = sorted(self.images)
        
        if not self.images:
            raise ValueError(f"No images found in {data_root}")
        
        print(f"✓ Loaded {len(self.images)} images from {data_root}")
        
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(p=0.2) if augment else A.NoOp(),
            A.HorizontalFlip(p=0.5) if augment else A.NoOp(),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Simple: just return image and a pair
        The model will handle pairing internally
        """
        # Get primary image
        img1_path = self.images[idx]
        
        # Get random pair
        pair_idx = random.randint(0, len(self.images) - 1)
        img2_path = self.images[pair_idx]
        
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img1_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transforms
        img1_tensor = self.transform(image=np.array(img1))["image"]
        img2_tensor = self.transform(image=np.array(img2))["image"]
        
        return {
            "image1": img1_tensor,
            "image2": img2_tensor,
            "path1": str(img1_path),
            "path2": str(img2_path),
        }

"""Virtual Try-On Dataset"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VirtualTryOnDataset(Dataset):
    """
    Virtual Try-On Dataset
    
    Expected directory structure:
    datasets/
    ├── person/
    │   ├── person_001.jpg
    │   ├── person_002.jpg
    │   └── ...
    ├── garment/
    │   ├── garment_001.jpg
    │   ├── garment_002.jpg
    │   └── ...
    └── ground_truth/
        ├── try_on_001.jpg
        ├── try_on_002.jpg
        └── ...
    """
    
    def __init__(
        self,
        data_root: str = "./datasets/raw",
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
    ):
        """
        Initialize dataset
        
        Args:
            data_root: Root directory of dataset
            split: "train", "val", or "test"
            image_size: Size of images (square)
            augment: Whether to apply augmentations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        self.person_dir = self.data_root / "person"
        self.garment_dir = self.data_root / "garment"
        self.ground_truth_dir = self.data_root / "ground_truth"
        
        # Get list of samples
        self.person_images = sorted(list(self.person_dir.glob("*.jpg")) + list(self.person_dir.glob("*.png")))
        self.garment_images = sorted(list(self.garment_dir.glob("*.jpg")) + list(self.garment_dir.glob("*.png")))
        self.try_on_images = sorted(list(self.ground_truth_dir.glob("*.jpg")) + list(self.ground_truth_dir.glob("*.png")))
        
        # Split dataset
        total_samples = len(self.person_images)
        train_end = int(0.85 * total_samples)
        val_end = int(0.95 * total_samples)
        
        if split == "train":
            self.indices = range(0, train_end)
        elif split == "val":
            self.indices = range(train_end, val_end)
        else:  # test
            self.indices = range(val_end, total_samples)
        
        # Augmentations
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get image transforms"""
        if self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussBlur(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc'))
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
        Get sample
        
        Returns:
            dict with keys:
                - person_image: Person image tensor
                - garment_image: Garment image tensor
                - try_on_image: Ground truth try-on image tensor
                - person_path: Path to person image
                - garment_path: Path to garment image
        """
        actual_idx = self.indices[idx]
        
        # Load images
        person_img = Image.open(self.person_images[actual_idx]).convert("RGB")
        
        # Cyclic sampling for garment images
        garment_idx = actual_idx % len(self.garment_images)
        garment_img = Image.open(self.garment_images[garment_idx]).convert("RGB")
        
        # Cyclic sampling for ground truth
        try_on_idx = actual_idx % len(self.try_on_images)
        try_on_img = Image.open(self.try_on_images[try_on_idx]).convert("RGB")
        
        # Apply transforms
        person_np = self.transform(image=np.array(person_img))["image"]
        garment_np = self.transform(image=np.array(garment_img))["image"]
        try_on_np = self.transform(image=np.array(try_on_img))["image"]
        
        return {
            "person_image": person_np,
            "garment_image": garment_np,
            "try_on_image": try_on_np,
            "person_path": str(self.person_images[actual_idx]),
            "garment_path": str(self.garment_images[garment_idx]),
        }

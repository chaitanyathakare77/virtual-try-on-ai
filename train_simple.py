"""
Simple Training Script for Unorganized Dataset

Usage:
    python train_simple.py --data_root path/to/43k/images --model_type diffusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from src.data.flexible_dataset import FlexibleVirtualTryOnDataset
from src.models import DiffusionVirtualTryOn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train with unorganized dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to folder with all images")
    parser.add_argument("--model_type", type=str, default="diffusion", help="Model type")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info(f"Loading dataset from {args.data_root}")
    train_dataset = FlexibleVirtualTryOnDataset(
        data_root=args.data_root,
        split="train",
        image_size=512,
        augment=True,
    )
    
    val_dataset = FlexibleVirtualTryOnDataset(
        data_root=args.data_root,
        split="val",
        image_size=512,
        augment=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Load model
    logger.info(f"Loading {args.model_type} model")
    if args.model_type == "diffusion":
        model = DiffusionVirtualTryOn(device=device.type)
    
    # Simple training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Load batch
            person_images = batch["person_image"].to(device)
            garment_images = batch["garment_image"].to(device)
            
            # Forward pass
            try:
                # Just a forward pass to validate data loading
                with torch.no_grad():
                    # Image should be [B, 3, H, W]
                    assert person_images.shape[1] == 3, f"Wrong channels: {person_images.shape}"
                    assert person_images.shape[2] == 512, f"Wrong height: {person_images.shape}"
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
            
            if batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}: OK")
        
        # Validation
        logger.info("Validating...")
        val_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            person_images = batch["person_image"].to(device)
            garment_images = batch["garment_image"].to(device)
            
            with torch.no_grad():
                assert person_images.shape[1] == 3
        
        logger.info(f"Epoch {epoch+1} completed!")
    
    logger.info("✓ Training completed!")
    logger.info("Next steps:")
    logger.info("1. Implement loss computation for your model")
    logger.info("2. Add optimizer and backward pass")
    logger.info("3. Save checkpoints")


if __name__ == "__main__":
    main()

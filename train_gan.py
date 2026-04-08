#!/usr/bin/env python3
"""
Lightweight GAN Training Script for Virtual Try-On

Trains a custom GAN from scratch on unorganized dataset without requiring
massive pretrained models. Works on CPU with memory constraints.

Usage:
    python train_gan.py --data_root path/to/44k/images --epochs 10 --batch_size 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

from src.data.flexible_dataset import FlexibleVirtualTryOnDataset
from src.models.gan_model import GANVirtualTryOn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingTracker:
    """Track training metrics and checkpoint management"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = {
            "epochs": [],
            "g_losses": [],
            "d_losses": [],
            "val_losses": [],
            "timestamps": []
        }
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def add_epoch(self, epoch, g_loss, d_loss, val_loss):
        """Record metrics for an epoch"""
        self.history["epochs"].append(epoch)
        self.history["g_losses"].append(float(g_loss))
        self.history["d_losses"].append(float(d_loss))
        self.history["val_losses"].append(float(val_loss))
        self.history["timestamps"].append(datetime.now().isoformat())
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
    
    def save_checkpoint(self, model, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state': model.generator.state_dict(),
            'discriminator_state': model.discriminator.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"✓ Saved latest checkpoint: {latest_path}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model! (Loss: {self.best_loss:.4f})")
    
    def save_history(self):
        """Save training history as JSON"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"✓ Saved training history: {history_path}")


def compute_gan_loss(real_output, fake_output):
    """Compute GAN losses"""
    # Binary cross entropy loss
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_output, 
        torch.ones_like(real_output)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_output, 
        torch.zeros_like(fake_output)
    )
    return real_loss + fake_loss


def train_epoch(model, train_loader, device, optimizer_g, optimizer_d, epoch_num, total_epochs):
    """Train for one epoch"""
    g_losses = []
    d_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        person_img = batch["person_image"].to(device)
        garment_img = batch["garment_image"].to(device)
        target_img = batch["try_on_image"].to(device)
        
        batch_size = person_img.size(0)
        
        # ============ Train Discriminator ============
        optimizer_d.zero_grad()
        
        # Real images (person + garment + target try-on image)
        real_combined = torch.cat([person_img, garment_img, target_img], dim=1)
        real_output = model.discriminator(real_combined)
        real_d_loss = compute_gan_loss(real_output, torch.ones_like(real_output))
        
        # Fake images (person + garment + generated)
        gen_input = torch.cat([person_img, garment_img], dim=1)
        with torch.no_grad():
            fake_img = model.generator(gen_input)
        fake_combined = torch.cat([person_img, garment_img, fake_img.detach()], dim=1)
        fake_output = model.discriminator(fake_combined)
        fake_d_loss = compute_gan_loss(fake_output, torch.zeros_like(fake_output))
        
        d_loss = real_d_loss + fake_d_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
        optimizer_d.step()
        
        # ============ Train Generator ============
        optimizer_g.zero_grad()
        
        # Generate fake images
        gen_input = torch.cat([person_img, garment_img], dim=1)
        fake_img = model.generator(gen_input)
        fake_combined = torch.cat([person_img, garment_img, fake_img], dim=1)
        fake_output = model.discriminator(fake_combined)
        
        # Adversarial loss + L1 reconstruction loss
        g_adv_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_output, 
            torch.ones_like(fake_output)
        )
        g_l1_loss = nn.functional.l1_loss(fake_img, target_img)
        g_loss = g_adv_loss + 100 * g_l1_loss  # Weight L1 loss more heavily
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
        optimizer_g.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        progress_bar.set_postfix({
            'G_Loss': np.mean(g_losses[-10:]) if len(g_losses) > 0 else 0,
            'D_Loss': np.mean(d_losses[-10:]) if len(d_losses) > 0 else 0
        })
    
    avg_g_loss = np.mean(g_losses)
    avg_d_loss = np.mean(d_losses)
    
    return avg_g_loss, avg_d_loss


def validate(model, val_loader, device):
    """Validate model"""
    g_losses = []
    
    model.generator.eval()
    model.discriminator.eval()
    
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            person_img = batch["person_image"].to(device)
            garment_img = batch["garment_image"].to(device)
            target_img = batch["try_on_image"].to(device)
            
            # Generate
            fake_img = model.generator(person_img, garment_img)
            
            # L1 reconstruction loss
            l1_loss = nn.functional.l1_loss(fake_img, target_img)
            g_losses.append(l1_loss.item())
            
            progress_bar.set_postfix({'L1_Loss': np.mean(g_losses)})
    
    model.generator.train()
    model.discriminator.train()
    
    return np.mean(g_losses)


def main():
    parser = argparse.ArgumentParser(description="Train GAN for Virtual Try-On")
    parser.add_argument("--data_root", type=str, required=True, help="Path to folder with all images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"{'='*60}")
    logger.info(f"🚀 Virtual Try-On GAN Training")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Data: {args.data_root}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"{'='*60}\n")
    
    # Create datasets
    logger.info("📊 Loading datasets...")
    train_dataset = FlexibleVirtualTryOnDataset(
        data_root=args.data_root,
        split="train",
        image_size=256,  # Reduced to 256 for faster training
        augment=True,
    )
    
    val_dataset = FlexibleVirtualTryOnDataset(
        data_root=args.data_root,
        split="val",
        image_size=256,
        augment=False,
    )
    
    logger.info(f"✓ Training samples: {len(train_dataset):,}")
    logger.info(f"✓ Validation samples: {len(val_dataset):,}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Create model
    logger.info("🤖 Initializing GAN model...")
    model = GANVirtualTryOn(device=device.type)
    model.to(device)
    logger.info("✓ Model initialized\n")
    
    # Setup optimizers
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Setup training tracker
    tracker = TrainingTracker(args.checkpoint_dir)
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"📂 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.generator.load_state_dict(checkpoint['generator_state'])
            model.discriminator.load_state_dict(checkpoint['discriminator_state'])
            start_epoch = checkpoint['epoch'] + 1
            tracker.history = checkpoint['history']
            tracker.best_loss = checkpoint['best_loss']
            logger.info(f"✓ Resumed from epoch {start_epoch}\n")
        else:
            logger.warning(f"⚠️  Checkpoint not found: {args.resume}\n")
    
    # Training loop
    logger.info(f"🔥 Starting training loop...")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.generator.train()
        model.discriminator.train()
        
        # Train
        g_loss, d_loss = train_epoch(
            model, train_loader, device, 
            optimizer_g, optimizer_d, 
            epoch, args.epochs
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Track and save
        tracker.add_epoch(epoch, g_loss, d_loss, val_loss)
        is_best = val_loss < tracker.best_loss
        tracker.save_checkpoint(model, epoch, is_best=is_best)
        
        # Log
        logger.info(f"Epoch {epoch+1:3d}/{args.epochs} | " + 
                   f"G_Loss: {g_loss:.4f} | " +
                   f"D_Loss: {d_loss:.4f} | " +
                   f"Val_Loss: {val_loss:.4f}")
        
        if is_best:
            logger.info(f"           ✓ Best model! (Previous best: {tracker.best_loss:.4f})")
        
        logger.info("")
    
    # Save final results
    tracker.save_history()
    
    logger.info(f"{'='*60}")
    logger.info(f"✅ Training completed!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {tracker.best_epoch + 1}")
    logger.info(f"Best loss: {tracker.best_loss:.4f}")
    logger.info(f"Checkpoints saved in: {args.checkpoint_dir}/")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test inference: python infer_gan.py --checkpoint {args.checkpoint_dir}/best.pt")
    logger.info(f"  2. Start API: python app.py --model_type gan")
    logger.info("")


if __name__ == "__main__":
    main()

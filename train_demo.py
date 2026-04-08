#!/usr/bin/env python3
"""
Quick Demo Training Script - Fast Validation of Pipeline

This script trains on a smaller subset of images for quick iteration.
Good for testing before running full training.

Usage:
    python train_demo.py --data_root path/to/44k/images --num_samples 500 --epochs 2
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


def compute_gan_loss(real_output, fake_output):
    """Compute GAN losses"""
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output)
    )
    return real_loss + fake_loss


def train_batch(model, batch, device, optimizer_g, optimizer_d):
    """Train one batch and return losses"""
    person_img = batch["person_image"].to(device)
    garment_img = batch["garment_image"].to(device)
    target_img = batch["try_on_image"].to(device)
    
    # Train Discriminator
    optimizer_d.zero_grad()
    
    # Real
    real_combined = torch.cat([person_img, garment_img, target_img], dim=1)
    real_output = model.discriminator(real_combined)
    real_d_loss = compute_gan_loss(real_output, torch.ones_like(real_output))
    
    # Fake
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
    
    # Train Generator
    optimizer_g.zero_grad()
    
    gen_input = torch.cat([person_img, garment_img], dim=1)
    fake_img = model.generator(gen_input)
    fake_combined = torch.cat([person_img, garment_img, fake_img], dim=1)
    fake_output = model.discriminator(fake_combined)
    
    g_adv_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.ones_like(fake_output)
    )
    g_l1_loss = nn.functional.l1_loss(fake_img, target_img)
    g_loss = g_adv_loss + 100 * g_l1_loss
    
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
    optimizer_g.step()
    
    return g_loss.item(), d_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Quick Demo GAN Training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to images")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"{'='*60}")
    logger.info(f"🚀 DEMO GAN Training (Fast)")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"{'='*60}\n")
    
    # Load dataset
    logger.info("📊 Loading data...")
    dataset = FlexibleVirtualTryOnDataset(
        data_root=args.data_root,
        split="train",
        image_size=128,  # SMALLER for speed
        augment=True,
        num_samples=args.num_samples,
    )
    
    # Take only subset
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=0)
    logger.info(f"✓ Loaded {len(subset)} samples, {len(loader)} batches\n")
    
    # Model
    logger.info("🤖 Initializing model...")
    model = GANVirtualTryOn(device=device.type)
    model.to(device)
    logger.info("✓ Model ready\n")
    
    # Optimizers
    opt_g = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training
    logger.info("🔥 Training...\n")
    for epoch in range(args.epochs):
        g_losses = []
        d_losses = []
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            g_l, d_l = train_batch(model, batch, device, opt_g, opt_d)
            g_losses.append(g_l)
            d_losses.append(d_l)
            progress.set_postfix({
                'G': np.mean(g_losses[-10:]),
                'D': np.mean(d_losses[-10:])
            })
        
        logger.info(f"Epoch {epoch+1}: G_Loss={np.mean(g_losses):.4f}, D_Loss={np.mean(d_losses):.4f}\n")
    
    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'generator_state': model.generator.state_dict(),
        'discriminator_state': model.discriminator.state_dict(),
    }, "checkpoints/demo.pt")
    
    logger.info(f"{'='*60}")
    logger.info(f"✅ Demo completed! Model saved to checkpoints/demo.pt")
    logger.info(f"{'='*60}\n")
    logger.info("Next: Run full training with train_gan.py\n")


if __name__ == "__main__":
    main()

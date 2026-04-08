"""Training Script for Virtual Try-On Models"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import logging
from typing import Dict, Any
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_diffusion_model(config: Dict[str, Any]):
    """Train diffusion-based model"""
    from src.models import DiffusionVirtualTryOn
    from src.data import create_dataloaders
    
    logger.info("Training Diffusion-based Virtual Try-On Model")
    
    device = torch.device(config['hardware']['device'])
    
    # Create model
    model = DiffusionVirtualTryOn(
        model_name_or_path=config['model']['pretrained_model_name_or_path'],
        device=device.type,
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['dataset']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        image_size=config['model']['image_size'],
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.pipe.unet.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch
            person_images = batch["person_image"].to(device)
            garment_images = batch["garment_image"].to(device)
            try_on_images = batch["try_on_image"].to(device)
            
            # Forward pass
            with torch.no_grad():
                latents = model.pipe.vae.encode(try_on_images).latent_dist.sample() * 0.18215
            
            # Random timestep
            timesteps = torch.randint(0, 1000, (person_images.shape[0],), device=device)
            
            # Add noise
            noise = torch.randn_like(latents, device=device)
            noisy_latents = model.pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            text_embeddings = model.pipe._encode_prompt("a person wearing clothes", device, 1, False)
            
            # Predict noise
            model_pred = model.pipe.unet(noisy_latents, timesteps, text_embeddings).sample
            
            # MSE loss
            loss = nn.functional.mse_loss(model_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config['logging']['log_dir']) / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.pipe.unet.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")


def train_gan_model(config: Dict[str, Any]):
    """Train GAN-based model"""
    from src.models import GANVirtualTryOn
    from src.data import create_dataloaders
    
    logger.info("Training GAN-based Virtual Try-On Model")
    
    device = torch.device(config['hardware']['device'])
    
    # Create model
    gan_model = GANVirtualTryOn(device=device.type)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['dataset']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        image_size=config['model']['image_size'],
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # Setup optimizers
    optimizer_g = optim.Adam(
        gan_model.generator.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
    )
    optimizer_d = optim.Adam(
        gan_model.discriminator.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
    )
    
    # Loss functions
    criterion = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            person_images = batch["person_image"].to(device)
            garment_images = batch["garment_image"].to(device)
            try_on_images = batch["try_on_image"].to(device)
            
            # Concatenate inputs
            input_images = torch.cat([person_images, garment_images], dim=1)
            
            # Train discriminator
            real_output = gan_model.discriminator(try_on_images)
            fake_images = gan_model.generator(input_images)
            fake_output = gan_model.discriminator(fake_images.detach())
            
            loss_d_real = criterion(real_output, torch.ones_like(real_output))
            loss_d_fake = criterion(fake_output, torch.zeros_like(fake_output))
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            
            # Train generator
            fake_output = gan_model.discriminator(fake_images)
            loss_g_gan = criterion(fake_output, torch.ones_like(fake_output))
            loss_g_l1 = l1_loss(fake_images, try_on_images)
            loss_g = loss_g_gan + 100 * loss_g_l1
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            if batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}: D_loss={loss_d:.4f}, G_loss={loss_g:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config['logging']['log_dir']) / f"checkpoint_gan_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            gan_model.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Virtual Try-On Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model_type", type=str, default="diffusion", choices=["diffusion", "gan"], help="Model type")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.model_type == "diffusion":
        train_diffusion_model(config)
    else:
        train_gan_model(config)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

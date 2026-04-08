#!/usr/bin/env python3
"""
Inference script for trained GAN model

Usage:
    python infer_gan.py --checkpoint checkpoints/best.pt --person_image person.jpg --garment_image dress.jpg --output result.png
"""

import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from src.models.gan_model import GANVirtualTryOn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path, size=256):
    """Load and preprocess an image"""
    img = Image.open(image_path).convert("RGB")
    
    # Resize
    img = transforms.Resize((size, size))(img)
    
    # Convert to tensor
    img_tensor = transforms.ToTensor()(img)
    
    # Normalize to [-1, 1]
    img_tensor = 2 * img_tensor - 1
    
    return img_tensor, img


def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    
    # Convert to uint8
    img_np = (img_np * 255).astype("uint8")
    
    return Image.fromarray(img_np)


def main():
    parser = argparse.ArgumentParser(description="Inference with trained GAN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--person_image", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment_image", type=str, required=True, help="Path to garment image")
    parser.add_argument("--output", type=str, default="result.png", help="Output path")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"{'='*60}")
    logger.info(f"🎨 Virtual Try-On Inference")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Image size: {args.size}x{args.size}")
    logger.info("")
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    person_path = Path(args.person_image)
    garment_path = Path(args.garment_image)
    
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not person_path.exists():
        logger.error(f"❌ Person image not found: {person_path}")
        return
    
    if not garment_path.exists():
        logger.error(f"❌ Garment image not found: {garment_path}")
        return
    
    # Load model
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    model = GANVirtualTryOn(device=device.type)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state'])
    logger.info("✓ Model loaded\n")
    
    # Load images
    logger.info(f"📷 Loading images...")
    person_tensor, person_img = load_image(person_path, args.size)
    garment_tensor, garment_img = load_image(garment_path, args.size)
    
    logger.info(f"✓ Person image: {person_img.size}")
    logger.info(f"✓ Garment image: {garment_img.size}\n")
    
    # Inference
    logger.info(f"🔮 Generating try-on result...")
    model.generator.eval()
    
    with torch.no_grad():
        person_batch = person_tensor.unsqueeze(0).to(device)
        garment_batch = garment_tensor.unsqueeze(0).to(device)
        
        result_tensor = model.generator(person_batch, garment_batch)
        result_img = tensor_to_image(result_tensor[0])
    
    logger.info(f"✓ Generation complete\n")
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    
    logger.info(f"✅ Result saved: {output_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

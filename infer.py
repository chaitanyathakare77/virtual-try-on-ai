"""Command-line Inference Script"""

import argparse
import logging
from pathlib import Path
from src.models import DiffusionVirtualTryOn, GANVirtualTryOn
from src.inference import Inferencer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Virtual Try-On Inference CLI")
    parser.add_argument("--person_image", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment_image", type=str, required=True, help="Path to garment image")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--model_type", type=str, default="diffusion", choices=["diffusion", "gan"],
                        help="Model type")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--prompt", type=str, default="a person wearing clothes", help="Text prompt")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading {args.model_type} model...")
    if args.model_type == "diffusion":
        model = DiffusionVirtualTryOn(device=device)
    else:
        model = GANVirtualTryOn(device=device)
        if args.model_path:
            model.load_checkpoint(args.model_path)
    
    # Run inference
    logger.info("Running inference...")
    inferencer = Inferencer(model, device=device)
    
    output_image = inferencer.inference(
        person_image_path=args.person_image,
        garment_image_path=args.garment_image,
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    
    logger.info(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()

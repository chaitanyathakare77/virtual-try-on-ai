"""Run FastAPI Server"""

import uvicorn
import argparse
import os
from src.api import create_app
from src.models import DiffusionVirtualTryOn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Virtual Try-On API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--model_type", type=str, default="gan", choices=["diffusion", "gan"],
                        help="Model type to load")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (for GAN)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🚀 Virtual Try-On AI - API Server")
    logger.info("="*60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("")
    
    try:
        if args.model_type == "diffusion":
            logger.info("Loading Stable Diffusion model...")
            model = DiffusionVirtualTryOn(device=args.device)
        else:
            logger.info("Loading GAN model...")
            from src.models import GANVirtualTryOn
            model = GANVirtualTryOn(device=args.device)
            
            # Load checkpoint if provided
            if args.checkpoint:
                logger.info(f"Loading checkpoint: {args.checkpoint}")
                import torch
                checkpoint = torch.load(args.checkpoint, map_location=args.device)
                model.generator.load_state_dict(checkpoint['generator_state'])
                logger.info("✓ Checkpoint loaded successfully")
        
        logger.info("✓ Model loaded successfully\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    app = create_app(model=model)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"API Docs: http://{args.host}:{args.port}/docs")
    logger.info("")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

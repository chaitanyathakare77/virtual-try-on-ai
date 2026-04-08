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
    parser.add_argument("--model_type", type=str, default="diffusion", choices=["diffusion", "gan"],
                        help="Model type to load")
    
    args = parser.parse_args()
    
    logger.info("Loading Virtual Try-On model...")
    
    if args.model_type == "diffusion":
        model = DiffusionVirtualTryOn()
    else:
        from src.models import GANVirtualTryOn
        model = GANVirtualTryOn()
    
    app = create_app(model=model)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

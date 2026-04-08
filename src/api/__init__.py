"""FastAPI Application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from .routes import router, set_model
from src.models import DiffusionVirtualTryOn


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(model=None) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        model: Optional pre-loaded model
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Virtual Try-On AI",
        description="Advanced AI Image Generator for Virtual Try-On with Pose and Clothing Changes",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    # Load model on startup
    @app.on_event("startup")
    async def startup_event():
        logger.info("Loading Virtual Try-On model...")
        if model is not None:
            set_model(model)
        else:
            # Load default model
            try:
                tryon_model = DiffusionVirtualTryOn()
                set_model(tryon_model)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down...")
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Welcome to Virtual Try-On AI API",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    
    return app

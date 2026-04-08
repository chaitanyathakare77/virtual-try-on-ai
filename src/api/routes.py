"""FastAPI Routes for Virtual Try-On"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import io
import torch
from PIL import Image
from typing import Optional
import os
from datetime import datetime

router = APIRouter(prefix="/api/v1/tryon", tags=["try-on"])

# Global model instance (initialized at startup)
model_instance = None


def set_model(model):
    """Set the global model instance"""
    global model_instance
    model_instance = model


def get_model():
    """Get the global model instance"""
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model_instance


@router.post("/generate")
async def generate_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
):
    """
    Generate virtual try-on image
    
    Args:
        person_image: Person image file
        garment_image: Garment image file
        prompt: Optional text prompt
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        
    Returns:
        Generated image as PNG
    """
    try:
        model = get_model()
        
        # Load images
        person_content = await person_image.read()
        garment_content = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_content)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_content)).convert("RGB")
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                person_image=person_img,
                garment_image=garment_img,
                prompt=prompt or "a person wearing clothes",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        output.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        return FileResponse(
            io.BytesIO(img_byte_arr.getvalue()),
            media_type="image/png",
            filename=f"tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_generate")
async def batch_generate(
    num_images: int = Form(5),
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Generate multiple try-on variations
    
    Args:
        num_images: Number of variations to generate
        person_image: Person image file
        garment_image: Garment image file
        prompt: Optional text prompt
        
    Returns:
        Multiple generated images as ZIP file
    """
    try:
        model = get_model()
        
        # Load base images
        person_content = await person_image.read()
        garment_content = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_content)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_content)).convert("RGB")
        
        # Generate multiple variations
        outputs = []
        for i in range(num_images):
            with torch.no_grad():
                seed = i * 42
                torch.manual_seed(seed)
                output = model.generate(
                    person_image=person_img,
                    garment_image=garment_img,
                    prompt=prompt or f"a person wearing clothes, variation {i+1}",
                    seed=seed,
                )
            outputs.append(output)
        
        # Create ZIP file
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for i, img in enumerate(outputs):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                zip_file.writestr(f"tryon_{i+1}.png", img_byte_arr.getvalue())
        
        zip_buffer.seek(0)
        return FileResponse(
            io.BytesIO(zip_buffer.getvalue()),
            media_type="application/zip",
            filename=f"tryons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": str(next(model_instance.parameters()).device) if model_instance else None
    }

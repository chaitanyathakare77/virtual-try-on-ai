"""Diffusion Model for Virtual Try-On"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Union, List
import PIL.Image


class DiffusionVirtualTryOn:
    """Advanced Diffusion-based Virtual Try-On Model"""
    
    def __init__(
        self,
        model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the Diffusion Virtual Try-On model
        
        Args:
            model_name_or_path: HF model path or name
            device: Device to load model on
            dtype: Data type for inference
        """
        self.device = device
        self.dtype = dtype
        
        # Load pretrained Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        if device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
    def generate(
        self,
        person_image: PIL.Image.Image,
        garment_image: PIL.Image.Image,
        prompt: str = "a person wearing clothes",
        negative_prompt: str = "blurry, distorted, disfigured",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> PIL.Image.Image:
        """
        Generate virtual try-on image
        
        Args:
            person_image: Image of person
            garment_image: Image of garment
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed
            
        Returns:
            Generated image
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Prepare images
        person_image = person_image.resize((512, 512))
        garment_image = garment_image.resize((512, 512))
        
        # Generate prompt embedding with garment info
        full_prompt = f"{prompt}, wearing {garment_image.filename if hasattr(garment_image, 'filename') else 'the garment'}"
        
        # Run inference
        with torch.no_grad():
            output = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=person_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
            )
        
        return output.images[0]
    
    def generate_batch(
        self,
        person_images: List[PIL.Image.Image],
        garment_images: List[PIL.Image.Image],
        prompts: List[str],
        **kwargs
    ) -> List[PIL.Image.Image]:
        """
        Generate virtual try-on for batch of images
        
        Args:
            person_images: List of person images
            garment_images: List of garment images
            prompts: List of text prompts
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated images
        """
        results = []
        for person, garment, prompt in zip(person_images, garment_images, prompts):
            img = self.generate(person, garment, prompt, **kwargs)
            results.append(img)
        return results
    
    def enable_attention_slicing(self):
        """Enable attention slicing for memory efficiency"""
        self.pipe.enable_attention_slicing()
    
    def disable_attention_slicing(self):
        """Disable attention slicing"""
        self.pipe.disable_attention_slicing()
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers optimizations"""
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers: {e}")

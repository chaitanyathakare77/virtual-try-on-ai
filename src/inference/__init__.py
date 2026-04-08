"""Inference Module"""

import torch
import PIL.Image
from typing import Optional, Union
import numpy as np
from pathlib import Path


class Inferencer:
    """Inference class for Virtual Try-On models"""
    
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Inferencer
        
        Args:
            model: Pre-trained model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def inference(
        self,
        person_image_path: Union[str, Path],
        garment_image_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> PIL.Image.Image:
        """
        Run inference
        
        Args:
            person_image_path: Path to person image
            garment_image_path: Path to garment image
            prompt: Optional text prompt
            **kwargs: Additional inference arguments
            
        Returns:
            Generated image
        """
        # Load images
        person_img = PIL.Image.open(person_image_path).convert("RGB")
        garment_img = PIL.Image.open(garment_image_path).convert("RGB")
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                person_image=person_img,
                garment_image=garment_img,
                prompt=prompt or "a person wearing clothes",
                **kwargs
            )
        
        return output
    
    def batch_inference(
        self,
        person_image_paths: list,
        garment_image_paths: list,
        prompts: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Run batch inference
        
        Args:
            person_image_paths: List of person image paths
            garment_image_paths: List of garment image paths
            prompts: Optional list of prompts
            **kwargs: Additional inference arguments
            
        Returns:
            List of generated images
        """
        results = []
        for i, (person_path, garment_path) in enumerate(zip(person_image_paths, garment_image_paths)):
            prompt = prompts[i] if prompts else None
            img = self.inference(person_path, garment_path, prompt, **kwargs)
            results.append(img)
        
        return results

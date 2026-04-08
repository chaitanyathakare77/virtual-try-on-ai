"""GAN-based Model for Virtual Try-On"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import PIL.Image


class ResidualBlock(nn.Module):
    """Residual Block for Generator"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return out


class Generator(nn.Module):
    """Generator Network for Virtual Try-On"""
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(256, 256) for _ in range(9)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    """Discriminator Network for Virtual Try-On"""
    
    def __init__(self, in_channels: int = 9):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),
        )
    
    def forward(self, x):
        return self.model(x)


class GANVirtualTryOn(nn.Module):
    """GAN-based Virtual Try-On Model"""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize GAN Virtual Try-On model
        
        Args:
            device: Device to load model on
            checkpoint_path: Path to pretrained weights
        """
        super().__init__()
        self.device = device
        self.generator = Generator(in_channels=6, out_channels=3)
        self.discriminator = Discriminator(in_channels=9)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def generate(
        self,
        person_image: PIL.Image.Image,
        garment_image: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Generate virtual try-on image using GAN
        
        Args:
            person_image: Image of person (512x512)
            garment_image: Image of garment (512x512)
            
        Returns:
            Generated image
        """
        # Prepare images
        person_image = person_image.resize((512, 512))
        garment_image = garment_image.resize((512, 512))
        
        # Convert to tensors
        person_tensor = self._image_to_tensor(person_image)
        garment_tensor = self._image_to_tensor(garment_image)
        
        # Concatenate
        input_tensor = torch.cat([person_tensor, garment_tensor], dim=1).to(self.device)
        
        # Generate
        with torch.no_grad():
            self.generator.eval()
            output = self.generator(input_tensor)
        
        # Convert back to image
        output_image = self._tensor_to_image(output[0])
        return output_image
    
    def _image_to_tensor(self, image: PIL.Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor"""
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(image).unsqueeze(0)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> PIL.Image.Image:
        """Convert tensor to PIL image"""
        import torchvision.transforms as transforms
        tensor = (tensor + 1) / 2  # Denormalize
        tensor = torch.clamp(tensor, 0, 1)
        to_image = transforms.ToPILImage()
        return to_image(tensor.cpu())
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint.get('generator', checkpoint))
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, path)

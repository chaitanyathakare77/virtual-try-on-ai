# Virtual Try-On AI

Advanced AI Image Generator for Virtual Try-On with Pose and Clothing Changes. Transform people's appearances by applying different clothing and poses using state-of-the-art diffusion models and GANs.

## Features

- 🎯 **Diffusion-based & GAN Models**: Choice of two advanced architectures
- 👗 **Virtual Try-On**: Swap clothing while maintaining body proportions
- 🎨 **Pose Generation**: Generate different poses and movements
- 📊 **Large Dataset Support**: Optimized for 43,000+ image datasets
- ⚡ **Memory Efficient**: Multiple optimization techniques for GPUs
- 🔄 **Batch Processing**: Generate multiple variations efficiently
- 🌐 **FastAPI Integration**: Production-ready REST API
- 📱 **CLI & Programmatic Access**: Multiple ways to use the model

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/virtual-try-on-ai.git
cd virtual-try-on-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Organize your 43,000 images in the following structure:

```
datasets/raw/
├── person/          # Person images (43,000 images)
│   ├── person_001.jpg
│   ├── person_002.jpg
│   └── ...
├── garment/         # Clothing images (43,000 images)
│   ├── garment_001.jpg
│   ├── garment_002.jpg
│   └── ...
└── ground_truth/    # Expected try-on results (43,000 images)
    ├── try_on_001.jpg
    ├── try_on_002.jpg
    └── ...
```

### 2. Configure Training

Edit `configs/config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- Hardware settings

### 3. Train Model

```bash
# Train diffusion model
python train.py --config configs/config.yaml --model_type diffusion

# OR train GAN model
python train.py --config configs/config.yaml --model_type gan
```

### 4. Run Inference

#### CLI Usage
```bash
python infer.py \
    --person_image path/to/person.jpg \
    --garment_image path/to/garment.jpg \
    --output output.png \
    --model_type diffusion \
    --prompt "a person wearing red dress"
```

#### API Server
```bash
# Start API server
python app.py --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## API Usage

### Generate Single Try-On

```bash
curl -X POST "http://localhost:8000/api/v1/tryon/generate" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -F "prompt=a person wearing formal clothes"
```

### Generate Multiple Variations

```bash
curl -X POST "http://localhost:8000/api/v1/tryon/batch_generate" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -F "num_images=5"
```

### Health Check

```bash
curl "http://localhost:8000/api/v1/tryon/health"
```

## Programmatic Usage

```python
from src.models import DiffusionVirtualTryOn
from PIL import Image

# Initialize model
model = DiffusionVirtualTryOn(device="cuda")

# Load images
person_img = Image.open("person.jpg")
garment_img = Image.open("garment.jpg")

# Generate
output = model.generate(
    person_image=person_img,
    garment_image=garment_img,
    prompt="a person wearing clothes",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Save
output.save("try_on_result.png")
```

## Project Structure

```
virtual-try-on-ai/
├── src/
│   ├── models/              # Model architectures
│   │   ├── diffusion_model.py
│   │   └── gan_model.py
│   ├── data/                # Data loading
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── inference/           # Inference
│   │   └── __init__.py
│   ├── api/                 # FastAPI
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/               # Utilities
│       └── __init__.py
├── configs/                 # Configuration files
│   └── config.yaml
├── datasets/                # Dataset directory
│   └── raw/
├── checkpoints/             # Model checkpoints
├── outputs/                 # Generated outputs
├── train.py                 # Training script
├── app.py                   # FastAPI server
├── infer.py                 # CLI inference
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## Configuration

See `configs/config.yaml` for all available options:

- **Model**: Architecture, pretrained weights, image size
- **Dataset**: Data paths, splits, augmentation
- **Training**: Learning rate, batch size, epochs, optimizations
- **Hardware**: Device, GPU settings, mixed precision
- **Inference**: Number of steps, guidance scale

## Training Tips

1. **Start with a pretrained model** for faster convergence
2. **Use mixed precision** (`fp16`) to save memory
3. **Enable attention slicing** for low-VRAM GPUs
4. **Increase gradient accumulation** if batch size is limited
5. **Monitor TensorBoard** for training metrics

## Performance

- Inference time: ~5-10 seconds per image (GPU)
- Memory requirements: 8-16GB VRAM
- Batch processing: 4-8 images simultaneously

## Troubleshooting

### Out of Memory (OOM)
```python
# Enable memory optimizations
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
```

### Slow Inference
- Reduce `num_inference_steps` (default: 50)
- Use a faster scheduler (e.g., DDIM)
- Enable xformers optimizations

### Low Quality Results
- Increase guidance_scale (7.5-15)
- Use more inference steps (50-100)
- Improve input image quality

## GPU Recommendations

| GPU | VRAM | Images/Batch | Speed |
|-----|------|--------------|-------|
| RTX 4090 | 24GB | 8 | Fast |
| RTX 3090 | 24GB | 4 | Medium |
| RTX 3080 | 10GB | 2 | Slow |
| A100 | 40GB | 16 | Very Fast |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Hugging Face Diffusers library
- Stable Diffusion team
- OpenAI CLIP model
- PyTorch community

## Citation

If you use this project, please cite:

```bibtex
@software{virtualtryonai2024,
  title={Virtual Try-On AI: Advanced Image Generator},
  author={Your Name},
  url={https://github.com/yourusername/virtual-try-on-ai},
  year={2024}
}
```

## Support

For issues, questions, or suggestions:
- Open an GitHub issue
- Email: your.email@example.com
- Documentation: [https://docs.example.com](https://docs.example.com)

---

**Note**: This is a research/educational project. For production use, ensure compliance with local regulations and respective model licenses.

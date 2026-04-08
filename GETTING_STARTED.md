# Getting Started Checklist

## 🚀 Quick Start

- [ ] **Read This First**: Open `GITHUB_PUSH_GUIDE.md` for detailed GitHub setup
- [ ] **Install Dependencies**: Run `pip install -r requirements.txt`
- [ ] **Test Import**: Run `python -c "from src.models import DiffusionVirtualTryOn"`

## 📊 Dataset Preparation

- [ ] Collect/download 43,000 images
- [ ] **Organize dataset**:
  ```
  datasets/raw/
  ├── person/        (43,000 person images)
  ├── garment/       (43,000 clothing images)
  └── ground_truth/  (43,000 expected results)
  ```
- [ ] Verify image formats (JPG/PNG)
- [ ] Check image sizes (recommend 512x512+)

## ⚙️ Configuration

- [ ] Review `configs/config.yaml`
- [ ] Adjust hyperparameters for your hardware:
  - `training.batch_size`: 2-8 (less on lower VRAM)
  - `model.num_inference_steps`: 50 (lower = faster)
  - `hardware.device`: "cuda" or "cpu"

## 🤖 Training

### Quick Test (Without Full Dataset)
```bash
# Test model loading
python -c "from src.models import DiffusionVirtualTryOn; m = DiffusionVirtualTryOn()"

# Test inference
python infer.py --person_image person.jpg --garment_image garment.jpg --output out.png
```

### Full Training
```bash
# Diffusion model (recommended)
python train.py --config configs/config.yaml --model_type diffusion

# OR GAN model
python train.py --config configs/config.yaml --model_type gan
```

## 🌐 API Server

```bash
# Start API (downloads model on first run)
python app.py --host 0.0.0.0 --port 8000

# Test API endpoints
# - Browse: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/api/v1/tryon/health
```

## 🐳 Docker Setup (Optional)

```bash
# Build image
docker build -t virtual-tryon-ai .

# Run container
docker-compose up -d

# Access at http://localhost:8000
```

## 📤 Push to GitHub

1. Create repository: https://github.com/new
2. Run commands (see GITHUB_PUSH_GUIDE.md):
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/virtual-try-on-ai.git
   git branch -M main
   git push -u origin main
   ```
3. Add badges to README.md
4. Enable GitHub Pages (optional)

## 📝 Project Updates

After pushing, keep repository updated:

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
python train.py --config configs/config.yaml

# Commit with description
git add .
git commit -m "feat: Add your feature description"

# Push back
git push origin feature/your-feature

# Create Pull Request on GitHub
```

## 🔍 Verify Installation

Run each command to verify setup:

```bash
# Python version
python --version

# PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# GPU support
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Diffusers
python -c "from diffusers import StableDiffusionPipeline; print('✓')"

# FastAPI
python -c "from fastapi import FastAPI; print('✓')"

# Project imports
python -c "from src.models import DiffusionVirtualTryOn, GANVirtualTryOn; print('✓')"
python -c "from src.data import VirtualTryOnDataset, create_dataloader; print('✓')"
python -c "from src.api import create_app; print('✓')"
```

## ⚡ Performance Tips

**GPU Memory Optimization:**
```python
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
```

**Faster Inference:**
- Reduce `num_inference_steps` (default 50 → try 30)
- Use smaller image size (512 → 384)
- Enable mixed precision (fp16)

**Better Results:**
- Increase `guidance_scale` (7.5 → 15)
- More inference steps (50 → 100)
- High-quality input images

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size, enable attention slicing |
| Slow inference | Reduce num_inference_steps, use GPU |
| Import errors | Run `pip install -r requirements.txt` |
| Model download fails | Check internet, increase timeout |
| API won't start | Check port 8000 is free, try different port |

## 📚 Documentation

- **API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Code Docs**: See docstrings in source files
- **Models**: Check `src/models/` for architecture details
- **Config**: Edit `configs/config.yaml` for customization
- **Dataset**: See dataset structure in README.md

## 📞 Support

If you encounter issues:
1. Check TROUBLESHOOTING section in README.md
2. Review GitHub Issues: https://github.com/YOUR-USERNAME/virtual-try-on-ai/issues
3. Consult documentation links in README.md

## ✅ Final Checklist Before Production

- [ ] Dataset is complete and verified
- [ ] Model training completed successfully
- [ ] API server tested with sample images
- [ ] Code committed and pushed to GitHub
- [ ] README.md updated with your changes
- [ ] GitHub Actions CI/CD passing
- [ ] Docker image builds successfully
- [ ] Performance tested on target hardware

---

**Next Step**: Open `GITHUB_PUSH_GUIDE.md` to push your project to GitHub! 🚀

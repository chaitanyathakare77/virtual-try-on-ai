# 🎨 Virtual Try-On AI - Complete Training Guide

## Quick Start (5 minutes)

```bash
# 1. Activate environment
cd "C:\Users\chait\Downloads\Ai image genaretor"
.\.venv\Scripts\Activate.ps1

# 2. Navigate to project
cd virtual-try-on-ai

# 3. Train GAN model (lightweight, works on CPU)
python train_gan.py --data_root "C:\Users\chait\Downloads\images" --epochs 5 --batch_size 4

# 4. Test inference
python infer_gan.py --checkpoint checkpoints/best.pt --person_image test_person.jpg --garment_image test_garment.jpg

# 5. Start API server
python app.py --model_type gan
```

## What Was Done

Your project now has TWO training approaches:

### ✅ RECOMMENDED: GAN Training (train_gan.py)
- **Memory**: Works on CPU with 4GB+ RAM
- **Speed**: 5-10x faster than Diffusion
- **Quality**: Custom architecture trained from scratch
- **Checkpointing**: Auto-saves best model
- **Training Time**: ~30 mins per epoch on CPU with your 44k images

```bash
python train_gan.py \
  --data_root "C:\Users\chait\Downloads\images" \
  --batch_size 4 \
  --epochs 10 \
  --lr 0.0002 \
  --checkpoint_dir checkpoints
```

### 🎯 Optional: Stable Diffusion (train_simple.py)
- Requires 16GB+ RAM or GPU
- Larger pretrained model
- Production-grade results but memory-intensive
- **Skip this for now** due to memory constraints

---

## Architecture

### Generator (train_gan.py)
```
Input: [Person Image] + [Garment Image] (6 channels)
  ↓
Encoder: 64 → 128 → 256 channels (downsampling)
  ↓
9x Residual Blocks: Feature refinement
  ↓
Decoder: 256 → 128 → 64 channels (upsampling)
  ↓
Output: [Virtual Try-On Image] (3 channels)
```

### Discriminator
```
Input: [Person] + [Garment] + [Generated] (9 channels)
  ↓
4x Conv Blocks with LeakyReLU
  ↓
Output: Binary prediction (Real vs Fake)
```

---

## Training Loop Details

### Epoch Breakdown (per ~9,371 batches):
1. **Discriminator Training**
   - Real images → should output 1
   - Fake images → should output 0
   - Loss = BCE(real, 1) + BCE(fake, 0)

2. **Generator Training**
   - Adversarial loss: fool discriminator
   - L1 reconstruction loss: match target image (weighted 100x)
   - Loss = Adversarial + 100*L1

3. **Gradient Clipping**
   - Prevents exploding gradients
   - Stabilizes training

4. **Checkpointing**
   - Saves every epoch
   - Keeps best model (lowest validation loss)
   - Saves training history as JSON

---

## Key Features

✅ **Auto Dataset Discovery**
- Scans folder recursively
- Auto-pairs images randomly
- 85% train / 10% val / 5% test split

✅ **Progressive Logging**
- Real-time training progress bars
- Loss metrics every 10 batches
- Best model tracking

✅ **Checkpoint Management**
```
checkpoints/
├── best.pt                  # Best model so far
├── latest.pt               # Most recent epoch
├── epoch_001.pt
├── epoch_002.pt
└── training_history.json   # All metrics
```

✅ **Resumable Training**
```bash
# Resume from checkpoint
python train_gan.py \
  --data_root "path/to/images" \
  --resume checkpoints/best.pt \
  --epochs 20
```

---

## Usage Examples

### 1. First Training Run (Fresh)
```bash
python train_gan.py \
  --data_root "C:\Users\chait\Downloads\images" \
  --batch_size 4 \
  --epochs 5 \
  --num_workers 0
```

### 2. Resume Training (Continue)
```bash
python train_gan.py \
  --data_root "C:\Users\chait\Downloads\images" \
  --resume checkpoints/best.pt \
  --epochs 10
```

### 3. Inference on Single Image Pair
```bash
python infer_gan.py \
  --checkpoint checkpoints/best.pt \
  --person_image path/to/person.jpg \
  --garment_image path/to/garment.jpg \
  --output result.png \
  --size 256
```

### 4. Batch Inference (Multiple Images)
```bash
for person in people/*.jpg; do
  python infer_gan.py \
    --checkpoint checkpoints/best.pt \
    --person_image "$person" \
    --garment_image garments/dress.jpg \
    --output "results/$(basename $person)"
done
```

### 5. Start API Server
```bash
# GAN model (fast, lightweight)
python app.py --model_type gan --port 8000

# Then test at: http://localhost:8000/docs
```

---

## Performance Expectations

### On CPU (Intel/AMD):
- **Per Epoch**: ~5-10 minutes for 37,481 training samples
- **Per Image Inference**: ~2-5 seconds
- **Memory**: ~2GB active usage

### On GPU (NVIDIA/AMD):
- **Per Epoch**: ~30-60 seconds
- **Per Image Inference**: ~0.1-0.3 seconds
- **Memory**: ~1.5GB (can batch 32+ images)

---

## Troubleshooting

### ❌ "MemoryError" or "Out of Memory"
**Solution**: Reduce batch size
```bash
python train_gan.py \
  --data_root "C:\Users\chait\Downloads\images" \
  --batch_size 2  # ← Changed from 4 to 2
```

### ❌ "ModuleNotFoundError: diffusers"
**Solution**: Install dependencies
```bash
pip install diffusers transformers accelerate safetensors tqdm
```

### ❌ "No images found"
**Solution**: Check data_root
```bash
# Verify images exist
ls -la "C:\Users\chait\Downloads\images" | head -20
```

### ❌ Training stuck/frozen
**Solution**: Use CPU explicitly
```bash
python train_gan.py \
  --device cpu \
  --data_root "C:\Users\chait\Downloads\images"
```

---

## Next Steps

1. **Train for 5 epochs** (test quality)
   ```bash
   python train_gan.py --data_root "C:\Users\chait\Downloads\images" --epochs 5
   ```

2. **Check results**
   ```bash
   python infer_gan.py \
     --checkpoint checkpoints/best.pt \
     --person_image "C:\Users\chait\Downloads\images\person1.jpg" \
     --garment_image "C:\Users\chait\Downloads\images\garment1.jpg"
   ```

3. **Deploy API**
   ```bash
   python app.py --model_type gan
   ```

4. **Push to GitHub**
   ```bash
   git add -A
   git commit -m "feat: Add GAN training pipeline"
   git push origin main
   ```

---

## File Structure

```
virtual-try-on-ai/
├── train_gan.py              ← GAN training script (RECOMMENDED)
├── train_simple.py           ← Diffusion (skip, GPU needed)
├── infer_gan.py              ← GAN inference
├── app.py                    ← FastAPI server
├── checkpoints/              ← Saved models
│   ├── best.pt
│   ├── latest.pt
│   └── training_history.json
└── src/
    ├── models/
    │   ├── gan_model.py      ← GAN architecture
    │   └── diffusion_model.py
    └── data/
        └── flexible_dataset.py ← Data loader
```

---

## Questions?

Refer to:
- Architecture: [src/models/gan_model.py](src/models/gan_model.py)
- Dataset: [src/data/flexible_dataset.py](src/data/flexible_dataset.py)
- Training: [train_gan.py](train_gan.py)
- API: [app.py](app.py)

**Happy training! 🚀**

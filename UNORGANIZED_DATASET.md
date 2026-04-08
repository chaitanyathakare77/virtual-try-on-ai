# Using Unorganized Dataset

If you have 43,000 images that are **NOT organized** in the standard structure, use these flexible loaders!

## ✅ Option 1: FlexibleVirtualTryOnDataset (Recommended)

Works with ANY folder structure. Automatically finds and pairs images randomly.

### Usage

```python
from src.data import FlexibleVirtualTryOnDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = FlexibleVirtualTryOnDataset(
    data_root="/path/to/all/images",  # Can be nested folders
    split="train",  # or "val", "test"
    image_size=512,
    augment=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in dataloader:
    person_image = batch["person_image"]  # [B, 3, 512, 512]
    garment_image = batch["garment_image"]  # [B, 3, 512, 512]
    try_on_image = batch["try_on_image"]  # [B, 3, 512, 512]
    print(f"Batch shape: {person_image.shape}")
```

### Features
- ✅ Works with nested folders
- ✅ Automatically finds all images (JPG, PNG, etc.)
- ✅ Random image pairing
- ✅ Train/Val/Test splits
- ✅ Augmentations support

---

## ✅ Option 2: SimpleImageDataset

Ultra-simple loader for pairs of images.

```python
from src.data import SimpleImageDataset
from torch.utils.data import DataLoader

dataset = SimpleImageDataset(
    data_root="/path/to/images",
    image_size=512,
    augment=True
)

dataloader = DataLoader(dataset, batch_size=4)

for batch in dataloader:
    img1 = batch["image1"]   # First image
    img2 = batch["image2"]   # Random pair
    print(f"Image 1: {img1.shape}, Image 2: {img2.shape}")
```

---

## 🚀 Training with Unorganized Data

Use the provided `train_simple.py` script:

```bash
python train_simple.py \
    --data_root /path/to/43k/images \
    --model_type diffusion \
    --batch_size 4 \
    --epochs 5 \
    --device cuda
```

This will:
1. ✅ Automatically find all images
2. ✅ Split into train/val/test
3. ✅ Create random image pairs
4. ✅ Handle different image formats
5. ✅ Apply augmentations

---

## 📁 Expected Dataset Structure

Your images can be in **ANY structure**:

```
/path/to/images/
├── folder1/
│   ├── image1.jpg
│   ├── image2.png
│   └── subfolder/
│       └── image3.jpg
├── folder2/
│   └── image4.jpg
├── image5.jpg
└── ... (more images)
```

**Total: 43,000 images** ✓

The loader will:
- Find all `.jpg`, `.png`, `.bmp`, `.tiff` files
- Automatically pair them for training
- Split into train (85%), val (10%), test (5%)

---

## 💡 How Pairing Works

For each sample:
1. **Person image**: Uses image at index `idx`
2. **Garment image**: Randomly picks another image
3. **Try-on image**: Randomly picks another image

This creates synthetic pairs from your unorganized data!

---

## 🔍 Example: Load & Check Data

```python
from src.data import FlexibleVirtualTryOnDataset

# Create dataset
dataset = FlexibleVirtualTryOnDataset(
    data_root="/path/to/43k/images",
    split="train"
)

print(f"Total samples: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"Person image shape: {sample['person_image'].shape}")
print(f"Garment image shape: {sample['garment_image'].shape}")
print(f"Try-on image shape: {sample['try_on_image'].shape}")
print(f"Person path: {sample['person_path']}")
```

---

## ⚙️ Configuration

Adjust in code:

```python
dataset = FlexibleVirtualTryOnDataset(
    data_root="/path/to/images",
    split="train",
    image_size=256,        # Change from 512 to 256
    augment=True,          # Enable/disable augmentations
    num_samples=43000,     # Total samples (auto-detected)
)
```

---

## 🎯 Full Training Example

```python
import torch
from torch.utils.data import DataLoader
from src.data import FlexibleVirtualTryOnDataset
from src.models import DiffusionVirtualTryOn

# Dataset
train_dataset = FlexibleVirtualTryOnDataset(
    data_root="/path/to/43k/images",
    split="train",
    image_size=512,
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

# Model
model = DiffusionVirtualTryOn(device="cuda")

# Training loop
for epoch in range(5):
    for batch in train_loader:
        person_img = batch["person_image"]
        garment_img = batch["garment_image"]
        try_on_img = batch["try_on_image"]
        
        # Your training code here
        print(f"Batch: {person_img.shape}")
```

---

## ❓ FAQ

**Q: Do I need to organize my images first?**  
A: No! The flexible loader handles any structure.

**Q: What if images have different sizes?**  
A: Automatically resized to 512x512 (configurable).

**Q: How does pairing work?**  
A: Random pairing creates synthetic try-on pairs.

**Q: Can I use this for testing too?**  
A: Yes! Use `split="test"` for test data.

---

## 📝 Next Steps

1. Point to your 43k images folder
2. Run: `python train_simple.py --data_root /path/to/images`
3. Implement your custom loss functions
4. Train your model!

Happy training! 🚀

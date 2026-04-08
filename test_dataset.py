"""Test flexible dataset with user's images"""

from src.data import FlexibleVirtualTryOnDataset
from torch.utils.data import DataLoader

print("=" * 60)
print("Testing FlexibleVirtualTryOnDataset with Your Images")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset from C:\\Users\\chait\\Downloads\\images...")
dataset = FlexibleVirtualTryOnDataset(
    data_root=r"C:\Users\chait\Downloads\images",
    split="train",
    image_size=512,
    augment=True
)

print(f"✓ Dataset loaded!")
print(f"  Total images found in dataset: {len(dataset)}")

# Load a sample
print("\n2. Loading a sample...")
sample = dataset[0]
print(f"✓ Sample loaded!")
print(f"  Person image shape: {sample['person_image'].shape}")
print(f"  Garment image shape: {sample['garment_image'].shape}")
print(f"  Try-on image shape: {sample['try_on_image'].shape}")

# Create dataloader
print("\n3. Creating DataLoader...")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,  # Set to 0 for testing
)

print(f"✓ DataLoader created!")
print(f"  Total batches: {len(loader)}")

# Load a batch
print("\n4. Loading a batch...")
batch = next(iter(loader))
print(f"✓ Batch loaded!")
print(f"  Batch person images: {batch['person_image'].shape}")
print(f"  Batch garment images: {batch['garment_image'].shape}")
print(f"  Batch try-on images: {batch['try_on_image'].shape}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour dataset is ready to use!")
print("Next steps:")
print("  1. Run: python train_simple.py --data_root C:\\Users\\chait\\Downloads\\images")
print("  2. Start training your model!")
print("=" * 60)

"""
YOLOv8-ES Training on Kaggle - Complete Notebook
Copy this entire file into a Kaggle notebook and run!
"""

# ============================================================================
# CELL 1: Install and Setup
# ============================================================================
import os
import sys
from pathlib import Path
import yaml

print("=" * 70)
print("YOLOv8-ES Training on Kaggle")
print("=" * 70)
print()

print("📦 Installing dependencies...")
get_ipython().system('pip install -q ultralytics')

# Setup YOLOv8-ES code
# Option 1: If uploaded as Kaggle dataset
if Path('/kaggle/input/yolov8es-code').exists():
    print("📂 Extracting YOLOv8-ES code from dataset...")
    get_ipython().system('unzip -q /kaggle/input/yolov8es-code/yolov8es.zip -d /kaggle/working/')
    sys.path.insert(0, '/kaggle/working/yolov8es')
# Option 2: If cloning from GitHub (uncomment if needed)
# elif not Path('/kaggle/working/yolov8es').exists():
#     print("📂 Cloning YOLOv8-ES from GitHub...")
#     !git clone https://github.com/yourusername/yolov8es.git /kaggle/working/yolov8es
#     sys.path.insert(0, '/kaggle/working/yolov8es')
else:
    print("⚠️  YOLOv8-ES code not found!")
    print("Please upload yolov8es.zip as a Kaggle dataset")
    print("Or uncomment the GitHub clone option above")

print("✅ Setup complete!")
print()


# ============================================================================
# CELL 2: Verify Installation
# ============================================================================
print("🔍 Verifying YOLOv8-ES modules...")

try:
    from model.edcm import EDCM
    from model.sgam import SGAM
    from model.loss_wiou import WIoUv3Loss
    import torch
    from ultralytics import YOLO
    
    print("✅ EDCM module loaded")
    print("✅ SGAM module loaded")
    print("✅ WIoU v3 module loaded")
    print("✅ PyTorch loaded")
    print("✅ Ultralytics loaded")
    print()
    print("All modules ready!")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure YOLOv8-ES code is properly uploaded")
    raise

print()


# ============================================================================
# CELL 3: Configure Dataset
# ============================================================================
print("⚙️  Configuring dataset...")

# ⚠️ UPDATE THIS PATH TO YOUR KAGGLE DATASET
DATASET_PATH = '/kaggle/input/rdd2022'  # <-- CHANGE THIS!

# Check if dataset exists
if not Path(DATASET_PATH).exists():
    print(f"⚠️  Dataset not found at: {DATASET_PATH}")
    print("Available datasets:")
    get_ipython().system('ls /kaggle/input/')
    print("\nPlease update DATASET_PATH variable above")
else:
    print(f"✅ Dataset found: {DATASET_PATH}")

# Create dataset configuration
dataset_config = {
    'path': DATASET_PATH,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,  # Number of classes
    'names': {
        0: 'D00',  # Longitudinal crack
        1: 'D10',  # Transverse crack
        2: 'D20',  # Alligator crack
        3: 'D40'   # Pothole
    }
}

# Save configuration
config_path = '/kaggle/working/dataset.yaml'
with open(config_path, 'w') as f:
    yaml.dump(dataset_config, f)

print(f"✅ Dataset config saved: {config_path}")
print()


# ============================================================================
# CELL 4: Check GPU
# ============================================================================
print("🖥️  Checking GPU availability...")

import torch

if torch.cuda.is_available():
    print(f"✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = 0
else:
    print("⚠️  No GPU detected!")
    print("   Enable GPU: Settings → Accelerator → GPU T4 x2")
    device = 'cpu'

print()


# ============================================================================
# CELL 5: Train YOLOv8n Baseline
# ============================================================================
print("=" * 70)
print("🚀 Training YOLOv8n Baseline")
print("=" * 70)
print()

from ultralytics import YOLO

# Create model
print("Creating YOLOv8n model...")
model = YOLO('yolov8n.pt')

# Training parameters
train_args = {
    'data': config_path,
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': device,
    'project': '/kaggle/working/runs',
    'name': 'yolov8n_baseline',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'plots': True,
    'save': True,
    'save_period': 10,  # Save checkpoint every 10 epochs
    'val': True,
    'cache': True,  # Cache images for faster training
    'workers': 8,
}

print("Training configuration:")
for key, value in train_args.items():
    print(f"  {key}: {value}")
print()

# Start training
print("Starting training...")
print("This will take a while. Monitor progress below.")
print()

results = model.train(**train_args)

print()
print("=" * 70)
print("✅ Training Complete!")
print("=" * 70)
print(f"Results saved to: {results.save_dir}")
print()


# ============================================================================
# CELL 6: Validate Model
# ============================================================================
print("📊 Validating model...")

metrics = model.val()

print()
print("=" * 70)
print("Validation Results")
print("=" * 70)
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print("=" * 70)
print()


# ============================================================================
# CELL 7: Save Results
# ============================================================================
print("💾 Saving results...")

from shutil import make_archive, copy2

# Copy best weights to easy location
best_weights = '/kaggle/working/runs/yolov8n_baseline/weights/best.pt'
last_weights = '/kaggle/working/runs/yolov8n_baseline/weights/last.pt'

if Path(best_weights).exists():
    copy2(best_weights, '/kaggle/working/best.pt')
    print("✅ Best weights: /kaggle/working/best.pt")

if Path(last_weights).exists():
    copy2(last_weights, '/kaggle/working/last.pt')
    print("✅ Last weights: /kaggle/working/last.pt")

# Create zip of all results
print("Creating results archive...")
make_archive('/kaggle/working/training_results', 'zip', '/kaggle/working/runs')
print("✅ Results archive: /kaggle/working/training_results.zip")

print()
print("=" * 70)
print("📥 Download Files")
print("=" * 70)
print("Go to Output tab (right sidebar) and download:")
print("  • best.pt - Best model weights")
print("  • last.pt - Last checkpoint")
print("  • training_results.zip - All training results")
print("=" * 70)
print()


# ============================================================================
# CELL 8: Test Inference (Optional)
# ============================================================================
print("🔍 Testing inference on validation images...")

# Load best model
model = YOLO('/kaggle/working/best.pt')

# Predict on a few validation images
val_images = f'{DATASET_PATH}/images/val'

if Path(val_images).exists():
    results = model.predict(
        source=val_images,
        save=True,
        conf=0.25,
        iou=0.7,
        max_det=300,
        project='/kaggle/working/predictions',
        name='test',
        exist_ok=True,
    )
    
    print(f"✅ Predictions saved to: /kaggle/working/predictions/test")
    print(f"   Processed {len(results)} images")
else:
    print(f"⚠️  Validation images not found at: {val_images}")

print()
print("=" * 70)
print("🎉 All Done!")
print("=" * 70)
print()
print("Next steps:")
print("1. Download your trained model (best.pt)")
print("2. Download results (training_results.zip)")
print("3. Check predictions in /kaggle/working/predictions/test")
print()
print("To train YOLOv8-ES with EDCM and SGAM, see:")
print("  docs/KAGGLE_TRAINING_GUIDE.md")

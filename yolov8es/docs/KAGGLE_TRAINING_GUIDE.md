# 🚀 Training YOLOv8-ES on Kaggle

Complete guide to train YOLOv8-ES on Kaggle with free GPU.

## 📋 Prerequisites

1. Kaggle account (free)
2. Your dataset uploaded to Kaggle or available as Kaggle dataset
3. YOLOv8-ES code (this repository)

---

## 🎯 Quick Start (5 Steps)

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Settings → Accelerator → Select **"GPU T4 x2"** (free)
4. Settings → Internet → Turn **ON**

### Step 2: Upload YOLOv8-ES Code

**Option A: Upload as Dataset (Recommended)**

1. Zip your `yolov8es` folder:
   ```bash
   zip -r yolov8es.zip yolov8es/
   ```

2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
3. Click **"New Dataset"**
4. Upload `yolov8es.zip`
5. Title: "YOLOv8-ES Code"
6. Click **"Create"**

**Option B: Clone from GitHub**

If you have the code on GitHub:
```python
!git clone https://github.com/yourusername/yolov8es.git
```

### Step 3: Add Dataset

In your Kaggle notebook:
1. Click **"+ Add Data"** (right sidebar)
2. Search for your dataset (e.g., "RDD2022")
3. Click **"Add"**
4. Also add your YOLOv8-ES code dataset

### Step 4: Setup Code

Copy this into your Kaggle notebook:

```python
# Cell 1: Setup
import os
import sys
from pathlib import Path

# Unzip YOLOv8-ES code if uploaded as dataset
!unzip -q /kaggle/input/yolov8es-code/yolov8es.zip -d /kaggle/working/

# Add to path
sys.path.insert(0, '/kaggle/working/yolov8es')

# Install dependencies
!pip install -q ultralytics

print("✅ Setup complete!")
```

```python
# Cell 2: Verify modules
from model.edcm import EDCM
from model.sgam import SGAM
from model.loss_wiou import WIoUv3Loss

print("✅ All modules loaded successfully!")
```

```python
# Cell 3: Prepare dataset config
import yaml

# Update paths for Kaggle
dataset_config = {
    'path': '/kaggle/input/rdd2022',  # Your dataset path
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,
    'names': {
        0: 'D00',
        1: 'D10',
        2: 'D20',
        3: 'D40'
    }
}

# Save config
with open('/kaggle/working/dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

print("✅ Dataset config created!")
```

```python
# Cell 4: Train
from ultralytics import YOLO

# Create model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='/kaggle/working/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,  # Use GPU
    project='/kaggle/working/runs',
    name='yolov8n',
    exist_ok=True,
    plots=True,
    save=True,
)

print("✅ Training complete!")
```

### Step 5: Save Results

```python
# Cell 5: Save trained model
from shutil import make_archive

# Create zip of results
make_archive('/kaggle/working/yolov8n_results', 'zip', '/kaggle/working/runs')

print("✅ Results saved to: /kaggle/working/yolov8n_results.zip")
print("Download from: Output tab → yolov8n_results.zip")
```

---

## 📦 Complete Kaggle Notebook Template

Here's a complete notebook you can copy-paste:

```python
# ============================================================================
# YOLOv8-ES Training on Kaggle
# ============================================================================

# Cell 1: Install and Setup
# ============================================================================
import os
import sys
from pathlib import Path
import yaml

print("Installing dependencies...")
!pip install -q ultralytics

# If you uploaded YOLOv8-ES as a dataset
if Path('/kaggle/input/yolov8es-code').exists():
    print("Extracting YOLOv8-ES code...")
    !unzip -q /kaggle/input/yolov8es-code/yolov8es.zip -d /kaggle/working/
    sys.path.insert(0, '/kaggle/working/yolov8es')
else:
    # Or clone from GitHub
    print("Cloning YOLOv8-ES from GitHub...")
    !git clone https://github.com/yourusername/yolov8es.git /kaggle/working/yolov8es
    sys.path.insert(0, '/kaggle/working/yolov8es')

print("✅ Setup complete!")


# Cell 2: Verify Installation
# ============================================================================
print("Verifying modules...")

try:
    from model.edcm import EDCM
    from model.sgam import SGAM
    from model.loss_wiou import WIoUv3Loss
    print("✅ All YOLOv8-ES modules loaded!")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure YOLOv8-ES code is properly uploaded")


# Cell 3: Configure Dataset
# ============================================================================
print("Configuring dataset...")

# Update this path to match your Kaggle dataset
DATASET_PATH = '/kaggle/input/rdd2022'  # Change this!

dataset_config = {
    'path': DATASET_PATH,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,
    'names': {
        0: 'D00',  # Longitudinal crack
        1: 'D10',  # Transverse crack
        2: 'D20',  # Alligator crack
        3: 'D40'   # Pothole
    }
}

# Save config
config_path = '/kaggle/working/dataset.yaml'
with open(config_path, 'w') as f:
    yaml.dump(dataset_config, f)

print(f"✅ Dataset config saved to: {config_path}")
print(f"Dataset path: {DATASET_PATH}")


# Cell 4: Check GPU
# ============================================================================
import torch

print("Checking GPU availability...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected! Enable GPU in notebook settings.")


# Cell 5: Train Baseline YOLOv8n
# ============================================================================
from ultralytics import YOLO

print("=" * 70)
print("Training YOLOv8n Baseline")
print("=" * 70)

# Create model
model = YOLO('yolov8n.pt')

# Training parameters
train_args = {
    'data': '/kaggle/working/dataset.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 0,  # GPU
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
    'val': True,
    'cache': True,  # Cache images for faster training
}

# Train
results = model.train(**train_args)

print("✅ Training complete!")
print(f"Results: {results.save_dir}")


# Cell 6: Validate Model
# ============================================================================
print("Validating model...")

metrics = model.val()

print("=" * 70)
print("Validation Results")
print("=" * 70)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")


# Cell 7: Save Results
# ============================================================================
from shutil import make_archive
import shutil

print("Saving results...")

# Copy best weights to easy location
best_weights = '/kaggle/working/runs/yolov8n_baseline/weights/best.pt'
shutil.copy(best_weights, '/kaggle/working/best.pt')

# Create zip of all results
make_archive('/kaggle/working/training_results', 'zip', '/kaggle/working/runs')

print("✅ Results saved!")
print("Files available for download:")
print("  - best.pt (best model weights)")
print("  - training_results.zip (all results)")
print("\nDownload from: Output tab (right sidebar)")


# Cell 8: Test Inference (Optional)
# ============================================================================
# Test on a sample image
print("Testing inference...")

# Load best model
model = YOLO('/kaggle/working/best.pt')

# Predict on validation images
results = model.predict(
    source=f'{DATASET_PATH}/images/val',
    save=True,
    conf=0.25,
    project='/kaggle/working/predictions',
    name='test'
)

print(f"✅ Predictions saved to: /kaggle/working/predictions/test")
```

---

## 🔧 Training YOLOv8-ES (with EDCM + SGAM)

To train the full YOLOv8-ES with custom modules:

```python
# Cell: Train YOLOv8-ES
# ============================================================================
import sys
sys.path.insert(0, '/kaggle/working/yolov8es')

from scripts.train_yolov8es import build_yolov8es_model
from ultralytics import YOLO

print("Building YOLOv8-ES model...")

# Build model with EDCM and SGAM
model = build_yolov8es_model(base_model='yolov8n.pt', nc=4)

print("Training YOLOv8-ES...")

# Train
results = model.train(
    data='/kaggle/working/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    project='/kaggle/working/runs',
    name='yolov8es',
    exist_ok=True,
    plots=True,
    save=True,
)

print("✅ YOLOv8-ES training complete!")
```

---

## 📊 Kaggle-Specific Tips

### 1. GPU Selection
- **T4 x2** (free): Good for training, 16GB memory
- **P100** (free): Faster, 16GB memory
- Enable in: Settings → Accelerator

### 2. Session Limits
- Free tier: 30 hours/week GPU time
- Sessions timeout after 12 hours
- Save checkpoints regularly!

### 3. Dataset Paths
Kaggle datasets are mounted at:
```
/kaggle/input/your-dataset-name/
```

### 4. Output Files
Save to `/kaggle/working/` to download:
```python
# Files here can be downloaded
/kaggle/working/best.pt
/kaggle/working/results.zip
```

### 5. Faster Training
```python
# Enable image caching
cache=True

# Use smaller batch if OOM
batch=8

# Reduce image size
imgsz=512
```

### 6. Save Checkpoints
```python
# Auto-save every N epochs
save_period=10  # Save every 10 epochs
```

---

## 🐛 Common Issues

### Issue 1: "No module named 'model'"
**Solution:**
```python
import sys
sys.path.insert(0, '/kaggle/working/yolov8es')
```

### Issue 2: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
batch=8  # or even 4

# Or reduce image size
imgsz=512
```

### Issue 3: "Dataset not found"
**Solution:**
```python
# Check dataset path
!ls /kaggle/input/

# Update path in dataset.yaml
path: '/kaggle/input/your-dataset-name'
```

### Issue 4: Session timeout
**Solution:**
```python
# Enable auto-save
save_period=10

# Resume training
model = YOLO('/kaggle/working/runs/yolov8n/weights/last.pt')
model.train(resume=True)
```

---

## 📥 Download Results

After training:

1. Go to **Output** tab (right sidebar)
2. Download files:
   - `best.pt` - Best model weights
   - `training_results.zip` - All results
3. Or use Kaggle API:
   ```python
   from kaggle import api
   api.dataset_create_version(...)
   ```

---

## 🎯 Complete Workflow

```
1. Create Kaggle Notebook
   ↓
2. Enable GPU (T4 x2)
   ↓
3. Add YOLOv8-ES code dataset
   ↓
4. Add your training dataset
   ↓
5. Run setup cells
   ↓
6. Configure dataset.yaml
   ↓
7. Train model
   ↓
8. Download results
```

---

## 📝 Example Notebook Structure

```
Cell 1: Install & Setup
Cell 2: Verify Modules
Cell 3: Configure Dataset
Cell 4: Check GPU
Cell 5: Train Model
Cell 6: Validate
Cell 7: Save Results
Cell 8: Test Inference (optional)
```

---

## 🚀 Quick Commands

```python
# Check GPU
!nvidia-smi

# Check dataset
!ls /kaggle/input/

# Monitor training
!tail -f /kaggle/working/runs/yolov8n/train.log

# Check disk space
!df -h
```

---

## ✅ Checklist

Before training:
- [ ] GPU enabled (T4 x2)
- [ ] Internet enabled
- [ ] YOLOv8-ES code uploaded
- [ ] Dataset added
- [ ] Dataset path updated in config
- [ ] Modules verified

During training:
- [ ] Monitor GPU usage
- [ ] Check training plots
- [ ] Save checkpoints

After training:
- [ ] Download best.pt
- [ ] Download results.zip
- [ ] Save notebook

---

## 🎓 Resources

- **Kaggle Docs**: https://www.kaggle.com/docs
- **Ultralytics Docs**: https://docs.ultralytics.com
- **YOLOv8-ES Paper**: See `papers/` folder

---

## 💡 Pro Tips

1. **Use Kaggle Datasets**: Upload your data as a Kaggle dataset for faster loading
2. **Enable Caching**: `cache=True` speeds up training significantly
3. **Save Often**: Use `save_period=10` to save checkpoints
4. **Monitor GPU**: Use `!nvidia-smi` to check GPU usage
5. **Version Control**: Save notebook versions regularly

---

**Ready to train on Kaggle!** 🚀

Just copy the complete notebook template above and run it!

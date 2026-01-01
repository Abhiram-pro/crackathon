# 🚀 Kaggle Setup - Quick Guide

## Step-by-Step Setup

### 1️⃣ Prepare Your Code

On your local machine:

```bash
# Zip the yolov8es folder
cd /path/to/your/project
zip -r yolov8es.zip yolov8es/
```

### 2️⃣ Upload to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"**
3. Upload `yolov8es.zip`
4. Title: "YOLOv8-ES Code"
5. Click **"Create"**

### 3️⃣ Create Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Settings:
   - Accelerator: **GPU T4 x2**
   - Internet: **ON**
   - Persistence: **Files only**

### 4️⃣ Add Data

In your notebook:
1. Click **"+ Add Data"** (right sidebar)
2. Add your datasets:
   - Your training dataset (e.g., RDD2022)
   - YOLOv8-ES Code dataset (from step 2)

### 5️⃣ Copy Notebook Code

Copy the entire content from `kaggle_notebook.py` into your Kaggle notebook.

Or use this minimal version:

```python
# Minimal Kaggle Training Script
import sys
from pathlib import Path

# Install
!pip install -q ultralytics

# Extract YOLOv8-ES code
!unzip -q /kaggle/input/yolov8es-code/yolov8es.zip -d /kaggle/working/
sys.path.insert(0, '/kaggle/working/yolov8es')

# Create dataset config
import yaml
config = {
    'path': '/kaggle/input/your-dataset',  # UPDATE THIS!
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,
    'names': {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}
}
with open('/kaggle/working/data.yaml', 'w') as f:
    yaml.dump(config, f)

# Train
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=100,
    batch=16,
    device=0,
    project='/kaggle/working/runs',
    cache=True
)

# Save
from shutil import copy2
copy2('/kaggle/working/runs/train/weights/best.pt', '/kaggle/working/best.pt')
print("✅ Done! Download best.pt from Output tab")
```

### 6️⃣ Run Training

1. Click **"Run All"** or run cells one by one
2. Monitor training progress
3. Wait for completion (~2-4 hours for 100 epochs)

### 7️⃣ Download Results

1. Go to **Output** tab (right sidebar)
2. Download:
   - `best.pt` - Your trained model
   - `training_results.zip` - All results

---

## 📋 Quick Checklist

Before running:
- [ ] GPU enabled (T4 x2)
- [ ] Internet enabled
- [ ] YOLOv8-ES code uploaded as dataset
- [ ] Training dataset added
- [ ] Dataset path updated in code

---

## 🎯 What You Need

### Required Files
1. `yolov8es.zip` - Your code (upload as Kaggle dataset)
2. Your training dataset (upload as Kaggle dataset or use existing)

### Kaggle Settings
- **Accelerator**: GPU T4 x2 (free)
- **Internet**: ON
- **Session**: Can run up to 12 hours

---

## 💡 Tips

1. **Test First**: Run with `epochs=1` to test everything works
2. **Save Often**: Use `save_period=10` in training args
3. **Monitor**: Check GPU usage with `!nvidia-smi`
4. **Cache**: Use `cache=True` for faster training
5. **Batch Size**: Reduce to 8 if you get OOM errors

---

## 🐛 Troubleshooting

### "Dataset not found"
```python
# Check available datasets
!ls /kaggle/input/

# Update path
path: '/kaggle/input/your-actual-dataset-name'
```

### "No module named 'model'"
```python
# Add to path
import sys
sys.path.insert(0, '/kaggle/working/yolov8es')
```

### "CUDA out of memory"
```python
# Reduce batch size
batch=8  # or 4
```

### "Session timeout"
```python
# Resume from checkpoint
model = YOLO('/kaggle/working/runs/train/weights/last.pt')
model.train(resume=True)
```

---

## 📚 Full Documentation

For complete guide, see: **[docs/KAGGLE_TRAINING_GUIDE.md](docs/KAGGLE_TRAINING_GUIDE.md)**

---

## ✅ Ready!

1. Zip your code: `zip -r yolov8es.zip yolov8es/`
2. Upload to Kaggle as dataset
3. Create notebook with GPU
4. Copy code from `kaggle_notebook.py`
5. Run and train!

**That's it!** 🎉

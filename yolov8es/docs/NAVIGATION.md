# 🧭 YOLOv8-ES Navigation Guide

## 🚀 Quick Start (3 Steps)

1. **Read**: [docs/START_HERE.md](docs/START_HERE.md)
2. **Check**: `python tests/test_training_ready.py`
3. **Train**: `python scripts/simple_train.py --data configs/rdd2022.yaml`

---

## 📂 Directory Guide

### 📦 [model/](model/) - Core Implementation
- `edcm.py` - Enhanced Dynamic Convolution Module
- `sgam.py` - Selective Global Attention Mechanism
- `loss_wiou.py` - Wise-IoU v3 Loss Function

### 🚀 [scripts/](scripts/) - Training & Inference
- `simple_train.py` ⭐ - Start here for training
- `train_yolov8es.py` - Full YOLOv8-ES with EDCM + SGAM
- `predict.py` - Inference and validation

### ⚙️ [configs/](configs/) - Configuration
- `rdd2022.yaml` ⚠️ - Edit this! Update dataset path
- `yolov8es.yaml` - Model architecture

### 🧪 [tests/](tests/) - Verification
- `test_training_ready.py` ⭐ - Run this first
- `run_all_tests.py` - Run all module tests
- `verify_*.py` - Individual module tests

### 📚 [docs/](docs/) - Documentation
- `START_HERE.md` ⭐ - Quick start guide
- `HOW_TO_TRAIN.txt` - Simple instructions
- `TRAINING_GUIDE.md` - Comprehensive guide
- `INDEX.md` - Full navigation index
- `PROJECT_STRUCTURE.md` - Detailed file descriptions
- `ORGANIZATION_COMPLETE.md` - Organization summary

### 📄 [papers/](papers/) - Research
- `s43684-025-00091-3.pdf` - Original paper

---

## 🎯 Common Tasks

### First Time Setup
```bash
# 1. Read the guide
cat docs/START_HERE.md

# 2. Check readiness
python tests/test_training_ready.py

# 3. Edit dataset config
nano configs/rdd2022.yaml  # Update 'path:' line

# 4. Train
python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 100
```

### Run Tests
```bash
# All tests
python tests/run_all_tests.py

# Individual tests
python tests/verify_edcm.py
python tests/verify_sgam.py
python tests/verify_wiou.py
```

### Training
```bash
# Baseline YOLOv8n
python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 100

# Full YOLOv8-ES (with EDCM + SGAM)
python scripts/train_yolov8es.py --data configs/rdd2022.yaml --epochs 100
```

### Inference
```bash
# Validate model
python scripts/predict.py val \
  --weights runs/train/yolov8n/weights/best.pt \
  --data configs/rdd2022.yaml

# Predict on image
python scripts/predict.py predict \
  --weights runs/train/yolov8n/weights/best.pt \
  --source path/to/image.jpg
```

---

## 📖 Documentation Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [README.md](README.md) | Project overview | First |
| [docs/START_HERE.md](docs/START_HERE.md) | Quick start | Before training |
| [docs/HOW_TO_TRAIN.txt](docs/HOW_TO_TRAIN.txt) | Simple guide | Quick reference |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Detailed guide | Deep dive |
| [docs/INDEX.md](docs/INDEX.md) | Full navigation | Find anything |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | File details | Understanding code |
| [docs/VERIFICATION_SUMMARY.md](docs/VERIFICATION_SUMMARY.md) | Module status | Technical details |
| [docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md) | Integration info | Advanced usage |

---

## 🔍 Find What You Need

| I want to... | Go to... |
|--------------|----------|
| Start training now | [docs/START_HERE.md](docs/START_HERE.md) |
| Understand the project | [README.md](README.md) |
| See all files | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |
| Check if ready | `python tests/test_training_ready.py` |
| Train baseline | `python scripts/simple_train.py` |
| Train YOLOv8-ES | `python scripts/train_yolov8es.py` |
| Run tests | `python tests/run_all_tests.py` |
| Read the paper | [papers/s43684-025-00091-3.pdf](papers/s43684-025-00091-3.pdf) |
| Get help | [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) |

---

## 📊 Project Status

| Component | Status | Location |
|-----------|--------|----------|
| EDCM | ✅ Complete | [model/edcm.py](model/edcm.py) |
| SGAM | ✅ Complete | [model/sgam.py](model/sgam.py) |
| WIoU v3 | ✅ Complete | [model/loss_wiou.py](model/loss_wiou.py) |
| Training | ✅ Ready | [scripts/](scripts/) |
| Tests | ✅ Passing | [tests/](tests/) |
| Docs | ✅ Complete | [docs/](docs/) |

---

## 💡 Tips

- **New to the project?** Start with [docs/START_HERE.md](docs/START_HERE.md)
- **Want to train?** Run `python tests/test_training_ready.py` first
- **Need help?** Check [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- **Looking for something?** Use [docs/INDEX.md](docs/INDEX.md)

---

## ✅ Everything is Ready!

All modules are implemented, tested, and organized.

**Next step:** Update `configs/rdd2022.yaml` and start training!

```bash
python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 100
```

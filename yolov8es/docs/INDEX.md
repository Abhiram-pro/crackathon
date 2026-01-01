# YOLOv8-ES Project Index

## 📂 Quick Navigation

### 🚀 Getting Started
1. **[README.md](README.md)** - Project overview
2. **[docs/START_HERE.md](docs/START_HERE.md)** - Quick start (3 steps)
3. **[docs/HOW_TO_TRAIN.txt](docs/HOW_TO_TRAIN.txt)** - Simple instructions
4. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - File organization

### 🎯 Training
- **[scripts/simple_train.py](scripts/simple_train.py)** - Start here
- **[scripts/train_yolov8es.py](scripts/train_yolov8es.py)** - Full YOLOv8-ES
- **[scripts/predict.py](scripts/predict.py)** - Inference & validation
- **[configs/rdd2022.yaml](configs/rdd2022.yaml)** - Dataset config (edit this!)

### 🧪 Testing
- **[tests/test_training_ready.py](tests/test_training_ready.py)** - Pre-training check
- **[tests/run_all_tests.py](tests/run_all_tests.py)** - Run all tests
- **[tests/verify_edcm.py](tests/verify_edcm.py)** - EDCM tests
- **[tests/verify_sgam.py](tests/verify_sgam.py)** - SGAM tests
- **[tests/verify_wiou.py](tests/verify_wiou.py)** - WIoU tests

### 📦 Core Modules
- **[model/edcm.py](model/edcm.py)** - Enhanced Dynamic Convolution
- **[model/sgam.py](model/sgam.py)** - Selective Global Attention
- **[model/loss_wiou.py](model/loss_wiou.py)** - Wise-IoU v3 Loss

### 📚 Documentation
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Comprehensive guide
- **[docs/VERIFICATION_SUMMARY.md](docs/VERIFICATION_SUMMARY.md)** - Module verification
- **[docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md)** - Integration details
- **[docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md)** - Development log

### 📄 Research
- **[papers/s43684-025-00091-3.pdf](papers/s43684-025-00091-3.pdf)** - Original paper

## 🎬 Quick Start Commands

```bash
# 1. Check readiness
python tests/test_training_ready.py

# 2. Update dataset path in configs/rdd2022.yaml

# 3. Train
python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 100
```

## 📊 Project Status

| Component | Status | File |
|-----------|--------|------|
| EDCM | ✅ Complete | `model/edcm.py` |
| SGAM | ✅ Complete | `model/sgam.py` |
| WIoU v3 | ✅ Complete | `model/loss_wiou.py` |
| Training | ✅ Ready | `scripts/` |
| Tests | ✅ Passing | `tests/` |
| Docs | ✅ Complete | `docs/` |

## 🗂️ Directory Structure

```
yolov8es/
├── 📦 model/          Core modules (EDCM, SGAM, WIoU)
├── 🚀 scripts/        Training & inference scripts
├── ⚙️  configs/        Configuration files
├── 🧪 tests/          Verification tests
├── 📚 docs/           Documentation
└── 📄 papers/         Research papers
```

## 🎯 Recommended Reading Order

1. **[README.md](README.md)** - Overview
2. **[docs/START_HERE.md](docs/START_HERE.md)** - Quick start
3. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - File organization
4. **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Detailed guide

## 💡 Common Tasks

### First Time Setup
→ Read: `docs/START_HERE.md`  
→ Check: `python tests/test_training_ready.py`  
→ Edit: `configs/rdd2022.yaml`  
→ Train: `python scripts/simple_train.py`

### Run Tests
→ All: `python tests/run_all_tests.py`  
→ EDCM: `python tests/verify_edcm.py`  
→ SGAM: `python tests/verify_sgam.py`  
→ WIoU: `python tests/verify_wiou.py`

### Training
→ Baseline: `python scripts/simple_train.py --data configs/rdd2022.yaml`  
→ YOLOv8-ES: `python scripts/train_yolov8es.py --data configs/rdd2022.yaml`

### Inference
→ Validate: `python scripts/predict.py val --weights best.pt --data configs/rdd2022.yaml`  
→ Predict: `python scripts/predict.py predict --weights best.pt --source image.jpg`

## 🔍 Find What You Need

| I want to... | Go to... |
|--------------|----------|
| Start training quickly | `docs/START_HERE.md` |
| Understand the project | `README.md` |
| See all files | `PROJECT_STRUCTURE.md` |
| Train baseline | `scripts/simple_train.py` |
| Train YOLOv8-ES | `scripts/train_yolov8es.py` |
| Run tests | `tests/run_all_tests.py` |
| Check modules | `docs/VERIFICATION_SUMMARY.md` |
| Learn training options | `docs/TRAINING_GUIDE.md` |
| Understand integration | `docs/INTEGRATION_STATUS.md` |
| Read the paper | `papers/s43684-025-00091-3.pdf` |

## ✅ Everything is Ready!

All modules are implemented, verified, and organized.  
Just update `configs/rdd2022.yaml` and start training!

**Next:** `python tests/test_training_ready.py`

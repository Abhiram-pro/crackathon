# YOLOv8-ES Project Structure

## Directory Organization

```
yolov8es/
│
├── 📦 model/                          Core Implementation
│   ├── __init__.py                   Package initialization
│   ├── edcm.py                       Enhanced Dynamic Convolution Module
│   ├── sgam.py                       Selective Global Attention Mechanism
│   ├── loss_wiou.py                  Wise-IoU v3 Loss Function
│   └── yolov8es_model.py            Model integration utilities
│
├── 🚀 scripts/                        Training & Inference
│   ├── simple_train.py               Quick training script (START HERE)
│   ├── train_yolov8es.py            Full YOLOv8-ES training
│   ├── train.py                      Advanced training with options
│   └── predict.py                    Inference and validation
│
├── ⚙️  configs/                       Configuration Files
│   ├── rdd2022.yaml                  Dataset configuration
│   └── yolov8es.yaml                 Model architecture (YAML)
│
├── 🧪 tests/                          Verification & Testing
│   ├── test_training_ready.py        Pre-training readiness check
│   ├── verify_edcm.py                EDCM module tests
│   ├── verify_sgam.py                SGAM module tests
│   ├── verify_sgam_detailed.py       SGAM detailed tests
│   ├── verify_wiou.py                WIoU v3 tests
│   ├── verify_wiou_detailed.py       WIoU v3 detailed tests
│   ├── run_all_tests.py              Run all verification tests
│   └── test_model.py                 Model creation test
│
├── 📚 docs/                           Documentation
│   ├── START_HERE.md                 Quick start guide (READ FIRST)
│   ├── HOW_TO_TRAIN.txt              Simple training instructions
│   ├── TRAINING_GUIDE.md             Comprehensive training guide
│   ├── QUICK_START.md                Quick reference
│   ├── VERIFICATION_SUMMARY.md       Module verification details
│   ├── INTEGRATION_STATUS.md         Integration status & options
│   └── IMPLEMENTATION_LOG.md         Development log
│
├── 📄 papers/                         Research Papers
│   ├── s43684-025-00091-3.pdf        Original paper (PDF)
│   └── s43684-025-00091-3.png        Paper figure
│
├── 📋 README.md                       Main project README
├── 📋 PROJECT_STRUCTURE.md            This file
└── 📋 __init__.py                     Package initialization
```

## File Descriptions

### Core Modules (`model/`)

#### `edcm.py`
- **Purpose**: Enhanced Dynamic Convolution Module
- **Paper Section**: 3.2
- **Features**: ODConv + PSA, stride=1, dynamic kernels
- **Usage**: `from model.edcm import EDCM`
- **Status**: ✅ Verified

#### `sgam.py`
- **Purpose**: Selective Global Attention Mechanism
- **Paper Section**: 3.3
- **Features**: SE → GAM → CA sequential attention
- **Usage**: `from model.sgam import SGAM`
- **Status**: ✅ Verified

#### `loss_wiou.py`
- **Purpose**: Wise-IoU v3 Loss Function
- **Paper Section**: 3.4
- **Features**: Non-monotonic focusing, dynamic gradient allocation
- **Usage**: `from model.loss_wiou import WIoUv3Loss`
- **Status**: ✅ Verified

### Training Scripts (`scripts/`)

#### `simple_train.py` ⭐ START HERE
- **Purpose**: Simplest training script
- **What it does**: Trains baseline YOLOv8n
- **When to use**: First time, testing setup
- **Command**: `python scripts/simple_train.py --data configs/rdd2022.yaml`

#### `train_yolov8es.py`
- **Purpose**: Full YOLOv8-ES training
- **What it does**: Adds EDCM and SGAM to YOLOv8n
- **When to use**: After baseline works
- **Command**: `python scripts/train_yolov8es.py --data configs/rdd2022.yaml`

#### `train.py`
- **Purpose**: Advanced training with all options
- **What it does**: Full control over training parameters
- **When to use**: Custom experiments
- **Command**: `python scripts/train.py --data configs/rdd2022.yaml [options]`

#### `predict.py`
- **Purpose**: Inference and validation
- **What it does**: Run predictions or validate model
- **Commands**:
  - Predict: `python scripts/predict.py predict --weights best.pt --source image.jpg`
  - Validate: `python scripts/predict.py val --weights best.pt --data rdd2022.yaml`

### Configuration Files (`configs/`)

#### `rdd2022.yaml`
- **Purpose**: Dataset configuration
- **What to edit**: Update `path:` to your dataset location
- **Format**: YOLO dataset format
- **Classes**: 4 (D00, D10, D20, D40)

#### `yolov8es.yaml`
- **Purpose**: Model architecture definition
- **Status**: ⚠️ YAML parser integration in progress
- **Note**: Use training scripts instead for now

### Tests (`tests/`)

#### `test_training_ready.py` ⭐ RUN FIRST
- **Purpose**: Check if everything is ready
- **What it checks**: Packages, modules, scripts, GPU
- **Command**: `python tests/test_training_ready.py`
- **When**: Before training

#### `verify_*.py`
- **Purpose**: Module verification tests
- **What they test**: Functionality, shapes, gradients, edge cases
- **Command**: `python tests/verify_edcm.py` (or sgam, wiou)
- **When**: After code changes

#### `run_all_tests.py`
- **Purpose**: Run all verification tests
- **Command**: `python tests/run_all_tests.py`
- **When**: Complete verification

### Documentation (`docs/`)

#### `START_HERE.md` ⭐ READ FIRST
- Quick start guide
- 3-step training process
- Common issues

#### `HOW_TO_TRAIN.txt`
- Simple text instructions
- Copy-paste commands
- Troubleshooting

#### `TRAINING_GUIDE.md`
- Comprehensive training guide
- All parameters explained
- Advanced options

#### `VERIFICATION_SUMMARY.md`
- Module verification details
- Test results
- Implementation status

#### `INTEGRATION_STATUS.md`
- Current integration status
- Available options
- Next steps

## Workflow

### First Time Setup

1. **Read**: `docs/START_HERE.md`
2. **Check**: `python tests/test_training_ready.py`
3. **Configure**: Edit `configs/rdd2022.yaml`
4. **Train**: `python scripts/simple_train.py --data configs/rdd2022.yaml`

### Development Workflow

1. **Verify modules**: `python tests/run_all_tests.py`
2. **Train baseline**: `python scripts/simple_train.py`
3. **Train YOLOv8-ES**: `python scripts/train_yolov8es.py`
4. **Validate**: `python scripts/predict.py val`
5. **Inference**: `python scripts/predict.py predict`

### File Dependencies

```
Training Scripts depend on:
  ├── model/edcm.py
  ├── model/sgam.py
  ├── model/loss_wiou.py
  └── configs/rdd2022.yaml

Tests depend on:
  ├── model/edcm.py
  ├── model/sgam.py
  └── model/loss_wiou.py

Documentation:
  └── Standalone (no dependencies)
```

## Quick Commands

```bash
# Check readiness
python tests/test_training_ready.py

# Run all tests
python tests/run_all_tests.py

# Train baseline
python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 100

# Train YOLOv8-ES
python scripts/train_yolov8es.py --data configs/rdd2022.yaml --epochs 100

# Validate
python scripts/predict.py val --weights runs/train/yolov8n/weights/best.pt --data configs/rdd2022.yaml

# Predict
python scripts/predict.py predict --weights runs/train/yolov8n/weights/best.pt --source image.jpg
```

## Status Summary

| Component | Status | Location |
|-----------|--------|----------|
| EDCM | ✅ Complete | `model/edcm.py` |
| SGAM | ✅ Complete | `model/sgam.py` |
| WIoU v3 | ✅ Complete | `model/loss_wiou.py` |
| Training Scripts | ✅ Ready | `scripts/` |
| Tests | ✅ Passing | `tests/` |
| Documentation | ✅ Complete | `docs/` |
| YAML Integration | ⚠️ In Progress | `configs/yolov8es.yaml` |

## Next Steps

1. ✅ Update `configs/rdd2022.yaml` with your dataset path
2. ✅ Run `python tests/test_training_ready.py`
3. ✅ Start training with `python scripts/simple_train.py`

Everything is organized and ready to use!

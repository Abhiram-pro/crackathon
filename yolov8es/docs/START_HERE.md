# Start Training YOLOv8-ES

## Quick Start (3 Steps)

### Step 1: Prepare Your Dataset

Organize your dataset in YOLO format:

```
datasets/rdd2022/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

Each label file contains: `class x_center y_center width height` (normalized 0-1)

### Step 2: Update Dataset Config

Edit `rdd2022.yaml`:

```yaml
path: /path/to/your/datasets/rdd2022  # <-- Change this
train: images/train
val: images/val
nc: 4
names:
  0: D00
  1: D10
  2: D20
  3: D40
```

### Step 3: Train!

**Option A - Simple baseline first (recommended):**

```bash
python simple_train.py --data rdd2022.yaml --epochs 100 --batch 16
```

This trains standard YOLOv8n to verify everything works.

**Option B - Train with EDCM and SGAM:**

```bash
python train_yolov8es.py --data rdd2022.yaml --epochs 100 --batch 16
```

This adds EDCM (backbone) and SGAM (neck) modules to YOLOv8n.

## That's It!

Training will start and save results to `runs/train/yolov8n/` or `runs/train/yolov8es/`

## Common Issues

### "No module named ultralytics"
```bash
pip install ultralytics
```

### "Dataset not found"
Update the `path:` in `rdd2022.yaml` to your actual dataset location.

### "CUDA out of memory"
Reduce batch size:
```bash
python simple_train.py --data rdd2022.yaml --batch 8
```

### "No GPU detected"
Training will use CPU automatically (slower). To use GPU:
```bash
python simple_train.py --data rdd2022.yaml --device 0
```

## What Gets Trained

### simple_train.py
- Standard YOLOv8n
- Good for baseline comparison
- Fastest to train

### train_yolov8es.py
- YOLOv8n + EDCM (backbone)
- YOLOv8n + SGAM (neck)
- Paper-accurate YOLOv8-ES
- Slightly slower but better accuracy

## After Training

### Validate
```bash
python predict.py val --weights runs/train/yolov8n/weights/best.pt --data rdd2022.yaml
```

### Inference
```bash
python predict.py predict --weights runs/train/yolov8n/weights/best.pt --source path/to/image.jpg
```

## Training Tips

1. **Start small**: Train for 10 epochs first to verify everything works
   ```bash
   python simple_train.py --data rdd2022.yaml --epochs 10
   ```

2. **Monitor training**: Results are saved with plots showing loss curves

3. **Use GPU**: Training is much faster with GPU
   ```bash
   python simple_train.py --data rdd2022.yaml --device 0
   ```

4. **Adjust batch size**: Based on your GPU memory
   - 16GB GPU: batch=16
   - 8GB GPU: batch=8
   - 4GB GPU: batch=4

## Expected Results

From the paper (RDD2022 dataset):
- YOLOv8n baseline: ~65% mAP50
- YOLOv8-ES: ~70% mAP50

Your results may vary based on:
- Dataset quality
- Training epochs
- Hyperparameters
- Hardware

## Need Help?

1. Check `TRAINING_GUIDE.md` for detailed options
2. Check `INTEGRATION_STATUS.md` for technical details
3. All modules are verified - see `VERIFICATION_SUMMARY.md`

## Files You Need

- ✅ `simple_train.py` - Start here
- ✅ `train_yolov8es.py` - Full YOLOv8-ES
- ✅ `predict.py` - Inference/validation
- ✅ `rdd2022.yaml` - Dataset config (update path!)
- ✅ `model/edcm.py` - EDCM module
- ✅ `model/sgam.py` - SGAM module
- ✅ `model/loss_wiou.py` - WIoU v3 loss

Everything is ready. Just update `rdd2022.yaml` and run `simple_train.py`!

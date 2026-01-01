# YOLOv8-ES Training Guide

Complete guide for training YOLOv8-ES on road crack detection datasets.

## Quick Start

### 1. Prepare Dataset

Organize your dataset in YOLO format:

```
datasets/rdd2022/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

Label format (YOLO): `class x_center y_center width height` (normalized 0-1)

### 2. Update Dataset Configuration

Edit `rdd2022.yaml`:

```yaml
path: ../datasets/rdd2022  # Update with your path
train: images/train
val: images/val
nc: 4
names:
  0: D00  # Longitudinal crack
  1: D10  # Transverse crack
  2: D20  # Alligator crack
  3: D40  # Pothole
```

### 3. Train Model

```bash
# Basic training
python train.py --data rdd2022.yaml --epochs 100 --batch 16

# With GPU
python train.py --data rdd2022.yaml --epochs 100 --batch 16 --device 0

# Multi-GPU
python train.py --data rdd2022.yaml --epochs 100 --batch 32 --device 0,1

# Resume training
python train.py --data rdd2022.yaml --epochs 100 --batch 16 --pretrained runs/train/yolov8es/weights/last.pt
```

## Training Options

### Basic Parameters

```bash
python train.py \
  --data rdd2022.yaml \        # Dataset configuration
  --epochs 100 \               # Number of epochs
  --batch 16 \                 # Batch size
  --imgsz 640 \                # Image size
  --device 0 \                 # GPU device (0, 0,1, or cpu)
  --project runs/train \       # Save directory
  --name yolov8es \            # Experiment name
  --workers 8 \                # Number of dataloader workers
  --cache                      # Cache images for faster training
```

### Advanced Parameters

The training script supports all Ultralytics YOLO training parameters:

- **Optimizer**: `--optimizer SGD` (SGD, Adam, AdamW)
- **Learning rate**: `--lr0 0.01` (initial), `--lrf 0.01` (final)
- **Momentum**: `--momentum 0.937`
- **Weight decay**: `--weight_decay 0.0005`
- **Warmup**: `--warmup_epochs 3.0`
- **Loss weights**: `--box 7.5 --cls 0.5 --dfl 1.5`

### Data Augmentation

```bash
python train.py \
  --data rdd2022.yaml \
  --epochs 100 \
  --hsv_h 0.015 \              # HSV-Hue augmentation
  --hsv_s 0.7 \                # HSV-Saturation
  --hsv_v 0.4 \                # HSV-Value
  --degrees 0.0 \              # Rotation
  --translate 0.1 \            # Translation
  --scale 0.5 \                # Scale
  --fliplr 0.5 \               # Horizontal flip
  --mosaic 1.0 \               # Mosaic augmentation
  --mixup 0.0                  # Mixup augmentation
```

## Validation

### Validate Trained Model

```bash
python predict.py val \
  --weights runs/train/yolov8es/weights/best.pt \
  --data rdd2022.yaml \
  --batch 16 \
  --imgsz 640
```

### Metrics

The validation will output:
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Precision score
- **Recall**: Recall score
- **Per-class metrics**: Individual class performance

## Inference

### Predict on Images

```bash
# Single image
python predict.py predict \
  --weights runs/train/yolov8es/weights/best.pt \
  --source path/to/image.jpg \
  --conf 0.25 \
  --save

# Directory of images
python predict.py predict \
  --weights runs/train/yolov8es/weights/best.pt \
  --source path/to/images/ \
  --conf 0.25 \
  --save

# Video
python predict.py predict \
  --weights runs/train/yolov8es/weights/best.pt \
  --source path/to/video.mp4 \
  --conf 0.25 \
  --save
```

### Inference Options

```bash
python predict.py predict \
  --weights best.pt \
  --source image.jpg \
  --conf 0.25 \                # Confidence threshold
  --iou 0.7 \                  # IoU threshold for NMS
  --imgsz 640 \                # Image size
  --device 0 \                 # GPU device
  --save \                     # Save results
  --save-txt \                 # Save as txt files
  --project runs/predict \     # Save directory
  --name yolov8es              # Experiment name
```

## Python API

### Training

```python
from train import train_yolov8es

results = train_yolov8es(
    data='rdd2022.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='0',
    project='runs/train',
    name='yolov8es'
)
```

### Validation

```python
from predict import validate_yolov8es

metrics = validate_yolov8es(
    weights='runs/train/yolov8es/weights/best.pt',
    data='rdd2022.yaml',
    batch=16,
    imgsz=640
)

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### Inference

```python
from predict import predict_yolov8es

results = predict_yolov8es(
    weights='runs/train/yolov8es/weights/best.pt',
    source='path/to/image.jpg',
    conf=0.25,
    save=True
)

# Access results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls}, Conf: {conf:.3f}, Box: {xyxy}")
```

## Transfer Learning

### From YOLOv8n Pretrained

```bash
# Download YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Train with pretrained weights
python train.py \
  --data rdd2022.yaml \
  --epochs 100 \
  --batch 16 \
  --pretrained yolov8n.pt
```

### Fine-tuning

```bash
# Fine-tune on new data
python train.py \
  --data new_dataset.yaml \
  --epochs 50 \
  --batch 16 \
  --pretrained runs/train/yolov8es/weights/best.pt \
  --lr0 0.001  # Lower learning rate for fine-tuning
```

## Hyperparameter Tuning

### Recommended Settings for Road Crack Detection

Based on the paper:

```bash
python train.py \
  --data rdd2022.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --optimizer SGD \
  --lr0 0.01 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight_decay 0.0005 \
  --warmup_epochs 3.0 \
  --box 7.5 \                  # WIoU-v3 weight
  --cls 0.5 \
  --dfl 1.5 \
  --hsv_h 0.015 \
  --hsv_s 0.7 \
  --hsv_v 0.4 \
  --fliplr 0.5 \
  --mosaic 1.0
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/train

# View in browser
# http://localhost:6006
```

### Training Outputs

Training saves:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.csv` - Training metrics
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `val_batch*.jpg` - Validation predictions

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py --batch 8

# Reduce image size
python train.py --imgsz 512

# Use mixed precision (automatic in PyTorch 2.0+)
```

### Slow Training

```bash
# Enable image caching
python train.py --cache

# Increase workers
python train.py --workers 16

# Use smaller image size
python train.py --imgsz 512
```

### Poor Performance

1. **Check dataset quality**
   - Verify labels are correct
   - Ensure sufficient training data
   - Balance class distribution

2. **Adjust hyperparameters**
   - Increase epochs
   - Tune learning rate
   - Adjust augmentation

3. **Use pretrained weights**
   - Start from YOLOv8n pretrained
   - Fine-tune on your dataset

## Performance Benchmarks

Expected performance on RDD2022 (from paper):

| Metric | YOLOv8n | YOLOv8-ES-n |
|--------|---------|-------------|
| mAP50 | ~0.65 | ~0.70 |
| mAP50-95 | ~0.45 | ~0.50 |
| Params | 3.2M | ~3.5M |
| FPS | ~140 | ~120 |

## Next Steps

1. ✅ Prepare dataset in YOLO format
2. ✅ Update `rdd2022.yaml` with correct paths
3. ✅ Run training with `train.py`
4. ✅ Validate model with `predict.py val`
5. ✅ Run inference with `predict.py predict`
6. ✅ Fine-tune hyperparameters if needed

## Support

For issues:
1. Check dataset format and paths
2. Verify custom modules are registered
3. Review training logs
4. Consult Ultralytics documentation

## References

- Paper: "Efficient and accurate road crack detection technology based on YOLOv8-ES"
- Ultralytics YOLOv8: https://docs.ultralytics.com
- RDD2022 Dataset: https://github.com/sekilab/RoadDamageDetector

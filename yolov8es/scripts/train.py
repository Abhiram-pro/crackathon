"""
YOLOv8-ES Training Script
Train YOLOv8-ES model on RDD2022 or custom dataset
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import yaml_load, LOGGER
import yaml

# Import custom modules
from model.edcm import EDCM
from model.sgam import SGAM
from model.loss_wiou import WIoUv3Loss


def register_custom_modules():
    """Register EDCM and SGAM with Ultralytics"""
    import ultralytics.nn.modules as modules
    
    if not hasattr(modules, 'EDCM'):
        modules.EDCM = EDCM
        LOGGER.info("✓ Registered EDCM module")
    
    if not hasattr(modules, 'SGAM'):
        modules.SGAM = SGAM
        LOGGER.info("✓ Registered SGAM module")


def train_yolov8es(
    data='rdd2022.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='',
    project='runs/train',
    name='yolov8es',
    pretrained=None,
    **kwargs
):
    """
    Train YOLOv8-ES model
    
    Args:
        data: Path to dataset YAML file
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
        device: Device to use ('', '0', '0,1', 'cpu')
        project: Project directory
        name: Experiment name
        pretrained: Path to pretrained weights (optional)
        **kwargs: Additional training arguments
    """
    # Register custom modules
    register_custom_modules()
    
    # Model configuration
    model_cfg = Path(__file__).parent / 'model' / 'yolov8es.yaml'
    
    print("=" * 70)
    print("YOLOv8-ES Training")
    print("=" * 70)
    print(f"Model config: {model_cfg}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto'}")
    print("=" * 70)
    print()
    
    # Create model
    if pretrained:
        print(f"Loading pretrained weights from: {pretrained}")
        model = YOLO(pretrained)
    else:
        print(f"Creating model from: {model_cfg}")
        model = YOLO(str(model_cfg))
    
    # Training arguments
    train_args = {
        'data': data,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'pretrained': False,  # We handle pretrained separately
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Class loss weight
        'dfl': 1.5,  # DFL loss weight
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'val': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'plots': True,
        'verbose': True,
    }
    
    # Update with any additional kwargs
    train_args.update(kwargs)
    
    # Train model
    print("Starting training...")
    print()
    results = model.train(**train_args)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {results.save_dir / 'weights' / 'last.pt'}")
    
    return results


def create_dataset_yaml(
    train_path,
    val_path,
    nc=4,
    names=None,
    output='dataset.yaml'
):
    """
    Create dataset YAML file
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        nc: Number of classes
        names: List of class names
        output: Output YAML file path
    """
    if names is None:
        names = [f'class_{i}' for i in range(nc)]
    
    dataset_config = {
        'path': str(Path(train_path).parent.parent),
        'train': str(Path(train_path).name),
        'val': str(Path(val_path).name),
        'nc': nc,
        'names': names
    }
    
    with open(output, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {output}")
    return output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8-ES model')
    parser.add_argument('--data', type=str, default='rdd2022.yaml', help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (e.g., 0 or 0,1 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov8es', help='Experiment name')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    
    args = parser.parse_args()
    
    # Train model
    train_yolov8es(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        workers=args.workers,
        cache=args.cache,
    )

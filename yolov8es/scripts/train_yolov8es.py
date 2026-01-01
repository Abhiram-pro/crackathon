"""
Simple YOLOv8-ES Training Script
Just run: python train_yolov8es.py --data your_data.yaml
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

# Import our custom modules
from model.edcm import EDCM
from model.sgam import SGAM
from model.loss_wiou import WIoUv3Loss


def build_yolov8es_model(base_model='yolov8n.pt', nc=4):
    """
    Build YOLOv8-ES by modifying YOLOv8n
    Adds EDCM to backbone and SGAM to neck
    """
    print("=" * 70)
    print("Building YOLOv8-ES Model")
    print("=" * 70)
    
    # Load base YOLOv8n
    print(f"\n1. Loading base model: {base_model}")
    model = YOLO(base_model)
    net = model.model
    
    # Get model structure
    print(f"   Base model loaded: {len(net.model)} layers")
    
    # Find Conv layers in backbone to replace with EDCM
    print("\n2. Replacing Conv layers with EDCM in backbone...")
    backbone_conv_indices = []
    for i, m in enumerate(net.model):
        if i < 12:  # Backbone typically ends around layer 11-12
            if isinstance(m, nn.modules.conv.Conv2d) or (hasattr(m, 'conv') and isinstance(m.conv, nn.modules.conv.Conv2d)):
                # Check if it's a downsampling conv (stride=2)
                if hasattr(m, 'conv'):
                    conv = m.conv
                else:
                    conv = m
                
                if hasattr(conv, 'stride') and conv.stride == (2, 2):
                    backbone_conv_indices.append(i)
    
    # Replace 2 Conv layers with EDCM (as per paper)
    # Typically replace the last 2 downsampling convs in backbone
    if len(backbone_conv_indices) >= 2:
        # Replace last 2 downsampling convs
        for idx in backbone_conv_indices[-2:]:
            layer = net.model[idx]
            # Get input/output channels
            if hasattr(layer, 'conv'):
                in_c = layer.conv.in_channels
                out_c = layer.conv.out_channels
            else:
                in_c = layer.in_channels
                out_c = layer.out_channels
            
            # Replace with EDCM (stride=1, so no downsampling)
            # Keep the original Conv for downsampling, add EDCM after
            edcm = EDCM(c1=out_c, c2=out_c)
            
            # Wrap in Sequential: Conv (downsample) -> EDCM (feature enhancement)
            net.model[idx] = nn.Sequential(layer, edcm)
            print(f"   ✓ Layer {idx}: Added EDCM after Conv ({out_c} channels)")
    
    # Add SGAM to neck (after concatenation layers)
    print("\n3. Adding SGAM to neck...")
    neck_start = 12  # Neck typically starts around layer 12
    sgam_count = 0
    
    for i in range(neck_start, len(net.model) - 1):  # -1 to skip Detect head
        layer = net.model[i]
        # Add SGAM after Concat layers
        if hasattr(layer, '__class__') and 'Concat' in layer.__class__.__name__:
            # Get the next layer to determine channels
            next_layer = net.model[i + 1]
            if hasattr(next_layer, 'conv'):
                channels = next_layer.conv.in_channels
            elif hasattr(next_layer, 'cv1'):
                channels = next_layer.cv1.conv.in_channels
            else:
                continue
            
            # Insert SGAM
            sgam = SGAM(c1=channels, c2=channels)
            # Wrap next layer with SGAM
            net.model[i + 1] = nn.Sequential(sgam, next_layer)
            sgam_count += 1
            print(f"   ✓ After layer {i}: Added SGAM ({channels} channels)")
            
            if sgam_count >= 4:  # Add SGAM to 4 locations in neck
                break
    
    print(f"\n✅ YOLOv8-ES model built successfully!")
    print(f"   - EDCM modules: 2")
    print(f"   - SGAM modules: {sgam_count}")
    print("=" * 70)
    print()
    
    return model


def train_with_wiou(model, **train_args):
    """
    Train model with WIoU v3 loss
    Note: WIoU integration into Ultralytics training loop requires
    modifying the loss function, which is complex.
    For now, we train with standard loss and can add WIoU in custom loop.
    """
    print("Starting training...")
    print("Note: Using standard YOLOv8 loss (WIoU v3 requires custom training loop)")
    print()
    
    results = model.train(**train_args)
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8-ES')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--base', type=str, default='yolov8n.pt', help='Base model (yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (0, 0,1, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov8es', help='Experiment name')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--cache', action='store_true', help='Cache images')
    
    args = parser.parse_args()
    
    # Build YOLOv8-ES model
    model = build_yolov8es_model(base_model=args.base)
    
    # Training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'workers': args.workers,
        'cache': args.cache,
        'exist_ok': True,
        'pretrained': False,
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
    }
    
    # Train
    results = train_with_wiou(model, **train_args)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
    

if __name__ == '__main__':
    main()

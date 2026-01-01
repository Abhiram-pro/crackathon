"""
YOLOv8-ES Inference Script
Run inference with trained YOLOv8-ES model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Import custom modules
from model.edcm import EDCM
from model.sgam import SGAM


def register_custom_modules():
    """Register EDCM and SGAM with Ultralytics"""
    import ultralytics.nn.modules as modules
    
    if not hasattr(modules, 'EDCM'):
        modules.EDCM = EDCM
        LOGGER.info("✓ Registered EDCM module")
    
    if not hasattr(modules, 'SGAM'):
        modules.SGAM = SGAM
        LOGGER.info("✓ Registered SGAM module")


def predict_yolov8es(
    weights,
    source,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device='',
    save=True,
    save_txt=False,
    save_conf=False,
    project='runs/predict',
    name='yolov8es',
    **kwargs
):
    """
    Run inference with YOLOv8-ES model
    
    Args:
        weights: Path to trained model weights
        source: Input source (image, video, directory, or stream)
        conf: Confidence threshold
        iou: IoU threshold for NMS
        imgsz: Input image size
        device: Device to use ('', '0', 'cpu')
        save: Save results
        save_txt: Save results as txt
        save_conf: Save confidence scores
        project: Project directory
        name: Experiment name
        **kwargs: Additional prediction arguments
    """
    # Register custom modules
    register_custom_modules()
    
    print("=" * 70)
    print("YOLOv8-ES Inference")
    print("=" * 70)
    print(f"Weights: {weights}")
    print(f"Source: {source}")
    print(f"Confidence: {conf}")
    print(f"IoU threshold: {iou}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto'}")
    print("=" * 70)
    print()
    
    # Load model
    model = YOLO(weights)
    
    # Prediction arguments
    predict_args = {
        'source': source,
        'conf': conf,
        'iou': iou,
        'imgsz': imgsz,
        'device': device,
        'save': save,
        'save_txt': save_txt,
        'save_conf': save_conf,
        'project': project,
        'name': name,
        'exist_ok': True,
        'show_labels': True,
        'show_conf': True,
        'line_width': 2,
    }
    
    # Update with any additional kwargs
    predict_args.update(kwargs)
    
    # Run inference
    print("Running inference...")
    results = model.predict(**predict_args)
    
    print()
    print("=" * 70)
    print("Inference completed!")
    print("=" * 70)
    
    # Print results summary
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.path}")
        print(f"  Detections: {len(result.boxes)}")
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"    Class {cls}: {conf:.3f}")
    
    return results


def validate_yolov8es(
    weights,
    data='rdd2022.yaml',
    batch=16,
    imgsz=640,
    device='',
    project='runs/val',
    name='yolov8es',
    **kwargs
):
    """
    Validate YOLOv8-ES model
    
    Args:
        weights: Path to trained model weights
        data: Dataset YAML path
        batch: Batch size
        imgsz: Input image size
        device: Device to use
        project: Project directory
        name: Experiment name
        **kwargs: Additional validation arguments
    """
    # Register custom modules
    register_custom_modules()
    
    print("=" * 70)
    print("YOLOv8-ES Validation")
    print("=" * 70)
    print(f"Weights: {weights}")
    print(f"Dataset: {data}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto'}")
    print("=" * 70)
    print()
    
    # Load model
    model = YOLO(weights)
    
    # Validation arguments
    val_args = {
        'data': data,
        'batch': batch,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.7,
        'plots': True,
    }
    
    # Update with any additional kwargs
    val_args.update(kwargs)
    
    # Run validation
    print("Running validation...")
    metrics = model.val(**val_args)
    
    print()
    print("=" * 70)
    print("Validation Results")
    print("=" * 70)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 70)
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8-ES Inference/Validation')
    parser.add_argument('mode', choices=['predict', 'val'], help='Mode: predict or val')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, help='Source for prediction')
    parser.add_argument('--data', type=str, default='rdd2022.yaml', help='Dataset YAML for validation')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device')
    parser.add_argument('--project', type=str, default='runs', help='Project directory')
    parser.add_argument('--name', type=str, default='yolov8es', help='Experiment name')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--save-txt', action='store_true', help='Save results as txt')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        if not args.source:
            parser.error("--source is required for prediction mode")
        predict_yolov8es(
            weights=args.weights,
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            project=args.project,
            name=args.name,
        )
    else:  # val
        validate_yolov8es(
            weights=args.weights,
            data=args.data,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
        )

"""
Simplest YOLOv8-ES Training
Usage: python simple_train.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO

# Just train YOLOv8n first to make sure everything works
# Then we'll add EDCM and SGAM

def train_baseline():
    """Train baseline YOLOv8n"""
    print("Training baseline YOLOv8n...")
    print("(We'll add EDCM and SGAM in the next step)")
    print()
    
    # Create model
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data='rdd2022.yaml',  # Update this path
        epochs=100,
        batch=16,
        imgsz=640,
        device='',  # Auto-select device
        project='runs/train',
        name='yolov8n_baseline',
        exist_ok=True,
    )
    
    print(f"\nTraining complete! Results: {results.save_dir}")
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='rdd2022.yaml', help='Dataset YAML')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device')
    
    args = parser.parse_args()
    
    # Create model
    model = YOLO('yolov8n.pt')
    
    # Train
    print("=" * 70)
    print("Training YOLOv8n")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print("=" * 70)
    print()
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        project='runs/train',
        name='yolov8n',
        exist_ok=True,
        plots=True,
        save=True,
    )
    
    print()
    print("=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Results: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print()
    print("Next: Use train_yolov8es.py to train with EDCM and SGAM")

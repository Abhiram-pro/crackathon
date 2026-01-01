"""
Test YOLOv8-ES model creation and forward pass
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.utils import yaml_load

# Import custom modules
from model.edcm import EDCM
from model.sgam import SGAM

# Register in globals for YAML parser
import ultralytics.nn.tasks as tasks
tasks.EDCM = EDCM
tasks.SGAM = SGAM

# Also register in modules
import ultralytics.nn.modules as modules
modules.EDCM = EDCM
modules.SGAM = SGAM

print("=" * 70)
print("YOLOv8-ES Model Test")
print("=" * 70)
print()

# Load configuration
cfg_path = Path(__file__).parent / 'model' / 'yolov8es.yaml'
print(f"Loading configuration from: {cfg_path}")
cfg = yaml_load(cfg_path)
print(f"✓ Configuration loaded")
print(f"  Classes: {cfg['nc']}")
print(f"  Scales: {cfg['scales']}")
print()

# Create model
print("Creating YOLOv8-ES model...")
try:
    model = DetectionModel(cfg, ch=3, nc=cfg['nc'])
    print("✓ Model created successfully!")
    print()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640)
        print(f"  Input shape: {x.shape}")
        
        y = model(x)
        print(f"  Output type: {type(y)}")
        if isinstance(y, (list, tuple)):
            print(f"  Number of outputs: {len(y)}")
            for i, yi in enumerate(y):
                if isinstance(yi, torch.Tensor):
                    print(f"    Output {i} shape: {yi.shape}")
        else:
            print(f"  Output shape: {y.shape}")
    
    print("✓ Forward pass successful!")
    print()
    
    # Check for custom modules
    print("Checking custom modules in model...")
    edcm_count = 0
    sgam_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, EDCM):
            edcm_count += 1
            print(f"  ✓ Found EDCM at: {name}")
        elif isinstance(module, SGAM):
            sgam_count += 1
            print(f"  ✓ Found SGAM at: {name}")
    
    print()
    print(f"Summary:")
    print(f"  EDCM modules: {edcm_count}")
    print(f"  SGAM modules: {sgam_count}")
    print()
    
    if edcm_count > 0 and sgam_count > 0:
        print("=" * 70)
        print("✅ YOLOv8-ES MODEL TEST PASSED!")
        print("=" * 70)
        print()
        print("The model is ready for training!")
        print()
        print("Next steps:")
        print("  1. Prepare your dataset in YOLO format")
        print("  2. Update rdd2022.yaml with your dataset paths")
        print("  3. Run: python train.py --data rdd2022.yaml --epochs 100")
    else:
        print("⚠️  Warning: Custom modules not found in model")
        print(f"   EDCM count: {edcm_count} (expected: 2+)")
        print(f"   SGAM count: {sgam_count} (expected: 4+)")
        
except Exception as e:
    print(f"❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

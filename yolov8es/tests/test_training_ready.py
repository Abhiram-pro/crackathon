"""
Test if everything is ready for training
Run this before starting training
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("YOLOv8-ES Training Readiness Check")
print("=" * 70)
print()

# Check 1: Python packages
print("1. Checking Python packages...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError:
    print("   ✗ PyTorch not found. Install: pip install torch")
    sys.exit(1)

try:
    import ultralytics
    print(f"   ✓ Ultralytics {ultralytics.__version__}")
except ImportError:
    print("   ✗ Ultralytics not found. Install: pip install ultralytics")
    sys.exit(1)

# Check 2: Custom modules
print("\n2. Checking custom modules...")

try:
    from model.edcm import EDCM
    print("   ✓ EDCM module found")
except ImportError as e:
    print(f"   ✗ EDCM module error: {e}")
    sys.exit(1)

try:
    from model.sgam import SGAM
    print("   ✓ SGAM module found")
except ImportError as e:
    print(f"   ✗ SGAM module error: {e}")
    sys.exit(1)

try:
    from model.loss_wiou import WIoUv3Loss
    print("   ✓ WIoU v3 module found")
except ImportError as e:
    print(f"   ✗ WIoU v3 module error: {e}")
    sys.exit(1)

# Check 3: Module functionality
print("\n3. Testing module functionality...")
try:
    edcm = EDCM(c1=64, c2=64)
    x = torch.randn(1, 64, 32, 32)
    y = edcm(x)
    assert y.shape == x.shape
    print("   ✓ EDCM forward pass works")
except Exception as e:
    print(f"   ✗ EDCM test failed: {e}")
    sys.exit(1)

try:
    sgam = SGAM(c1=128)
    x = torch.randn(1, 128, 32, 32)
    y = sgam(x)
    assert y.shape == x.shape
    print("   ✓ SGAM forward pass works")
except Exception as e:
    print(f"   ✗ SGAM test failed: {e}")
    sys.exit(1)

try:
    loss_fn = WIoUv3Loss()
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[1.0, 1.0, 11.0, 11.0]])
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    print("   ✓ WIoU v3 loss works")
except Exception as e:
    print(f"   ✗ WIoU v3 test failed: {e}")
    sys.exit(1)

# Check 4: Training scripts
print("\n4. Checking training scripts...")
base_dir = Path(__file__).parent.parent
scripts_dir = base_dir / 'scripts'
scripts = ['simple_train.py', 'train_yolov8es.py', 'predict.py']
for script in scripts:
    if (scripts_dir / script).exists():
        print(f"   ✓ {script} found")
    else:
        print(f"   ✗ {script} not found")

# Check 5: Dataset config
print("\n5. Checking dataset configuration...")
configs_dir = base_dir / 'configs'
if (configs_dir / 'rdd2022.yaml').exists():
    print("   ✓ rdd2022.yaml found")
    print("   ⚠  Remember to update the 'path:' in configs/rdd2022.yaml!")
else:
    print("   ✗ rdd2022.yaml not found")

# Check 6: GPU availability
print("\n6. Checking GPU...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
else:
    print("   ⚠  No GPU detected (will use CPU - slower)")

print()
print("=" * 70)
print("✅ READY TO TRAIN!")
print("=" * 70)
print()
print("Next steps:")
print("1. Update 'path:' in configs/rdd2022.yaml with your dataset location")
print("2. Run: python scripts/simple_train.py --data configs/rdd2022.yaml --epochs 10")
print("3. Check results in runs/train/yolov8n/")
print()
print("For full YOLOv8-ES with EDCM and SGAM:")
print("   python scripts/train_yolov8es.py --data configs/rdd2022.yaml --epochs 100")

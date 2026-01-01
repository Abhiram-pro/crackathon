# YOLOv8-ES Quick Start Guide

## Installation

```bash
# Clone or navigate to the project
cd yolov8es

# Install dependencies
pip install torch torchvision numpy
```

## Verify Installation

Run all verification tests to ensure modules are working correctly:

```bash
python verifyscript/run_all_tests.py
```

Expected output:
```
🎉 ALL TESTS PASSED! YOLOv8-ES modules are ready for integration.
```

## Using the Modules

### EDCM (Backbone Enhancement)

```python
import torch
from model.edcm import EDCM

# Create module
edcm = EDCM(in_channels=64, out_channels=64, k=3)

# Forward pass
x = torch.randn(2, 64, 128, 128)
y = edcm(x)

print(f"Input:  {x.shape}")   # [2, 64, 128, 128]
print(f"Output: {y.shape}")   # [2, 64, 128, 128] (preserved)
```

### SGAM (Neck Enhancement)

```python
import torch
from model.sgam import SGAM

# Create module
sgam = SGAM(channels=128)

# Forward pass
x = torch.randn(2, 128, 64, 64)
y = sgam(x)

print(f"Input:  {x.shape}")   # [2, 128, 64, 64]
print(f"Output: {y.shape}")   # [2, 128, 64, 64] (preserved)
```

### WIoU v3 Loss

```python
import torch
from model.loss_wiou import WIoUv3Loss, wiou_v3_loss

# Predicted and target boxes (x1, y1, x2, y2 format)
pred = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                     [5.0, 5.0, 15.0, 15.0]])
target = torch.tensor([[1.0, 1.0, 11.0, 11.0],
                       [6.0, 6.0, 16.0, 16.0]])

# Method 1: Module interface
loss_fn = WIoUv3Loss(monotonous=False)  # v3 (recommended)
loss = loss_fn(pred, target)
print(f"Loss per box: {loss}")
print(f"Mean loss: {loss.mean()}")

# Method 2: Functional interface
loss = wiou_v3_loss(pred, target)
print(f"Mean loss: {loss}")

# Get IoU values along with loss
loss, iou = loss_fn(pred, target, ret_iou=True)
print(f"IoU values: {iou}")
```

## Testing Individual Modules

### Test EDCM

```bash
python verifyscript/verify_edcm.py
```

### Test SGAM

```bash
# Basic tests
python verifyscript/verify_sgam.py

# Detailed tests
python verifyscript/verify_sgam_detailed.py
```

### Test WIoU v3

```bash
# Basic tests
python verifyscript/verify_wiou.py

# Detailed tests
python verifyscript/verify_wiou_detailed.py
```

## Module Parameters

### EDCM Parameters

```python
EDCM(
    in_channels,    # Number of input channels
    out_channels,   # Number of output channels
    k=3,           # Kernel size (default: 3)
    stride=1,      # Ignored, always 1 (no downsampling)
    groups=1       # Number of groups for grouped convolution
)
```

### SGAM Parameters

```python
SGAM(
    c              # Number of channels
)
```

### WIoUv3Loss Parameters

```python
WIoUv3Loss(
    monotonous=False,  # False: v3 (non-monotonic), True: v1/v2 (monotonic)
    eps=1e-7          # Small constant for numerical stability
)
```

## Common Issues

### Import Errors

If you get `ModuleNotFoundError`, ensure you're running from the correct directory:

```bash
# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/yolov8es"
```

Or use the provided sys.path setup in verification scripts.

### CUDA/GPU Usage

All modules support GPU acceleration:

```python
import torch
from model.edcm import EDCM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

edcm = EDCM(64, 64).to(device)
x = torch.randn(2, 64, 128, 128).to(device)
y = edcm(x)
```

## Performance Tips

1. **Batch Processing**: Process multiple samples together for better GPU utilization
2. **Mixed Precision**: Use `torch.cuda.amp` for faster training
3. **Gradient Checkpointing**: For memory-constrained environments

## Next Steps

1. ✅ Verify all modules work correctly
2. ⏳ Create YOLOv8-ES YAML configuration
3. ⏳ Integrate modules into YOLOv8 architecture
4. ⏳ Prepare training pipeline
5. ⏳ Train on RDD2022 dataset

## Documentation

- `README.md` - Full project documentation
- `VERIFICATION_SUMMARY.md` - Detailed verification report
- Paper: `papers/s43684-025-00091-3.pdf`

## Support

For issues or questions:
1. Check verification scripts for usage examples
2. Review module docstrings
3. Consult the original paper (Section 3.2, 3.3, 3.4)

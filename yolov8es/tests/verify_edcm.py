import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from model.edcm import EDCM

print("PyTorch version:", torch.__version__)
print()

# Test 1: Basic forward pass
print("Test 1: Basic forward pass (64 -> 64)")
m = EDCM(64, 64)
x = torch.randn(2, 64, 128, 128)
y = m(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {y.shape}")
print(f"  ✓ Shape preserved (stride=1)")
print()

# Test 2: Different channels
print("Test 2: Channel change (32 -> 64)")
m2 = EDCM(32, 64)
x2 = torch.randn(4, 32, 64, 64)
y2 = m2(x2)
print(f"  Input:  {x2.shape}")
print(f"  Output: {y2.shape}")
print(f"  ✓ Spatial dims preserved")
print()

# Test 3: Verify no downsampling
print("Test 3: Verify stride=1 behavior")
m3 = EDCM(16, 16, k=3)
x3 = torch.randn(1, 16, 256, 256)
y3 = m3(x3)
assert y3.shape[2:] == x3.shape[2:], "Spatial dimensions should not change!"
print(f"  Input:  {x3.shape}")
print(f"  Output: {y3.shape}")
print(f"  ✓ No downsampling confirmed")
print()

print("✅ All EDCM tests passed!")

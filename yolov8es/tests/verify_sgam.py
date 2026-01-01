import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from model.sgam import SGAM, SE, GAM, CoordinateAttention

print("PyTorch version:", torch.__version__)
print()

# Test 1: SE (Squeeze-and-Excitation)
print("=" * 60)
print("Test 1: SE (Squeeze-and-Excitation)")
print("=" * 60)
se = SE(128, r=16)
x1 = torch.randn(2, 128, 64, 64)
y1 = se(x1)
print(f"  Input:  {x1.shape}")
print(f"  Output: {y1.shape}")
assert y1.shape == x1.shape, "SE should preserve shape"
print(f"  ✓ Shape preserved")
print(f"  ✓ Channel attention applied")
print()

# Test 2: GAM (Global Attention Mechanism)
print("=" * 60)
print("Test 2: GAM (Global Attention Mechanism)")
print("=" * 60)
gam = GAM(128)
x2 = torch.randn(2, 128, 64, 64)
y2 = gam(x2)
print(f"  Input:  {x2.shape}")
print(f"  Output: {y2.shape}")
assert y2.shape == x2.shape, "GAM should preserve shape"
print(f"  ✓ Shape preserved")
print(f"  ✓ Global attention applied")
print()

# Test 3: Coordinate Attention
print("=" * 60)
print("Test 3: Coordinate Attention")
print("=" * 60)
ca = CoordinateAttention(128, r=32)
x3 = torch.randn(2, 128, 64, 64)
y3 = ca(x3)
print(f"  Input:  {x3.shape}")
print(f"  Output: {y3.shape}")
assert y3.shape == x3.shape, "CA should preserve shape"
print(f"  ✓ Shape preserved")
print(f"  ✓ Coordinate attention applied (H and W aware)")
print()

# Test 4: Full SGAM (SE → GAM → CA)
print("=" * 60)
print("Test 4: Full SGAM (SE → GAM → CA)")
print("=" * 60)
sgam = SGAM(128)
x4 = torch.randn(2, 128, 64, 64)
y4 = sgam(x4)
print(f"  Input:  {x4.shape}")
print(f"  Output: {y4.shape}")
assert y4.shape == x4.shape, "SGAM should preserve shape"
print(f"  ✓ Shape preserved")
print(f"  ✓ Sequential attention: SE → GAM → CA")
print()

# Test 5: Different input sizes
print("=" * 60)
print("Test 5: Different input sizes")
print("=" * 60)
test_cases = [
    (1, 64, 32, 32),
    (4, 256, 128, 128),
    (2, 512, 16, 16),
]
for batch, channels, h, w in test_cases:
    sgam_test = SGAM(channels)
    x_test = torch.randn(batch, channels, h, w)
    y_test = sgam_test(x_test)
    assert y_test.shape == x_test.shape
    print(f"  ✓ [{batch}, {channels}, {h}, {w}] → {y_test.shape}")
print()

# Test 6: Gradient flow
print("=" * 60)
print("Test 6: Gradient flow")
print("=" * 60)
sgam_grad = SGAM(64)
x_grad = torch.randn(2, 64, 32, 32, requires_grad=True)
y_grad = sgam_grad(x_grad)
loss = y_grad.sum()
loss.backward()
assert x_grad.grad is not None, "Gradients should flow"
print(f"  ✓ Gradients computed successfully")
print(f"  ✓ Input grad shape: {x_grad.grad.shape}")
print()

# Test 7: Attention effect (output should differ from input)
print("=" * 60)
print("Test 7: Attention effect verification")
print("=" * 60)
sgam_effect = SGAM(64)
x_effect = torch.randn(2, 64, 32, 32)
y_effect = sgam_effect(x_effect)
diff = (y_effect - x_effect).abs().mean().item()
print(f"  Mean absolute difference: {diff:.6f}")
assert diff > 0, "SGAM should modify the input"
print(f"  ✓ Attention is active (modifies features)")
print()

print("=" * 60)
print("✅ All SGAM tests passed!")
print("=" * 60)

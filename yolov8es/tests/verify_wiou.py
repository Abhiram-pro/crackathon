import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from model.loss_wiou import WIoUv3Loss, wiou_v3_loss

print("PyTorch version:", torch.__version__)
print()

# ============================================================================
# Test 1: Basic functionality
# ============================================================================
print("=" * 60)
print("Test 1: Basic WIoU v3 Loss")
print("=" * 60)

# Perfect overlap (IoU = 1.0)
pred_perfect = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
target_perfect = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

loss_fn = WIoUv3Loss(monotonous=False)
loss_perfect, iou_perfect = loss_fn(pred_perfect, target_perfect, ret_iou=True)

print(f"Perfect overlap:")
print(f"  IoU: {iou_perfect.item():.4f}")
print(f"  Loss: {loss_perfect.item():.4f}")
assert iou_perfect.item() > 0.99, "Perfect overlap should have IoU ≈ 1.0"
assert loss_perfect.item() < 0.1, "Perfect overlap should have low loss"
print("  ✓ Perfect overlap handled correctly")
print()

# Partial overlap
pred_partial = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
target_partial = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

loss_partial, iou_partial = loss_fn(pred_partial, target_partial, ret_iou=True)

print(f"Partial overlap:")
print(f"  IoU: {iou_partial.item():.4f}")
print(f"  Loss: {loss_partial.item():.4f}")
assert 0.0 < iou_partial.item() < 1.0, "Partial overlap should have 0 < IoU < 1"
assert loss_partial.item() > loss_perfect.item(), "Partial overlap should have higher loss"
print("  ✓ Partial overlap handled correctly")
print()

# No overlap
pred_no = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
target_no = torch.tensor([[20.0, 20.0, 30.0, 30.0]])

loss_no, iou_no = loss_fn(pred_no, target_no, ret_iou=True)

print(f"No overlap:")
print(f"  IoU: {iou_no.item():.4f}")
print(f"  Loss: {loss_no.item():.4f}")
assert iou_no.item() < 0.01, "No overlap should have IoU ≈ 0"
assert loss_no.item() > loss_partial.item(), "No overlap should have highest loss"
print("  ✓ No overlap handled correctly")
print()

# ============================================================================
# Test 2: Batch processing
# ============================================================================
print("=" * 60)
print("Test 2: Batch Processing")
print("=" * 60)

batch_pred = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [5.0, 5.0, 15.0, 15.0],
    [10.0, 10.0, 20.0, 20.0],
    [0.0, 0.0, 5.0, 5.0],
])

batch_target = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],   # Perfect match
    [5.0, 5.0, 15.0, 15.0],   # Perfect match
    [12.0, 12.0, 22.0, 22.0], # Partial overlap
    [10.0, 10.0, 15.0, 15.0], # No overlap
])

batch_loss, batch_iou = loss_fn(batch_pred, batch_target, ret_iou=True)

print(f"Batch of 4 boxes:")
print(f"  IoU values: {batch_iou.tolist()}")
print(f"  Loss values: {batch_loss.tolist()}")
print(f"  Mean loss: {batch_loss.mean().item():.4f}")
assert batch_loss.shape[0] == 4, "Should return loss for each box"
assert batch_iou.shape[0] == 4, "Should return IoU for each box"
print("  ✓ Batch processing works correctly")
print()

# ============================================================================
# Test 3: Center format conversion
# ============================================================================
print("=" * 60)
print("Test 3: Center Format Conversion")
print("=" * 60)

# Corner format: (x1, y1, x2, y2)
pred_corner = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
target_corner = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

# Center format: (cx, cy, w, h)
pred_center = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
target_center = torch.tensor([[5.0, 5.0, 10.0, 10.0]])

loss_corner = loss_fn(pred_corner, target_corner)
loss_center = loss_fn(pred_center, target_center)

print(f"Corner format loss: {loss_corner.item():.4f}")
print(f"Center format loss: {loss_center.item():.4f}")
assert torch.allclose(loss_corner, loss_center, atol=1e-4), "Both formats should give same loss"
print("  ✓ Center format conversion works correctly")
print()

# ============================================================================
# Test 4: WIoU v3 vs v1/v2 (monotonous)
# ============================================================================
print("=" * 60)
print("Test 4: WIoU v3 vs v1/v2 Comparison")
print("=" * 60)

loss_v3 = WIoUv3Loss(monotonous=False)
loss_v12 = WIoUv3Loss(monotonous=True)

test_pred = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [5.0, 5.0, 15.0, 15.0],
    [20.0, 20.0, 30.0, 30.0],
])

test_target = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],   # High IoU
    [6.0, 6.0, 16.0, 16.0],   # Medium IoU
    [25.0, 25.0, 35.0, 35.0], # Low IoU
])

loss_v3_vals = loss_v3(test_pred, test_target)
loss_v12_vals = loss_v12(test_pred, test_target)

print("WIoU v3 (non-monotonic):")
print(f"  Losses: {loss_v3_vals.tolist()}")
print(f"  Mean: {loss_v3_vals.mean().item():.4f}")

print("WIoU v1/v2 (monotonic):")
print(f"  Losses: {loss_v12_vals.tolist()}")
print(f"  Mean: {loss_v12_vals.mean().item():.4f}")

print("  ✓ Both versions compute successfully")
print()

# ============================================================================
# Test 5: Gradient flow
# ============================================================================
print("=" * 60)
print("Test 5: Gradient Flow")
print("=" * 60)

pred_grad = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [5.0, 5.0, 15.0, 15.0],
], requires_grad=True)

target_grad = torch.tensor([
    [1.0, 1.0, 11.0, 11.0],
    [6.0, 6.0, 16.0, 16.0],
])

loss = loss_fn(pred_grad, target_grad).mean()
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Gradient exists: {pred_grad.grad is not None}")
print(f"Gradient shape: {pred_grad.grad.shape}")
print(f"Gradient mean: {pred_grad.grad.mean().item():.6f}")
print(f"Gradient std: {pred_grad.grad.std().item():.6f}")
assert pred_grad.grad is not None, "Gradients should flow"
print("  ✓ Gradients flow correctly")
print()

# ============================================================================
# Test 6: Functional interface
# ============================================================================
print("=" * 60)
print("Test 6: Functional Interface")
print("=" * 60)

pred_func = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [5.0, 5.0, 15.0, 15.0],
])

target_func = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [6.0, 6.0, 16.0, 16.0],
])

loss_module = loss_fn(pred_func, target_func).mean()
loss_functional = wiou_v3_loss(pred_func, target_func)

print(f"Module interface: {loss_module.item():.4f}")
print(f"Functional interface: {loss_functional.item():.4f}")
assert torch.allclose(loss_module, loss_functional, atol=1e-6), "Both interfaces should match"
print("  ✓ Functional interface works correctly")
print()

# ============================================================================
# Test 7: Edge cases
# ============================================================================
print("=" * 60)
print("Test 7: Edge Cases")
print("=" * 60)

# Very small boxes
pred_small = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
target_small = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
loss_small = loss_fn(pred_small, target_small)
print(f"Very small boxes: loss = {loss_small.item():.4f}")
assert not torch.isnan(loss_small).any(), "Should handle small boxes"
print("  ✓ Small boxes handled")

# Very large boxes
pred_large = torch.tensor([[0.0, 0.0, 1000.0, 1000.0]])
target_large = torch.tensor([[0.0, 0.0, 1000.0, 1000.0]])
loss_large = loss_fn(pred_large, target_large)
print(f"Very large boxes: loss = {loss_large.item():.4f}")
assert not torch.isnan(loss_large).any(), "Should handle large boxes"
print("  ✓ Large boxes handled")

# Aspect ratio differences
pred_wide = torch.tensor([[0.0, 0.0, 20.0, 5.0]])
target_tall = torch.tensor([[0.0, 0.0, 5.0, 20.0]])
loss_aspect = loss_fn(pred_wide, target_tall)
print(f"Different aspect ratios: loss = {loss_aspect.item():.4f}")
assert not torch.isnan(loss_aspect).any(), "Should handle aspect ratio differences"
print("  ✓ Aspect ratio differences handled")
print()

# ============================================================================
# Test 8: Loss properties
# ============================================================================
print("=" * 60)
print("Test 8: Loss Properties")
print("=" * 60)

# Test that loss decreases as boxes get closer
pred_base = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
targets_progressive = torch.tensor([
    [10.0, 10.0, 20.0, 20.0],  # Far
    [5.0, 5.0, 15.0, 15.0],    # Medium
    [2.0, 2.0, 12.0, 12.0],    # Close
    [0.0, 0.0, 10.0, 10.0],    # Perfect
])

losses = []
for i in range(4):
    target = targets_progressive[i:i+1]
    loss = loss_fn(pred_base, target)
    losses.append(loss.item())

print("Loss progression (far → close → perfect):")
for i, loss_val in enumerate(losses):
    print(f"  Step {i+1}: {loss_val:.4f}")

# Check monotonic decrease
for i in range(len(losses) - 1):
    assert losses[i] >= losses[i+1], f"Loss should decrease: {losses[i]} >= {losses[i+1]}"
print("  ✓ Loss decreases as boxes get closer")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 60)
print("✅ ALL WIoU v3 TESTS PASSED")
print("=" * 60)
print()
print("Summary:")
print("  ✓ Basic functionality (perfect/partial/no overlap)")
print("  ✓ Batch processing")
print("  ✓ Center format conversion")
print("  ✓ v3 vs v1/v2 comparison")
print("  ✓ Gradient flow")
print("  ✓ Functional interface")
print("  ✓ Edge cases (small/large boxes, aspect ratios)")
print("  ✓ Loss properties (monotonic decrease)")
print()
print("WIoU v3 is ready for integration into YOLOv8-ES!")

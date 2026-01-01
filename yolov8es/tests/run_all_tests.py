"""
Run all YOLOv8-ES module verification tests
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import time

print("=" * 80)
print("YOLOv8-ES COMPLETE VERIFICATION SUITE")
print("=" * 80)
print()

tests = [
    ("EDCM (Enhanced Dynamic Convolution Module)", "verify_edcm.py"),
    ("SGAM (Selective Global Attention Mechanism)", "verify_sgam.py"),
    ("WIoU v3 (Wise-IoU v3 Loss)", "verify_wiou.py"),
]

results = []
total_time = 0

for name, script in tests:
    print(f"Running: {name}")
    print("-" * 80)
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, f"yolov8es/verifyscript/{script}"],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    total_time += elapsed
    
    success = result.returncode == 0
    results.append((name, success, elapsed))
    
    if success:
        print(f"✅ PASSED ({elapsed:.2f}s)")
    else:
        print(f"❌ FAILED ({elapsed:.2f}s)")
        print("Error output:")
        print(result.stderr)
    
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

passed = sum(1 for _, success, _ in results if success)
total = len(results)

for name, success, elapsed in results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {name:<50} ({elapsed:.2f}s)")

print()
print(f"Total: {passed}/{total} tests passed")
print(f"Total time: {total_time:.2f}s")
print()

if passed == total:
    print("🎉 ALL TESTS PASSED! YOLOv8-ES modules are ready for integration.")
    sys.exit(0)
else:
    print("⚠️  Some tests failed. Please review the errors above.")
    sys.exit(1)

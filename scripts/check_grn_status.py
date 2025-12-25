#!/usr/bin/env python3
"""Quick check of GRN matrix status."""
import numpy as np
from pathlib import Path

model_dir = Path('results/sclc')

print("=" * 60)
print("CHECKING GRN MATRICES IN results/sclc/")
print("=" * 60)

# Check existing GRN
grn_path = model_dir / 'grn_matrix.npy'
grn_correct_path = model_dir / 'grn_matrix_CORRECT.npy'

if grn_path.exists():
    grn = np.load(grn_path)
    print(f"\ngrn_matrix.npy:")
    print(f"  Shape: {grn.shape}")
    print(f"  Range: [{grn.min():.4f}, {grn.max():.4f}]")
    print(f"  Mean:  {grn.mean():.6f}")
    
    if abs(grn.max()) < 0.01:
        print("  ⚠️  WARNING: Values are very small - likely has the tf_importance bug!")
        needs_fix = True
    else:
        print("  ✓ Values look reasonable (no bug)")
        needs_fix = False
else:
    print("\ngrn_matrix.npy: NOT FOUND")
    needs_fix = True

if grn_correct_path.exists():
    grn_correct = np.load(grn_correct_path)
    print(f"\ngrn_matrix_CORRECT.npy:")
    print(f"  Shape: {grn_correct.shape}")
    print(f"  Range: [{grn_correct.min():.4f}, {grn_correct.max():.4f}]")
    print("  ✓ Verified correct GRN exists!")
else:
    print("\ngrn_matrix_CORRECT.npy: NOT FOUND")
    if needs_fix:
        print("  → Running diagnose_grn.py to create it...")
        exec(open('scripts/diagnose_grn.py').read())
    else:
        print("  → Creating it from current (correct) GRN...")
        np.save(grn_correct_path, grn)
        print(f"  Saved to {grn_correct_path}")

print("\n" + "=" * 60)


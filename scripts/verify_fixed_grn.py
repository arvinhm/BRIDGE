#!/usr/bin/env python3
"""Verify the fixed GRN matrix was saved correctly."""

import numpy as np
import pandas as pd

print("="*70)
print("VERIFYING FIXED GRN MATRIX")
print("="*70)

grn = np.load('results/sclc_SSTR2_VERIFIED/grn_matrix.npy')
print(f'GRN shape: {grn.shape}')
print(f'GRN range: [{grn.min():.4f}, {grn.max():.4f}]')
print(f'Expected range: around [-0.18, 0.33]')

# Check if it matches the correct GRN
correct_grn = np.load('results/sclc/grn_matrix_CORRECT.npy')
is_match = np.allclose(grn, correct_grn, atol=1e-6)
print(f'\nMatches CORRECT GRN: {is_match}')

if is_match:
    print("✓ VERIFIED: Fine-tuning used the correct GRN matrix!")
else:
    diff = np.abs(grn - correct_grn).max()
    print(f"⚠️ Max difference: {diff:.6f}")

# SSTR2 effects
gene_names = pd.read_csv('results/sclc/gene_names.csv', header=None)[0].tolist()
sstr2_idx = gene_names.index('SSTR2')
effects = grn[sstr2_idx]
print(f'\nSSTR2 (index {sstr2_idx}):')
print(f'  Activators: {(effects > 0).sum()}')
print(f'  Repressors: {(effects < 0).sum()}')
print(f'  Effect range: [{effects.min():.4f}, {effects.max():.4f}]')


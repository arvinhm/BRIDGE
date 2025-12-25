#!/usr/bin/env python3
"""Diagnose why GRN effects are so small."""

import numpy as np
import pandas as pd
import torch

print('='*70)
print('CRITICAL DIAGNOSTIC: Why are GRN effects so small?')
print('='*70)

# Load saved GRN matrix - check multiple possible locations
import os
for path in ['results/sclc_grn_v6/grn_matrix.npy', 'results/sclc/grn_matrix.npy', 'results/sclc_grn_complete/grn_matrix.npy']:
    if os.path.exists(path):
        grn_path = path
        break
else:
    raise FileNotFoundError("No GRN matrix found!")
    
print(f"Using GRN from: {grn_path}")
grn = np.load(grn_path)
model_dir = os.path.dirname(grn_path)
print(f'Saved GRN matrix: shape={grn.shape}')
print(f'  Range: [{grn.min():.6f}, {grn.max():.6f}]')
print(f'  Mean: {grn.mean():.8f}')
print(f'  Std: {grn.std():.8f}')

# Load checkpoint and compute CORRECT GRN
print()
print('Now computing GRN from checkpoint directly...')
cp = torch.load(f'{model_dir}/best_model.pt', map_location='cpu', weights_only=False)
state = cp['model_state_dict']

direct_effects = state['direct_effects'].numpy()
tf_specificity = state['tf_specificity_weights'].numpy()
chip_mask = state['chip_mask'].numpy()

# Leakage
leakage_raw = state['_leakage_raw'].item()
leakage = 0.005 + 0.195 / (1 + np.exp(-leakage_raw))

print(f'direct_effects: range=[{direct_effects.min():.4f}, {direct_effects.max():.4f}]')
print(f'tf_specificity: range=[{tf_specificity.min():.4f}, {tf_specificity.max():.4f}]')
print(f'chip_mask: non-zero={(chip_mask > 0).sum()}')
print(f'leakage: {leakage:.4f}')

# Correct GRN computation (without tf_importance scaling!)
grn_correct = direct_effects * tf_specificity[np.newaxis, :] * (chip_mask + leakage)
print()
print(f'CORRECT GRN (without tf_importance):')
print(f'  Range: [{grn_correct.min():.4f}, {grn_correct.max():.4f}]')
print(f'  Mean: {grn_correct.mean():.6f}')

# Get SSTR2 effects
gene_names = pd.read_csv(f'{model_dir}/gene_names.csv', header=None)[0].tolist()
sstr2_idx = [g.upper() for g in gene_names].index('SSTR2')
print()
print(f'SSTR2 (index {sstr2_idx}):')
print(f'  Saved GRN: range=[{grn[sstr2_idx].min():.6f}, {grn[sstr2_idx].max():.6f}]')
print(f'  Correct GRN: range=[{grn_correct[sstr2_idx].min():.4f}, {grn_correct[sstr2_idx].max():.4f}]')
print(f'  Saved activators: {(grn[sstr2_idx] > 0).sum()}, repressors: {(grn[sstr2_idx] < 0).sum()}')
print(f'  Correct activators: {(grn_correct[sstr2_idx] > 0).sum()}, repressors: {(grn_correct[sstr2_idx] < 0).sum()}')

# The ratio between correct and saved
ratio = grn_correct.std() / (grn.std() + 1e-10)
print()
print(f'CRITICAL: Saved GRN is {ratio:.0f}x SMALLER than correct!')
print('This is why effects look like 0 and predictions are bad.')
print()

# Check what get_grn_matrix is doing differently
print('='*70)
print('ROOT CAUSE ANALYSIS')
print('='*70)
print("""
The saved grn_matrix.npy is computed by get_grn_matrix() which:
1. Gets tf_importance from TF encoder
2. Applies softmax to tf_importance (makes sum = 1.0!)
3. MULTIPLIES the GRN by this softmaxed importance

With 1480 TFs, each tf_importance is ~0.0007 on average.
Multiplying the GRN by 0.0007 reduces all effects by 1000x!

THIS IS THE BUG.
""")

# Save correct GRN for comparison
correct_path = f'{model_dir}/grn_matrix_CORRECT.npy'
np.save(correct_path, grn_correct)
print(f'Saved CORRECT GRN matrix to: {correct_path}')


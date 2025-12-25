#!/usr/bin/env python3
"""Quick check of GRN matrix for SSTR2."""

import numpy as np
import pandas as pd

# Load GRN matrix
grn = np.load('results/sclc_SSTR2/grn_matrix.npy')
print(f'GRN matrix shape: {grn.shape}')
print(f'GRN matrix range: [{grn.min():.6f}, {grn.max():.6f}]')
print(f'GRN matrix mean: {grn.mean():.6f}')
print(f'GRN matrix std: {grn.std():.6f}')
print(f'Non-zero entries: {(grn != 0).sum()} / {grn.size}')

# Check SSTR2 row
gene_names = pd.read_csv('results/sclc_SSTR2/gene_names.csv', header=None)[0].tolist()
sstr2_idx = gene_names.index('SSTR2') if 'SSTR2' in gene_names else -1
print(f'\nSSTR2 index: {sstr2_idx}')

if sstr2_idx >= 0:
    sstr2_effects = grn[sstr2_idx, :]
    print(f'SSTR2 effects range: [{sstr2_effects.min():.6f}, {sstr2_effects.max():.6f}]')
    print(f'SSTR2 positive effects: {(sstr2_effects > 0).sum()}')
    print(f'SSTR2 negative effects: {(sstr2_effects < 0).sum()}')
    print(f'SSTR2 zero effects: {(sstr2_effects == 0).sum()}')
    
    # Top effects
    tf_names = pd.read_csv('results/sclc_SSTR2/tf_names.csv', header=None)[0].tolist()
    sorted_idx = np.argsort(sstr2_effects)
    print('\nTop 10 activators (positive effects):')
    for i in sorted_idx[-10:][::-1]:
        print(f'  {tf_names[i]}: {sstr2_effects[i]:.6f}')
    print('\nTop 10 repressors (negative effects):')
    for i in sorted_idx[:10]:
        print(f'  {tf_names[i]}: {sstr2_effects[i]:.6f}')

# Also check the v6 base model GRN
print("\n" + "="*60)
print("Checking train_v6 base model GRN...")
try:
    grn_v6 = np.load('results/sclc_grn_v6/grn_matrix.npy')
    print(f'GRN v6 matrix shape: {grn_v6.shape}')
    print(f'GRN v6 matrix range: [{grn_v6.min():.6f}, {grn_v6.max():.6f}]')
    
    gene_names_v6 = pd.read_csv('results/sclc_grn_v6/gene_names.csv', header=None)[0].tolist()
    sstr2_idx_v6 = gene_names_v6.index('SSTR2') if 'SSTR2' in gene_names_v6 else -1
    
    if sstr2_idx_v6 >= 0:
        sstr2_effects_v6 = grn_v6[sstr2_idx_v6, :]
        print(f'SSTR2 effects range: [{sstr2_effects_v6.min():.6f}, {sstr2_effects_v6.max():.6f}]')
        print(f'SSTR2 positive effects: {(sstr2_effects_v6 > 0).sum()}')
        print(f'SSTR2 negative effects: {(sstr2_effects_v6 < 0).sum()}')
except Exception as e:
    print(f"Error loading v6 GRN: {e}")


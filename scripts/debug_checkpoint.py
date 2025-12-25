#!/usr/bin/env python3
"""Debug checkpoint loading issues."""

import torch
import numpy as np

# Load checkpoint
cp = torch.load('results/sclc_grn_v6/best_model.pt', map_location='cpu', weights_only=False)

print("=" * 70)
print("CHECKING tf_tf_adjacency")
print("=" * 70)
print('tf_tf_adjacency in state_dict:', 'tf_tf_adjacency' in cp['model_state_dict'])

# Check all keys with tf_tf or adjacency
print("\nAll keys containing 'tf_tf' or 'adjacency':")
for k in cp['model_state_dict'].keys():
    if 'tf_tf' in k.lower() or 'adjacency' in k.lower():
        print(f"  {k}")

# Check direct_effects
print("\n" + "=" * 70)
print("CHECKING direct_effects")
print("=" * 70)
de = cp['model_state_dict']['direct_effects']
print(f"direct_effects: shape={de.shape}")
print(f"  range: [{de.min():.4f}, {de.max():.4f}]")
print(f"  positive: {(de > 0).sum().item()}")
print(f"  negative: {(de < 0).sum().item()}")
print(f"  zero: {(de == 0).sum().item()}")

# Get SSTR2 row specifically
gene_names = []
try:
    import pandas as pd
    gene_names = pd.read_csv('results/sclc_grn_v6/gene_names.csv', header=None)[0].tolist()
    sstr2_idx = [g.upper() for g in gene_names].index('SSTR2')
    print(f"\nSSTR2 index: {sstr2_idx}")
    sstr2_row = de[sstr2_idx, :]
    print(f"SSTR2 direct_effects: range=[{sstr2_row.min():.6f}, {sstr2_row.max():.6f}]")
    print(f"  positive: {(sstr2_row > 0).sum().item()}")
    print(f"  negative: {(sstr2_row < 0).sum().item()}")
except Exception as e:
    print(f"Error getting SSTR2: {e}")

# Check chip_mask
print("\n" + "=" * 70)
print("CHECKING chip_mask")
print("=" * 70)
cm = cp['model_state_dict']['chip_mask']
print(f"chip_mask: shape={cm.shape}")
print(f"  non-zero: {(cm > 0).sum().item()}")
print(f"  max: {cm.max():.4f}")

# Check tf_specificity_weights
print("\n" + "=" * 70)
print("CHECKING tf_specificity_weights")
print("=" * 70)
tw = cp['model_state_dict']['tf_specificity_weights']
print(f"tf_specificity_weights: shape={tw.shape}")
print(f"  range: [{tw.min():.4f}, {tw.max():.4f}]")
print(f"  mean: {tw.mean():.4f}")

# Now compute what the GRN matrix SHOULD be
print("\n" + "=" * 70)
print("COMPUTING GRN MATRIX (correct way)")
print("=" * 70)

# The formula is: grn = direct_effects * tf_specificity * (chip_mask + leakage)
leakage_raw = cp['model_state_dict']['_leakage_raw'].item()
leakage = 0.005 + 0.195 * (1 / (1 + np.exp(-leakage_raw)))
print(f"Leakage: {leakage:.4f}")

grn_matrix = de.numpy() * tw.unsqueeze(0).numpy() * (cm.numpy() + leakage)
print(f"GRN matrix: shape={grn_matrix.shape}")
print(f"  range: [{grn_matrix.min():.6f}, {grn_matrix.max():.6f}]")
print(f"  positive: {(grn_matrix > 0).sum()}")
print(f"  negative: {(grn_matrix < 0).sum()}")

if len(gene_names) > 0:
    sstr2_grn = grn_matrix[sstr2_idx, :]
    print(f"\nSSTR2 GRN effects:")
    print(f"  range: [{sstr2_grn.min():.6f}, {sstr2_grn.max():.6f}]")
    print(f"  positive: {(sstr2_grn > 0).sum()}")
    print(f"  negative: {(sstr2_grn < 0).sum()}")

# Compare with saved grn_matrix.npy
print("\n" + "=" * 70)
print("COMPARING WITH SAVED grn_matrix.npy")
print("=" * 70)
try:
    saved_grn = np.load('results/sclc_grn_v6/grn_matrix.npy')
    print(f"Saved GRN: shape={saved_grn.shape}")
    print(f"  range: [{saved_grn.min():.6f}, {saved_grn.max():.6f}]")
    
    # Check if they match
    diff = np.abs(grn_matrix - saved_grn).max()
    print(f"\nMax difference between computed and saved: {diff:.6f}")
    
    if len(gene_names) > 0:
        saved_sstr2 = saved_grn[sstr2_idx, :]
        print(f"\nSaved SSTR2 effects:")
        print(f"  range: [{saved_sstr2.min():.6f}, {saved_sstr2.max():.6f}]")
        print(f"  positive: {(saved_sstr2 > 0).sum()}")
        print(f"  negative: {(saved_sstr2 < 0).sum()}")
except Exception as e:
    print(f"Error loading saved GRN: {e}")

# Also check the fine-tuned model
print("\n" + "=" * 70)
print("CHECKING FINE-TUNED MODEL grn_matrix.npy")
print("=" * 70)
try:
    ft_grn = np.load('results/sclc_SSTR2/grn_matrix.npy')
    print(f"Fine-tuned GRN: shape={ft_grn.shape}")
    print(f"  range: [{ft_grn.min():.6f}, {ft_grn.max():.6f}]")
    print(f"  positive: {(ft_grn > 0).sum()}")
    print(f"  negative: {(ft_grn < 0).sum()}")
    
    if len(gene_names) > 0:
        ft_sstr2 = ft_grn[sstr2_idx, :]
        print(f"\nFine-tuned SSTR2 effects:")
        print(f"  range: [{ft_sstr2.min():.6f}, {ft_sstr2.max():.6f}]")
        print(f"  positive: {(ft_sstr2 > 0).sum()}")
        print(f"  negative: {(ft_sstr2 < 0).sum()}")
        
        # This is the problem - check why all effects are positive
        if (ft_sstr2 < 0).sum() == 0 and (saved_sstr2 < 0).sum() > 0:
            print("\n!!! WARNING: Fine-tuned model lost all negative effects !!!")
            print("This is a BUG in finetune_gene.py")
except Exception as e:
    print(f"Error loading fine-tuned GRN: {e}")


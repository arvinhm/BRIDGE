#!/usr/bin/env python3
"""Check if activator/repressor effects are balanced."""

import numpy as np
import pandas as pd

print("="*70)
print("EFFECT BALANCE ANALYSIS")
print("="*70)

# Load CORRECT GRN
grn = np.load('results/sclc/grn_matrix_CORRECT.npy')
gene_names = pd.read_csv('results/sclc/gene_names.csv', header=None)[0].tolist()
tf_names = pd.read_csv('results/sclc/tf_names.csv', header=None)[0].tolist()

print(f"GRN shape: {grn.shape}")
print(f"Overall range: [{grn.min():.4f}, {grn.max():.4f}]")

# Global statistics
positive = grn[grn > 0]
negative = grn[grn < 0]

print(f"\nGLOBAL STATISTICS:")
print(f"  Positive effects (activators): {len(positive):,}")
print(f"    Range: [{positive.min():.4f}, {positive.max():.4f}]")
print(f"    Mean: {positive.mean():.4f}")
print(f"  Negative effects (repressors): {len(negative):,}")
print(f"    Range: [{negative.min():.4f}, {negative.max():.4f}]")
print(f"    Mean: {negative.mean():.4f}")
print(f"  Zero effects: {(grn == 0).sum():,}")

# SSTR2 analysis
sstr2_idx = gene_names.index('SSTR2')
sstr2_effects = grn[sstr2_idx, :]

pos = sstr2_effects[sstr2_effects > 0]
neg = sstr2_effects[sstr2_effects < 0]

print(f"\nSSTR2 EFFECTS:")
print(f"  Positive (activators): {len(pos)}")
print(f"    Range: [{pos.min():.6f}, {pos.max():.6f}]")
print(f"    Mean: {pos.mean():.6f}")
print(f"    Sum: {pos.sum():.4f}")
print(f"  Negative (repressors): {len(neg)}")
print(f"    Range: [{neg.min():.6f}, {neg.max():.6f}]")
print(f"    Mean: {neg.mean():.6f}")
print(f"    Sum: {neg.sum():.4f}")

# The issue: are repressors too weak?
print(f"\nACTIVATOR/REPRESSOR BALANCE:")
activator_strength = np.abs(pos).mean() if len(pos) > 0 else 0
repressor_strength = np.abs(neg).mean() if len(neg) > 0 else 0
ratio = activator_strength / (repressor_strength + 1e-8)
print(f"  Mean activator magnitude: {activator_strength:.6f}")
print(f"  Mean repressor magnitude: {repressor_strength:.6f}")
print(f"  Ratio (activator/repressor): {ratio:.2f}x")

if ratio > 5:
    print(f"\n⚠️  WARNING: Activators are {ratio:.1f}x stronger than repressors!")
    print("   This imbalance may indicate a training issue or biological reality.")
else:
    print(f"\n✓ Balance looks reasonable (ratio < 5)")

# Top activators and repressors
print("\nTOP 10 ACTIVATORS:")
top_act_idx = np.argsort(sstr2_effects)[::-1][:10]
for i in top_act_idx:
    print(f"  {tf_names[i]}: +{sstr2_effects[i]:.6f}")

print("\nTOP 10 REPRESSORS:")
top_rep_idx = np.argsort(sstr2_effects)[:10]
for i in top_rep_idx:
    if sstr2_effects[i] < 0:
        print(f"  {tf_names[i]}: {sstr2_effects[i]:.6f}")


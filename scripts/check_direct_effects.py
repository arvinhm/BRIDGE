#!/usr/bin/env python3
"""Check direct_effects distribution for bias."""

import torch
import numpy as np

cp = torch.load('results/sclc/best_model.pt', map_location='cpu', weights_only=False)
state = cp['model_state_dict']

direct = state['direct_effects'].numpy()
print('DIRECT_EFFECTS ANALYSIS:')
print(f'  Shape: {direct.shape}')
print(f'  Range: [{direct.min():.4f}, {direct.max():.4f}]')
print(f'  Mean: {direct.mean():.6f}')
print(f'  Std: {direct.std():.6f}')

pos = direct[direct > 0]
neg = direct[direct < 0]
print(f'  Positive: {len(pos):,} (mean={pos.mean():.6f})')
print(f'  Negative: {len(neg):,} (mean={neg.mean():.6f})')
if len(neg) > 0:
    print(f'  Ratio: {pos.mean() / abs(neg.mean()):.2f}x')
else:
    print('  No negative values!')

# Check if the issue is in the decoder or direct effects
if 'effect_decoder.0.weight' in state:
    dec0 = state['effect_decoder.0.weight'].numpy()
    print()
    print('DECODER LAYER 0:')
    print(f'  Shape: {dec0.shape}')
    print(f'  Range: [{dec0.min():.4f}, {dec0.max():.4f}]')
    print(f'  Mean: {dec0.mean():.6f}')

# Check chip_mask
if 'chip_mask' in state:
    chip = state['chip_mask'].numpy()
    print()
    print('CHIP_MASK:')
    print(f'  Shape: {chip.shape}')
    print(f'  Non-zero: {(chip > 0).sum():,}')
    print(f'  Zero: {(chip == 0).sum():,}')

# The issue might be in how positive/negative effects interact with chip_mask
print()
print('EFFECT × CHIP_MASK INTERACTION:')
if 'chip_mask' in state and 'tf_specificity_weights' in state:
    chip = state['chip_mask'].numpy()
    spec = state['tf_specificity_weights'].numpy()
    
    # Compute weighted effects
    weighted = direct * spec[np.newaxis, :]
    
    # Split by chip binding
    bound = chip > 0
    unbound = chip == 0
    
    # Effects for bound genes
    bound_effects = weighted[bound]
    unbound_effects = weighted[unbound]
    
    print(f'  Bound TF-gene pairs: {bound.sum():,}')
    print(f'    Positive: {(bound_effects > 0).sum():,}')
    print(f'    Negative: {(bound_effects < 0).sum():,}')
    print(f'    Mean: {bound_effects.mean():.6f}')
    
    print(f'  Unbound TF-gene pairs: {unbound.sum():,}')
    print(f'    Mean: {unbound_effects.mean():.6f}')

# Check leakage
if '_leakage_raw' in state:
    leakage_raw = state['_leakage_raw'].item()
    leakage = 0.005 + 0.195 / (1 + np.exp(-leakage_raw))
    print()
    print(f'CHIP LEAKAGE: {leakage:.4f}')
    print('  (0.005 = min, 0.2 = max)')


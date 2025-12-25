#!/usr/bin/env python3
"""Check checkpoint contents."""

import torch
import numpy as np

# Load the v6 checkpoint
print("Loading v6 checkpoint...")
cp = torch.load('results/sclc_grn_v6/best_model.pt', map_location='cpu', weights_only=False)

print('Keys in v6 checkpoint:')
for k in sorted(cp.keys()):
    if k == 'model_state_dict':
        print(f'  {k}: {len(cp[k])} entries')
    else:
        val = cp[k]
        if hasattr(val, 'shape'):
            print(f'  {k}: {val.shape}')
        elif isinstance(val, dict):
            print(f'  {k}: dict with {len(val)} keys')
        else:
            print(f'  {k}: {val}')

print('\nKeys in model_state_dict:')
for k in sorted(cp['model_state_dict'].keys()):
    t = cp['model_state_dict'][k]
    if hasattr(t, 'shape'):
        print(f'  {k}: {t.shape}')
    else:
        print(f'  {k}: {type(t)}')

# Check specific buffers
print('\n--- Key buffers ---')
if 'tf_specificity_weights' in cp['model_state_dict']:
    w = cp['model_state_dict']['tf_specificity_weights']
    print(f'tf_specificity_weights: shape={w.shape}, range=[{w.min():.4f}, {w.max():.4f}]')
else:
    print('tf_specificity_weights: NOT FOUND')

if 'direct_effects' in cp['model_state_dict']:
    de = cp['model_state_dict']['direct_effects']
    print(f'direct_effects: shape={de.shape}, range=[{de.min():.4f}, {de.max():.4f}]')
    print(f'  Non-zero: {(de != 0).sum().item()} / {de.numel()}')
    print(f'  Positive: {(de > 0).sum().item()}, Negative: {(de < 0).sum().item()}')
else:
    print('direct_effects: NOT FOUND')

if 'chip_mask' in cp['model_state_dict']:
    cm = cp['model_state_dict']['chip_mask']
    print(f'chip_mask: shape={cm.shape}, non-zero={(cm > 0).sum().item()}')

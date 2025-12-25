#!/usr/bin/env python3
"""Check TF-TF cascade statistics."""
import numpy as np
import pickle

adj = np.load('data/tf_binding/tf_tf_adjacency.npy')
print(f'TF-TF adjacency shape: {adj.shape}')
print(f'Total edges: {(adj > 0).sum()}')
print(f'Average regulators per TF: {(adj > 0).sum(axis=0).mean():.1f}')
print(f'Average targets per TF: {(adj > 0).sum(axis=1).mean():.1f}')
print(f'TFs with >10 regulators: {((adj > 0).sum(axis=0) > 10).sum()}')
print(f'TFs with >50 targets: {((adj > 0).sum(axis=1) > 50).sum()}')

# Find top master regulators (TFs that regulate many other TFs)
with open('data/tf_binding/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
tf_names = mappings['tf_names']

targets_per_tf = (adj > 0).sum(axis=1)
top_regulators = np.argsort(targets_per_tf)[::-1][:10]

print("\nTop 10 Master Regulators (regulate most other TFs):")
for i, idx in enumerate(top_regulators):
    print(f"  {i+1}. {tf_names[idx]}: {int(targets_per_tf[idx])} downstream TFs")


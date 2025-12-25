#!/usr/bin/env python3
"""Fix missing metadata files for sclc_grn model."""

import pickle
import pandas as pd
import numpy as np
import torch
import json
import shutil
from pathlib import Path

MODEL_DIR = Path('results/sclc_grn')
DATA_DIR = Path('data')

print("Generating missing metadata for sclc_grn model...")

# Load mappings for TF names
with open(DATA_DIR / 'tf_binding/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
tf_names = mappings.get('tf_names', [])
print(f'Found {len(tf_names)} TFs')

# Save tf_names
pd.DataFrame({'tf': tf_names}).to_csv(MODEL_DIR / 'tf_names.csv', index=False)
print('Saved tf_names.csv')

# Copy other files
shutil.copy(DATA_DIR / 'sclc/gene_names.csv', MODEL_DIR)
shutil.copy(DATA_DIR / 'sclc/celltype_names.csv', MODEL_DIR)
print('Copied gene_names.csv and celltype_names.csv')

# Generate grn_matrix from model
print('Loading model checkpoint...')
checkpoint = torch.load(MODEL_DIR / 'best_model.pt', map_location='cpu')
state = checkpoint['model_state_dict']
direct_effects = state['direct_effects'].numpy()
print(f'Direct effects shape: {direct_effects.shape}')
np.save(MODEL_DIR / 'grn_matrix.npy', direct_effects)
print('Saved grn_matrix.npy')

# Save config
config = {
    'n_genes': int(direct_effects.shape[0]),
    'n_tfs': int(direct_effects.shape[1])
}
with open(MODEL_DIR / 'model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Saved model_config.json')

# List final contents
print(f"\nFiles in {MODEL_DIR}:")
for f in sorted(MODEL_DIR.iterdir()):
    print(f"  {f.name}")

print("\nDone! Now you can run analyze_gene.py")


#!/usr/bin/env python3
"""
Memory-efficient pseudo-bulk aggregation.

Computes cell-type means directly from sparse matrix without 
converting to dense. Saves only the aggregated means.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gc
import os

print("="*60)
print("PSEUDO-BULK AGGREGATION (Memory Efficient)")
print("="*60)

# Parameters
INPUT_FILE = 'adata_sclc_clustered.h5ad'
CELLTYPE_KEY = 'sclc_subcluster'
OUTPUT_DIR = 'data/sclc'
MIN_CELLS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load data in backed mode
# =============================================================================
print(f"\n[1/3] Loading {INPUT_FILE} (backed mode)...")
adata = sc.read(INPUT_FILE, backed='r')
print(f"    Shape: {adata.shape}")

# Get metadata
cell_types = adata.obs[CELLTYPE_KEY].astype(str).values
gene_names = list(adata.var_names)
unique_types = sorted(set(cell_types))
print(f"    Cell types: {len(unique_types)}")
print(f"    Genes: {len(gene_names)}")

# =============================================================================
# Compute pseudo-bulk means (sparse, chunked)
# =============================================================================
print(f"\n[2/3] Computing pseudo-bulk means...")

means_list = []
valid_types = []
n_genes = len(gene_names)

for i, ct in enumerate(unique_types):
    mask = cell_types == ct
    n_cells = mask.sum()
    
    if n_cells < MIN_CELLS:
        continue
    
    # Get indices
    indices = np.where(mask)[0]
    
    # Compute mean in chunks to save memory
    chunk_size = 5000
    running_sum = np.zeros(n_genes, dtype=np.float64)
    
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        chunk_idx = indices[start:end]
        
        # Get chunk of data (sparse)
        chunk = adata.X[chunk_idx]
        if sp.issparse(chunk):
            chunk_sum = np.array(chunk.sum(axis=0)).flatten()
        else:
            chunk_sum = chunk.sum(axis=0)
        
        running_sum += chunk_sum
    
    # Compute mean
    mean_expr = (running_sum / n_cells).astype(np.float32)
    means_list.append(mean_expr)
    valid_types.append(ct)
    
    if (i + 1) % 10 == 0:
        print(f"    Processed {i+1}/{len(unique_types)} cell types...")
    
    gc.collect()

print(f"    Valid cell types: {len(valid_types)}")

# Stack means
expression_means = np.stack(means_list, axis=0)
print(f"    Pseudo-bulk matrix: {expression_means.shape}")

# =============================================================================
# Save results
# =============================================================================
print(f"\n[3/3] Saving to {OUTPUT_DIR}...")

# Save expression means
np.save(os.path.join(OUTPUT_DIR, 'expression_means.npy'), expression_means)

# Save gene names
pd.DataFrame({'gene': gene_names}).to_csv(
    os.path.join(OUTPUT_DIR, 'gene_names.csv'), index=False
)

# Save cell type names
pd.DataFrame({'celltype': valid_types}).to_csv(
    os.path.join(OUTPUT_DIR, 'celltype_names.csv'), index=False
)

# Save a minimal h5ad with just the pseudo-bulk
import anndata as ad
adata_pb = ad.AnnData(
    X=expression_means,
    obs=pd.DataFrame({'celltype': valid_types}, index=valid_types),
    var=pd.DataFrame({'gene': gene_names}, index=gene_names)
)
adata_pb.obs['sclc_subcluster'] = valid_types  # Add the key train_v5 expects
adata_pb.write(os.path.join(OUTPUT_DIR, 'preprocessed.h5ad'))

print("\nSaved files:")
print(f"  - {OUTPUT_DIR}/expression_means.npy ({expression_means.shape})")
print(f"  - {OUTPUT_DIR}/gene_names.csv ({len(gene_names)} genes)")
print(f"  - {OUTPUT_DIR}/celltype_names.csv ({len(valid_types)} cell types)")
print(f"  - {OUTPUT_DIR}/preprocessed.h5ad (pseudo-bulk AnnData)")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"""
The pseudo-bulk data is ready for training!

Run:
  python scripts/train_v5.py \\
      --singlecell data/sclc/preprocessed.h5ad \\
      --tf-binding data/tf_binding/ \\
      --celltype-key sclc_subcluster \\
      --output results/sclc_grn/ \\
      --epochs 300
""")


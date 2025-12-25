#!/usr/bin/env python3
"""
MEMORY-EFFICIENT SCLC Subset + Clustering

Uses disk-backed reading and minimal memory operations.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import gc
import os

print("="*60)
print("SCLC GRN PREPARATION - MEMORY EFFICIENT")
print("="*60)

# =============================================================================
# STEP 1: EXTRACT SCLC CELL INDICES (minimal memory)
# =============================================================================
print("\n[1/5] Reading metadata only...")

# Read just obs (metadata) first - much smaller
adata_backed = sc.read('adata.scvi.h5ad', backed='r')
print(f"    Full dataset: {adata_backed.shape[0]:,} cells")

# Get SCLC mask
sclc_types = ['SCLC-A', 'SCLC-N', 'SCLC-P', 'NonNE SCLC']
mask = adata_backed.obs['cell_type_final'].isin(sclc_types)
sclc_indices = adata_backed.obs_names[mask].tolist()
print(f"    SCLC cells: {len(sclc_indices):,}")

# Save cell types for later
obs_df = adata_backed.obs.loc[sclc_indices, ['cell_type_final']].copy()
print("\n    Distribution:")
print(obs_df['cell_type_final'].value_counts())

# Check if X_scVI exists
has_scvi = 'X_scVI' in adata_backed.obsm
if has_scvi:
    scvi_emb = adata_backed.obsm['X_scVI'][mask].copy()
    print(f"    scVI embedding: {scvi_emb.shape}")
else:
    print("    WARNING: No X_scVI found!")
    scvi_emb = None

# Close backed file
del adata_backed
gc.collect()

# =============================================================================
# STEP 2: CLUSTERING ON SCVI (without full expression matrix)
# =============================================================================
print("\n[2/5] Hierarchical clustering on scVI...")

if scvi_emb is not None:
    # Create minimal AnnData with just embeddings for clustering
    import anndata as ad
    adata_mini = ad.AnnData(
        X=np.zeros((len(sclc_indices), 10), dtype=np.float32),  # Dummy X
        obs=obs_df
    )
    adata_mini.obsm['X_scVI'] = scvi_emb
    adata_mini.obs_names = sclc_indices
    
    resolution = 1.0
    adata_mini.obs['sclc_subcluster'] = ''
    
    total_clusters = 0
    
    for subtype in sclc_types:
        print(f"\n    {subtype}...")
        subtype_mask = adata_mini.obs['cell_type_final'] == subtype
        n_cells = subtype_mask.sum()
        print(f"      {n_cells:,} cells")
        
        if n_cells < 100:
            adata_mini.obs.loc[subtype_mask, 'sclc_subcluster'] = f"{subtype}_1"
            total_clusters += 1
            continue
        
        # Subset for clustering
        idx = adata_mini.obs_names[subtype_mask]
        sub = adata_mini[idx].copy()
        
        sc.pp.neighbors(sub, use_rep='X_scVI', n_neighbors=30)
        sc.tl.leiden(sub, resolution=resolution, key_added='leiden')
        
        n_sub = sub.obs['leiden'].nunique()
        print(f"      -> {n_sub} sub-clusters")
        total_clusters += n_sub
        
        # Map clusters by size
        counts = sub.obs['leiden'].value_counts()
        cluster_map = {old: str(i+1) for i, old in enumerate(counts.index)}
        # Convert categorical to string before concatenation
        labels = subtype + '_' + sub.obs['leiden'].astype(str).map(cluster_map)
        
        adata_mini.obs.loc[idx, 'sclc_subcluster'] = labels.values
        
        del sub
        gc.collect()
    
    # Save the clustering results
    cluster_df = adata_mini.obs[['cell_type_final', 'sclc_subcluster']].copy()
    cluster_df.to_csv('sclc_clusters.csv')
    print(f"\n    Saved cluster assignments: sclc_clusters.csv")
    print(f"    Total clusters: {total_clusters}")
    
    del adata_mini
    gc.collect()

# =============================================================================
# STEP 3: LOAD AND SUBSET IN CHUNKS
# =============================================================================
print("\n[3/5] Loading SCLC cells in chunks...")

# Read backed again
adata = sc.read('adata.scvi.h5ad', backed='r')

# Subset using the indices - this creates a view
adata_sclc = adata[sclc_indices]

print(f"    View created: {adata_sclc.shape}")

# =============================================================================
# STEP 4: CONVERT TO MEMORY AND ADD CLUSTERS
# =============================================================================
print("\n[4/5] Converting to in-memory (this is the memory-intensive step)...")
print("    If this fails, you need more RAM or use a cluster/cloud machine.")

try:
    # This is unavoidable - we need to materialize to save
    adata_sclc = adata_sclc.to_memory()
    print(f"    In-memory: {adata_sclc.shape}")
    
    # Add cluster annotations
    cluster_df = pd.read_csv('sclc_clusters.csv', index_col=0)
    adata_sclc.obs['sclc_subcluster'] = cluster_df.loc[adata_sclc.obs_names, 'sclc_subcluster']
    
    # Make X sparse if not already
    import scipy.sparse as sp
    if not sp.issparse(adata_sclc.X):
        print("    Converting X to sparse...")
        adata_sclc.X = sp.csr_matrix(adata_sclc.X)
    
    # Remove large objects to free memory before save
    del adata
    gc.collect()
    
except MemoryError:
    print("\n    MEMORY ERROR!")
    print("    The cluster assignments are saved in: sclc_clusters.csv")
    print("    Run this on a machine with more RAM to complete.")
    exit(1)

# =============================================================================
# STEP 5: SAVE
# =============================================================================
print("\n[5/5] Saving...")

# Clean up unnecessary data
for key in ['neighbors', 'umap', 'pca']:
    if key in adata_sclc.uns:
        del adata_sclc.uns[key]
for key in ['connectivities', 'distances']:
    if key in adata_sclc.obsp:
        del adata_sclc.obsp[key]

gc.collect()

adata_sclc.write('adata_sclc_clustered.h5ad', compression='gzip')
print("    Saved: adata_sclc_clustered.h5ad")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)

# Final summary
print(f"""
Output: adata_sclc_clustered.h5ad
  Cells: {adata_sclc.shape[0]:,}
  Genes: {adata_sclc.shape[1]:,}
  
Cluster column: 'sclc_subcluster'
""")

print(adata_sclc.obs['sclc_subcluster'].value_counts())

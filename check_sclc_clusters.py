#!/usr/bin/env python3
"""
Check SCLC annotations and optionally re-cluster using Leiden on scVI embedding.

For GRN training we need sufficient cell type diversity.
With only 4 SCLC subtypes, we may need finer clustering.

HIERARCHICAL APPROACH:
  For each SCLC subtype (SCLC-A, SCLC-N, etc.):
    1. Subset to that subtype
    2. Run Leiden clustering within
    3. Create labels like SCLC-A_1, SCLC-A_2, etc.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import gc

# Load SCLC subset
print("Loading adata_sclc_all.h5ad...")
adata = sc.read('adata_sclc_all.h5ad')
print(f"Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# =============================================================================
# CHECK EXISTING ANNOTATIONS
# =============================================================================
print("\n" + "="*60)
print("EXISTING ANNOTATIONS IN SCLC SUBSET")
print("="*60)

annotation_cols = ['cell_type_final', 'leiden_fine', 'phenograph', 'patient', 'sample']

for col in annotation_cols:
    if col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        print(f"\n{col}: {n_unique} unique values")
        print(adata.obs[col].value_counts().head(20))

# =============================================================================
# CHECK IF leiden_fine IS USEFUL
# =============================================================================
print("\n" + "="*60)
print("LEIDEN_FINE WITHIN EACH SCLC SUBTYPE")
print("="*60)

if 'leiden_fine' in adata.obs.columns:
    cross_tab = pd.crosstab(adata.obs['cell_type_final'], adata.obs['leiden_fine'])
    print(f"\nCross-tabulation shape: {cross_tab.shape}")
    print(f"(cell_type_final x leiden_fine clusters)")
    
    # Show clusters per SCLC subtype
    for ct in ['SCLC-A', 'SCLC-N', 'SCLC-P', 'NonNE SCLC']:
        if ct in cross_tab.index:
            n_clusters = (cross_tab.loc[ct] > 10).sum()  # Clusters with >10 cells
            print(f"\n{ct}: {n_clusters} leiden_fine clusters (>10 cells)")

# =============================================================================
# RECOMMENDATION
# =============================================================================
print("\n" + "="*60)
print("RECOMMENDATION FOR GRN TRAINING")
print("="*60)

n_celltypes = adata.obs['cell_type_final'].nunique()
n_leiden = adata.obs['leiden_fine'].nunique() if 'leiden_fine' in adata.obs.columns else 0

print(f"""
Current annotations:
  cell_type_final: {n_celltypes} types
  leiden_fine:     {n_leiden} clusters

For GRN training:
  - MINIMUM: 10+ cell types/clusters for meaningful virtual perturbations
  - IDEAL: 15-30 clusters covering biological heterogeneity
  - TOO MANY: 50+ clusters may be too noisy

OPTIONS:
""")

if n_leiden >= 10:
    print(f"  1. USE leiden_fine ({n_leiden} clusters) - Already available!")
    print(f"     → Use --celltype-key leiden_fine in training")
elif n_leiden > 0 and n_leiden < 10:
    print(f"  1. leiden_fine has only {n_leiden} clusters - may need re-clustering")
else:
    print(f"  1. No leiden_fine available - need to cluster")

print("""
  2. HIERARCHICAL RE-CLUSTER (recommended!)
     → Run with --recluster flag
     → Creates SCLC-A_1, SCLC-A_2, SCLC-N_1, etc.

  3. USE cell_type_final + patient combination
     → Run with --combine flag
""")

# =============================================================================
# OPTION: HIERARCHICAL RE-CLUSTER ON scVI
# =============================================================================
import sys
if '--recluster' in sys.argv:
    print("\n" + "="*60)
    print("HIERARCHICAL RE-CLUSTERING ON scVI EMBEDDING")
    print("="*60)
    
    # Check if scVI embedding exists
    if 'X_scVI' not in adata.obsm:
        print("ERROR: X_scVI not found! Cannot re-cluster.")
        sys.exit(1)
    
    print(f"scVI embedding shape: {adata.obsm['X_scVI'].shape}")
    
    # Resolution for sub-clustering (higher = more clusters)
    resolution = 0.8  # Adjust as needed
    
    sclc_subtypes = ['SCLC-A', 'SCLC-N', 'SCLC-P', 'NonNE SCLC']
    
    # Initialize new column
    adata.obs['sclc_subcluster'] = ''
    
    total_clusters = 0
    
    for subtype in sclc_subtypes:
        print(f"\n{'='*40}")
        print(f"Processing {subtype}...")
        print(f"{'='*40}")
        
        # Get cells of this subtype
        mask = adata.obs['cell_type_final'] == subtype
        n_cells = mask.sum()
        print(f"  Cells: {n_cells:,}")
        
        if n_cells < 100:
            print(f"  Too few cells, keeping as single cluster")
            adata.obs.loc[mask, 'sclc_subcluster'] = f"{subtype}_1"
            total_clusters += 1
            continue
        
        # Subset to this subtype
        adata_sub = adata[mask].copy()
        
        # Compute neighbors on scVI for this subset
        print(f"  Computing neighbors...")
        sc.pp.neighbors(adata_sub, use_rep='X_scVI', n_neighbors=30)
        
        # Run Leiden
        print(f"  Running Leiden (resolution={resolution})...")
        sc.tl.leiden(adata_sub, resolution=resolution, key_added='leiden_sub')
        
        n_sub_clusters = adata_sub.obs['leiden_sub'].nunique()
        print(f"  Found {n_sub_clusters} sub-clusters")
        
        # Create labels like SCLC-A_1, SCLC-A_2, etc.
        # Sort by cluster size (largest first)
        cluster_counts = adata_sub.obs['leiden_sub'].value_counts()
        cluster_mapping = {old: str(i+1) for i, old in enumerate(cluster_counts.index)}
        
        new_labels = adata_sub.obs['leiden_sub'].map(cluster_mapping)
        new_labels = subtype + '_' + new_labels
        
        # Assign back to main adata
        adata.obs.loc[mask, 'sclc_subcluster'] = new_labels.values
        
        total_clusters += n_sub_clusters
        
        # Show distribution
        print(f"  Sub-cluster sizes:")
        for i, (old_cluster, count) in enumerate(cluster_counts.items()):
            print(f"    {subtype}_{i+1}: {count:,} cells")
        
        # Clean up
        del adata_sub
        gc.collect()
    
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING COMPLETE")
    print("="*60)
    print(f"\nTotal sub-clusters: {total_clusters}")
    print(f"\nFinal cluster distribution:")
    print(adata.obs['sclc_subcluster'].value_counts())
    
    # Summary by main subtype
    print("\n" + "="*40)
    print("SUMMARY BY SCLC SUBTYPE")
    print("="*40)
    for subtype in sclc_subtypes:
        mask = adata.obs['sclc_subcluster'].str.startswith(subtype)
        n_clusters = adata.obs.loc[mask, 'sclc_subcluster'].nunique()
        n_cells = mask.sum()
        print(f"  {subtype}: {n_clusters} clusters, {n_cells:,} cells")
    
    # Save
    print("\nSaving with new clusters...")
    adata.write('adata_sclc_all.h5ad')
    print("Done!")
    print("\nUse --celltype-key sclc_subcluster for training")
    print(f"Example: python scripts/train_v5.py --celltype-key sclc_subcluster ...")

# =============================================================================
# OPTION: CREATE COMBINED ANNOTATION (subtype + patient)
# =============================================================================
if '--combine' in sys.argv:
    print("\n" + "="*60)
    print("CREATING COMBINED ANNOTATION (subtype + patient)")
    print("="*60)
    
    # Combine SCLC subtype + patient
    adata.obs['sclc_patient'] = (
        adata.obs['cell_type_final'].astype(str) + '_' + 
        adata.obs['patient'].astype(str)
    )
    n_combined = adata.obs['sclc_patient'].nunique()
    print(f"Combined annotation: {n_combined} unique groups")
    
    # Show distribution
    print("\nGroup sizes:")
    counts = adata.obs['sclc_patient'].value_counts()
    print(counts.head(30))
    
    # Filter to groups with enough cells
    valid_groups = counts[counts >= 50].index
    print(f"\nGroups with >=50 cells: {len(valid_groups)}")
    
    # Save
    adata.write('adata_sclc_all.h5ad')
    print("\nSaved! Use --celltype-key sclc_patient for training")

#!/usr/bin/env python3
"""Verify all data files are correctly shaped and aligned."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

print("="*70)
print("DATA VERIFICATION FOR $200K EXPERIMENT")
print("="*70)

# === TF BINDING ===
print("\n[1] TF BINDING DATA")
print("-"*40)

mappings = pickle.load(open('data/tf_binding/mappings.pkl', 'rb'))
tf_names = mappings['tf_names']
gene_names = mappings['gene_names']
print(f"TF names: {len(tf_names)}")
print(f"Gene names: {len(gene_names)}")

binding_matrix = np.load('data/tf_binding/binding_matrix.npy')
print(f"Binding matrix: {binding_matrix.shape}")
print(f"  Expected: ({len(gene_names)}, {len(tf_names)})")
if binding_matrix.shape != (len(gene_names), len(tf_names)):
    print("  ⚠️  MISMATCH!")
else:
    print("  ✓ Shape matches")

tf_tf_adj = np.load('data/tf_binding/tf_tf_adjacency.npy')
print(f"TF-TF adjacency: {tf_tf_adj.shape}")
print(f"  Expected: ({len(tf_names)}, {len(tf_names)})")
if tf_tf_adj.shape != (len(tf_names), len(tf_names)):
    print("  ⚠️  MISMATCH!")
else:
    print("  ✓ Shape matches")

# === PPI DATA ===
print("\n[2] PPI DATA")
print("-"*40)

cof_to_tf = np.load('data/ppi/cof_to_tf_adj.npy')
cof_indices = np.load('data/ppi/cofactor_gene_indices.npy')
tf_tf_ppi = np.load('data/ppi/tf_tf_ppi.npy')
cof_info = pd.read_csv('data/ppi/cofactor_info.csv')

print(f"Cofactor->TF adjacency: {cof_to_tf.shape}")
print(f"  Expected: (n_cofactors, {len(tf_names)})")
print(f"  Cofactor indices: {len(cof_indices)}")
print(f"  Cofactor info rows: {len(cof_info)}")

if cof_to_tf.shape[0] != len(cof_indices):
    print("  ⚠️  MISMATCH: cof_to_tf rows != cofactor_indices!")
else:
    print("  ✓ Cofactor dimensions consistent")

if cof_to_tf.shape[1] != len(tf_names):
    print(f"  ⚠️  MISMATCH: cof_to_tf cols ({cof_to_tf.shape[1]}) != n_tfs ({len(tf_names)})")
else:
    print("  ✓ TF dimension matches")

print(f"TF-TF PPI: {tf_tf_ppi.shape}")
if tf_tf_ppi.shape != (len(tf_names), len(tf_names)):
    print("  ⚠️  MISMATCH!")
else:
    print("  ✓ Shape matches")

# Verify cofactor indices are valid
max_idx = max(cof_indices)
print(f"Cofactor gene index range: [0, {max_idx}]")
print(f"  Gene list length: {len(gene_names)}")
if max_idx >= len(gene_names):
    print("  ⚠️  INVALID: cofactor indices exceed gene list!")
else:
    print("  ✓ All cofactor indices valid")

# === TF SPECIFICITY ===
print("\n[3] TF SPECIFICITY (JASPAR)")
print("-"*40)

tf_spec = np.load('data/tf_specificity/tf_specificity_weights.npy')
print(f"TF specificity weights: {tf_spec.shape}")
print(f"  Expected: ({len(tf_names)},)")
if tf_spec.shape[0] != len(tf_names):
    print(f"  ⚠️  MISMATCH! {tf_spec.shape[0]} != {len(tf_names)}")
else:
    print("  ✓ Shape matches")

print(f"  Range: [{tf_spec.min():.4f}, {tf_spec.max():.4f}]")
print(f"  Mean: {tf_spec.mean():.4f}")

# === SINGLE CELL ===
print("\n[4] SINGLE CELL DATA")
print("-"*40)

import scanpy as sc
adata = sc.read_h5ad('data/sclc/preprocessed.h5ad')
print(f"AnnData: {adata.shape}")
print(f"  Cell types: {adata.shape[0]}")
print(f"  Genes: {adata.shape[1]}")

# Check gene overlap
sc_genes = set(adata.var_names)
binding_genes = set(gene_names)
overlap = sc_genes & binding_genes
print(f"  Gene overlap with binding: {len(overlap)}/{len(gene_names)}")

# === ALIGNMENT CHECK ===
print("\n[5] CRITICAL ALIGNMENTS")
print("-"*40)

# Check that TF names in different sources match
tfs_in_genes = [tf for tf in tf_names if tf in binding_genes]
print(f"TFs that are also in gene list: {len(tfs_in_genes)}/{len(tf_names)}")

# Check SSTR2 exists
if 'SSTR2' in sc_genes:
    print("✓ SSTR2 found in single-cell data")
else:
    print("⚠️  SSTR2 NOT FOUND in single-cell data!")

if 'SSTR2' in binding_genes:
    print("✓ SSTR2 found in binding gene list")
else:
    print("⚠️  SSTR2 NOT FOUND in binding gene list!")

# === SUMMARY ===
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
DATA DIMENSIONS:
  TFs: {len(tf_names)}
  Genes (binding): {len(gene_names)}
  Genes (single-cell): {adata.shape[1]}
  Cell types: {adata.shape[0]}
  Cofactors: {len(cof_indices)}

KEY MATRICES:
  Binding (gene×TF): {binding_matrix.shape}
  TF-TF regulatory: {tf_tf_adj.shape}
  TF-TF PPI: {tf_tf_ppi.shape}
  Cofactor->TF: {cof_to_tf.shape}
  TF specificity: {tf_spec.shape}

All dimensions should be consistent for the model to work correctly!
""")

# Check for any zeros or NaN issues
print("\n[6] DATA QUALITY")
print("-"*40)
print(f"Binding matrix: non-zero={(binding_matrix > 0).sum()} ({100*(binding_matrix > 0).sum()/binding_matrix.size:.2f}%)")
print(f"TF-TF adj: non-zero={(tf_tf_adj != 0).sum()}")
print(f"TF-TF PPI: non-zero={(tf_tf_ppi != 0).sum()}")
print(f"Cofactor->TF: non-zero={(cof_to_tf != 0).sum()}")

# Check for NaN
has_nan = False
if np.isnan(binding_matrix).any():
    print("⚠️  Binding matrix has NaN!")
    has_nan = True
if np.isnan(tf_tf_adj).any():
    print("⚠️  TF-TF adjacency has NaN!")
    has_nan = True
if np.isnan(tf_spec).any():
    print("⚠️  TF specificity has NaN!")
    has_nan = True
if not has_nan:
    print("✓ No NaN values detected in any matrix")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)


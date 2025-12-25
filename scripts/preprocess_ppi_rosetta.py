#!/usr/bin/env python3
"""
Preprocess Rosetta PPI predictions for GRN model integration.

This creates:
1. cof_to_tf_adj.npy: (n_cofactors, n_tfs) row-normalized adjacency
2. cofactor_gene_indices.npy: Gene indices for cofactors in expression matrix
3. tf_tf_ppi.npy: (n_tfs, n_tfs) TF-TF PPI adjacency (optional, for cascade augmentation)

Usage:
    python scripts/preprocess_ppi_rosetta.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def normalize_name(name: str) -> str:
    """Normalize protein/gene name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    name = name.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')
    return name


def load_ppi_predictions(ppi_file: str, min_prob: float = 0.5) -> pd.DataFrame:
    """Load PPI predictions with minimum AF2 probability."""
    print(f"Loading PPI predictions from {ppi_file}...")
    
    # Read file, skip comment lines
    with open(ppi_file, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
    
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(lines)), sep='\t')
    
    print(f"  Total PPIs: {len(df):,}")
    
    # Filter by AF2 probability
    df_filtered = df[df['AFprob'] >= min_prob].copy()
    print(f"  High confidence (AFprob >= {min_prob}): {len(df_filtered):,}")
    
    return df_filtered


def load_data(base_dir: Path):
    """Load TF names, gene names, and JASPAR classification."""
    
    # Load TF names from mappings.pkl (same source as training)
    mappings_path = base_dir / 'data' / 'tf_binding' / 'mappings.pkl'
    if mappings_path.exists():
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        tf_names = mappings.get('tf_names', [])
        gene_names = mappings.get('gene_names', [])
    else:
        # Fallback to results
        tf_names_path = base_dir / 'results' / 'sclc_grn' / 'tf_names.csv'
        gene_names_path = base_dir / 'results' / 'sclc_grn' / 'gene_names.csv'
        tf_names = pd.read_csv(tf_names_path, header=None)[0].tolist()
        gene_names = pd.read_csv(gene_names_path, header=None)[0].tolist()
    
    print(f"  TFs: {len(tf_names)}, Genes: {len(gene_names)}")
    
    # Load JASPAR classification (DNA-binding vs cofactor)
    jaspar_matched_path = base_dir / 'data' / 'tf_specificity' / 'jaspar_matched_tfs.csv'
    if jaspar_matched_path.exists():
        jaspar_df = pd.read_csv(jaspar_matched_path)
        col = 'tf' if 'tf' in jaspar_df.columns else jaspar_df.columns[0]
        dna_binding_tfs = set(jaspar_df[col].dropna().astype(str).str.upper())
    else:
        dna_binding_tfs = set()
    
    print(f"  DNA-binding TFs (JASPAR): {len(dna_binding_tfs)}")
    
    # Cofactors are TFs not in JASPAR
    tf_set = {str(tf).upper() for tf in tf_names}
    cofactors = tf_set - dna_binding_tfs
    print(f"  Cofactors (no JASPAR): {len(cofactors)}")
    
    return tf_names, gene_names, dna_binding_tfs, cofactors


def build_matrices(ppi_df: pd.DataFrame, 
                   tf_names: list, 
                   gene_names: list,
                   dna_binding_tfs: set,
                   cofactors: set):
    """Build PPI matrices for model integration."""
    
    # Create lookup dictionaries
    tf_to_idx = {str(tf).upper(): i for i, tf in enumerate(tf_names)}
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    
    # Also need to identify which TFs are genes in the expression matrix
    tf_in_genes = {}
    for tf in tf_names:
        tf_upper = str(tf).upper()
        if tf_upper in gene_to_idx:
            tf_in_genes[tf_upper] = gene_to_idx[tf_upper]
    
    print(f"  TFs found in gene list: {len(tf_in_genes)}")
    
    # Identify cofactors that are in the gene list
    cofactor_list = []
    cofactor_gene_indices = []
    for cof in sorted(cofactors):
        if cof in gene_to_idx:
            cofactor_list.append(cof)
            cofactor_gene_indices.append(gene_to_idx[cof])
    
    print(f"  Cofactors in gene list: {len(cofactor_list)}")
    
    # Build cofactor -> TF adjacency
    n_cofactors = len(cofactor_list)
    n_tfs = len(tf_names)
    cof_to_tf = np.zeros((n_cofactors, n_tfs), dtype=np.float32)
    
    # Also build TF-TF PPI adjacency
    tf_tf_ppi = np.zeros((n_tfs, n_tfs), dtype=np.float32)
    
    cofactor_to_idx = {cof: i for i, cof in enumerate(cofactor_list)}
    
    # Process PPI predictions
    edges_added_cof_tf = 0
    edges_added_tf_tf = 0
    
    for _, row in ppi_df.iterrows():
        name1 = str(row['Name1']).upper() if pd.notna(row['Name1']) else ""
        name2 = str(row['Name2']).upper() if pd.notna(row['Name2']) else ""
        prob = float(row['AFprob']) if pd.notna(row['AFprob']) else 0
        
        # Case 1: Cofactor -> DNA-binding TF
        if name1 in cofactor_to_idx and name2 in tf_to_idx and name2 in dna_binding_tfs:
            cof_idx = cofactor_to_idx[name1]
            tf_idx = tf_to_idx[name2]
            cof_to_tf[cof_idx, tf_idx] = max(cof_to_tf[cof_idx, tf_idx], prob)
            edges_added_cof_tf += 1
            
        # Case 2: DNA-binding TF -> Cofactor (reverse direction)
        if name2 in cofactor_to_idx and name1 in tf_to_idx and name1 in dna_binding_tfs:
            cof_idx = cofactor_to_idx[name2]
            tf_idx = tf_to_idx[name1]
            cof_to_tf[cof_idx, tf_idx] = max(cof_to_tf[cof_idx, tf_idx], prob)
            edges_added_cof_tf += 1
            
        # Case 3: TF-TF interaction (both are TFs)
        if name1 in tf_to_idx and name2 in tf_to_idx:
            idx1 = tf_to_idx[name1]
            idx2 = tf_to_idx[name2]
            tf_tf_ppi[idx1, idx2] = max(tf_tf_ppi[idx1, idx2], prob)
            tf_tf_ppi[idx2, idx1] = max(tf_tf_ppi[idx2, idx1], prob)  # Symmetric
            edges_added_tf_tf += 1
    
    print(f"  Cofactor-TF edges: {edges_added_cof_tf}")
    print(f"  TF-TF PPI edges: {edges_added_tf_tf}")
    
    # Row-normalize adjacencies
    row_sum = cof_to_tf.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # Avoid division by zero
    cof_to_tf_norm = cof_to_tf / row_sum
    
    row_sum_tf = tf_tf_ppi.sum(axis=1, keepdims=True)
    row_sum_tf[row_sum_tf == 0] = 1
    tf_tf_ppi_norm = tf_tf_ppi / row_sum_tf
    
    return (cof_to_tf_norm, np.array(cofactor_gene_indices), 
            cofactor_list, tf_tf_ppi_norm)


def main():
    base_dir = Path(__file__).parent.parent
    ppi_dir = base_dir / 'final_predictions'
    output_dir = base_dir / 'data' / 'ppi'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prefer 90% precision file
    ppi_file = ppi_dir / 'final_predictions_90.tsv'
    if not ppi_file.exists():
        ppi_file = ppi_dir / 'final_predictions_80.tsv'
    
    if not ppi_file.exists():
        print("ERROR: No PPI predictions file found!")
        sys.exit(1)
    
    print("=" * 70)
    print("ROSETTA PPI PREPROCESSING FOR GRN MODEL")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    tf_names, gene_names, dna_binding_tfs, cofactors = load_data(base_dir)
    
    # Load PPI predictions
    print("\n[2/3] Loading PPI predictions...")
    ppi_df = load_ppi_predictions(str(ppi_file), min_prob=0.5)
    
    # Build matrices
    print("\n[3/3] Building PPI matrices...")
    cof_to_tf, cofactor_gene_indices, cofactor_list, tf_tf_ppi = build_matrices(
        ppi_df, tf_names, gene_names, dna_binding_tfs, cofactors
    )
    
    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    np.save(output_dir / 'cof_to_tf_adj.npy', cof_to_tf)
    print(f"  cof_to_tf_adj.npy: {cof_to_tf.shape} (row-normalized)")
    
    np.save(output_dir / 'cofactor_gene_indices.npy', cofactor_gene_indices)
    print(f"  cofactor_gene_indices.npy: {len(cofactor_gene_indices)} cofactors")
    
    np.save(output_dir / 'tf_tf_ppi.npy', tf_tf_ppi)
    print(f"  tf_tf_ppi.npy: {tf_tf_ppi.shape} (row-normalized)")
    
    pd.DataFrame({'cofactor': cofactor_list, 'gene_idx': cofactor_gene_indices}).to_csv(
        output_dir / 'cofactor_info.csv', index=False
    )
    print(f"  cofactor_info.csv: cofactor names and gene indices")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    n_cof_with_tf = (cof_to_tf.sum(axis=1) > 0).sum()
    print(f"  Cofactors with TF partners: {n_cof_with_tf}/{len(cofactor_list)}")
    
    avg_partners = cof_to_tf.sum(axis=1)[cof_to_tf.sum(axis=1) > 0].mean()
    print(f"  Average TF partners per cofactor: {avg_partners:.1f}")
    
    n_tf_with_ppi = (tf_tf_ppi.sum(axis=1) > 0).sum()
    print(f"  TFs with PPI connections: {n_tf_with_ppi}/{len(tf_names)}")
    
    print("\n" + "=" * 70)
    print("DONE! Use --ppi data/ppi/ flag in training")
    print("=" * 70)


if __name__ == '__main__':
    main()


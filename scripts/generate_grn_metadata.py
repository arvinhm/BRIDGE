#!/usr/bin/env python3
"""
Generate GRN metadata files from the trained model checkpoint.
Run this after training to create gene_names.csv, tf_names.csv, grn_matrix.npy
"""

import scanpy as sc
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='results/sclc_grn')
    parser.add_argument('--singlecell', type=str, default='adata_sclc_clustered.h5ad')
    parser.add_argument('--tf-binding', type=str, default='data/tf_binding')
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    args = parser.parse_args()
    
    output_dir = Path(args.model_dir)
    
    # Load original data to get gene names
    print('Loading original scRNA-seq data...')
    adata = sc.read_h5ad(args.singlecell)
    gene_names = list(adata.var_names)
    print(f'  Genes: {len(gene_names)}')
    
    # Load TF names
    print('Loading TF binding data...')
    with open(Path(args.tf_binding) / 'mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    tf_names = mappings['tf_names']
    print(f'  TFs: {len(tf_names)}')
    
    # Get cell types
    celltype_names = sorted(adata.obs[args.celltype_key].unique())
    print(f'  Cell types: {len(celltype_names)}')
    for ct in celltype_names:
        print(f'    - {ct}')
    
    # Save metadata CSVs
    print('\nSaving metadata files...')
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False)
    pd.DataFrame({'celltype': celltype_names}).to_csv(output_dir / 'celltype_names.csv', index=False)
    print('  Saved gene_names.csv, tf_names.csv, celltype_names.csv')
    
    # Load model and extract GRN matrix
    print('\nLoading model checkpoint...')
    checkpoint_path = output_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / 'final_model.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    
    # Extract direct_effects matrix (this is the GRN)
    direct_effects = state['direct_effects'].numpy()
    print(f'  GRN matrix shape: {direct_effects.shape}')
    print(f'  Non-zero edges: {(np.abs(direct_effects) > 0.001).sum():,}')
    
    # Save GRN matrix
    np.save(output_dir / 'grn_matrix.npy', direct_effects)
    print(f'  Saved grn_matrix.npy')
    
    # Save model config
    if 'config' in checkpoint:
        import json
        with open(output_dir / 'model_config.json', 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        print('  Saved model_config.json')
    
    # Print training metrics
    if 'metrics' in checkpoint:
        print('\nFinal model metrics:')
        for k, v in checkpoint['metrics'].items():
            print(f'  {k}: {v:.4f}')
    
    # Search for interesting genes
    print('\n' + '='*60)
    print('SAMPLE GENE SEARCHES:')
    print('='*60)
    
    # T cell markers
    t_cell_genes = ['Cd3d', 'Cd3e', 'Cd3g', 'Cd4', 'Cd8a', 'Cd8b1', 'Trac', 'Trbc1', 'Trbc2']
    print('\nT cell markers found:')
    for gene in t_cell_genes:
        matches = [g for g in gene_names if g.lower() == gene.lower()]
        if matches:
            print(f'  {matches[0]} - FOUND')
        else:
            # Partial match
            partial = [g for g in gene_names if gene.lower() in g.lower()]
            if partial:
                print(f'  {gene} - partial matches: {partial[:3]}')
    
    # Immune TFs
    immune_tfs = ['Stat1', 'Stat3', 'Stat5a', 'Stat5b', 'Nfkb1', 'Rela', 'Irf1', 'Irf4', 'Tbx21', 'Gata3', 'Rorc', 'Foxp3', 'Bcl6']
    print('\nImmune TFs found:')
    for tf in immune_tfs:
        matches = [t for t in tf_names if t.lower() == tf.lower()]
        if matches:
            print(f'  {matches[0]} - FOUND')
    
    print('\n' + '='*60)
    print(f'All metadata saved to {output_dir}')
    print('='*60)


if __name__ == '__main__':
    main()


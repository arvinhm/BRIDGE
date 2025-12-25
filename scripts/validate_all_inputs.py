#!/usr/bin/env python3
"""
COMPREHENSIVE INPUT VALIDATION FOR $200K EXPERIMENT
=====================================================
Validates ALL input data files before training.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

def validate_all():
    print("=" * 80)
    print("COMPREHENSIVE INPUT VALIDATION")
    print("=" * 80)
    
    all_ok = True
    warnings = []
    errors = []
    
    # =========================================================================
    # 1. TF BINDING DATA (ChIP-Atlas)
    # =========================================================================
    print("\n[1/6] TF BINDING DATA (ChIP-Atlas)")
    print("-" * 50)
    
    try:
        mappings = pickle.load(open('data/tf_binding/mappings.pkl', 'rb'))
        tf_names = mappings['tf_names']
        gene_names_binding = mappings['gene_names']
        
        print(f"  ✓ TF names: {len(tf_names)}")
        print(f"  ✓ Gene names: {len(gene_names_binding)}")
        
        # Check for key TFs (known SSTR2 regulators)
        key_tfs = ['REST', 'NEUROD1', 'MYCN', 'ASCL1', 'FOXA2', 'NEUROG2', 'SOX2']
        found_key = [tf for tf in key_tfs if tf in tf_names]
        missing_key = [tf for tf in key_tfs if tf not in tf_names]
        
        print(f"  ✓ Key TFs found: {found_key}")
        if missing_key:
            warnings.append(f"Key TFs missing from binding data: {missing_key}")
            print(f"  ⚠ Missing: {missing_key}")
        
        # Binding matrix
        binding = np.load('data/tf_binding/binding_matrix.npy')
        print(f"  ✓ Binding matrix: {binding.shape}")
        print(f"    Non-zero: {(binding > 0).sum():,} ({100*(binding > 0).sum()/binding.size:.2f}%)")
        
        if binding.shape != (len(gene_names_binding), len(tf_names)):
            errors.append(f"Binding matrix shape mismatch: {binding.shape} vs ({len(gene_names_binding)}, {len(tf_names)})")
        
        # TF-TF adjacency
        tf_tf_adj = np.load('data/tf_binding/tf_tf_adjacency.npy')
        print(f"  ✓ TF-TF adjacency: {tf_tf_adj.shape}")
        print(f"    Non-zero edges: {(tf_tf_adj != 0).sum():,}")
        
        # Check SSTR2 in genes
        if 'SSTR2' in gene_names_binding:
            sstr2_idx = gene_names_binding.index('SSTR2')
            sstr2_tfs = (binding[sstr2_idx] > 0).sum()
            print(f"  ✓ SSTR2 has {sstr2_tfs} TF binding sites")
        else:
            errors.append("SSTR2 not found in binding gene list!")
            
    except Exception as e:
        errors.append(f"TF binding loading failed: {e}")
        print(f"  ✗ Error: {e}")
    
    # =========================================================================
    # 2. PPI DATA (Rosetta)
    # =========================================================================
    print("\n[2/6] PPI DATA (Rosetta)")
    print("-" * 50)
    
    try:
        cof_to_tf = np.load('data/ppi/cof_to_tf_adj.npy')
        cof_indices = np.load('data/ppi/cofactor_gene_indices.npy')
        tf_tf_ppi = np.load('data/ppi/tf_tf_ppi.npy')
        cof_info = pd.read_csv('data/ppi/cofactor_info.csv')
        
        print(f"  ✓ Cofactor→TF: {cof_to_tf.shape}")
        print(f"    Non-zero interactions: {(cof_to_tf != 0).sum():,}")
        print(f"  ✓ Cofactor indices: {len(cof_indices)}")
        print(f"  ✓ TF-TF PPI: {tf_tf_ppi.shape}")
        print(f"    Non-zero interactions: {(tf_tf_ppi != 0).sum():,}")
        
        # Check dimensions
        if cof_to_tf.shape[0] != len(cof_indices):
            errors.append(f"Cofactor dimension mismatch: {cof_to_tf.shape[0]} vs {len(cof_indices)}")
        if cof_to_tf.shape[1] != len(tf_names):
            errors.append(f"PPI TF dimension mismatch: {cof_to_tf.shape[1]} vs {len(tf_names)}")
            
        # Check for key cofactors
        key_cofactors = ['EP300', 'CREBBP', 'HDAC1', 'HDAC2', 'CTBP1', 'CTBP2', 'KDM1A']
        if 'cofactor' in cof_info.columns:
            found_cof = [c for c in key_cofactors if c in cof_info['cofactor'].values]
            print(f"  ✓ Key cofactors found: {found_cof}")
            
    except Exception as e:
        errors.append(f"PPI loading failed: {e}")
        print(f"  ✗ Error: {e}")
    
    # =========================================================================
    # 3. JASPAR SPECIFICITY
    # =========================================================================
    print("\n[3/6] JASPAR SPECIFICITY")
    print("-" * 50)
    
    try:
        tf_spec = np.load('data/tf_specificity/tf_specificity_weights.npy')
        print(f"  ✓ TF specificity: {tf_spec.shape}")
        print(f"    Range: [{tf_spec.min():.4f}, {tf_spec.max():.4f}]")
        print(f"    Mean: {tf_spec.mean():.4f}")
        
        if tf_spec.shape[0] != len(tf_names):
            errors.append(f"Specificity dimension mismatch: {tf_spec.shape[0]} vs {len(tf_names)}")
        
        # Check for NaN
        if np.isnan(tf_spec).any():
            errors.append("TF specificity contains NaN values!")
        
        # Load analysis
        jaspar_analysis = pd.read_csv('data/tf_specificity/tf_specificity_analysis.csv')
        matched = (jaspar_analysis['jaspar_matched'] == True).sum() if 'jaspar_matched' in jaspar_analysis.columns else 'unknown'
        print(f"  ✓ JASPAR matched TFs: {matched}")
        
    except Exception as e:
        errors.append(f"JASPAR loading failed: {e}")
        print(f"  ✗ Error: {e}")
    
    # =========================================================================
    # 4. SINGLE-CELL DATA
    # =========================================================================
    print("\n[4/6] SINGLE-CELL DATA")
    print("-" * 50)
    
    try:
        import scanpy as sc
        adata = sc.read_h5ad('data/sclc/preprocessed.h5ad')
        
        print(f"  ✓ AnnData shape: {adata.shape}")
        print(f"    Cell types: {adata.shape[0]}")
        print(f"    Genes: {adata.shape[1]}")
        
        # Check SSTR2
        adata_genes = [g.upper() for g in adata.var_names]
        if 'SSTR2' in adata_genes:
            sstr2_idx = adata_genes.index('SSTR2')
            if hasattr(adata.X, 'toarray'):
                expr = adata.X.toarray()[:, sstr2_idx]
            else:
                expr = np.array(adata.X)[:, sstr2_idx]
            print(f"  ✓ SSTR2 expression: [{expr.min():.3f}, {expr.max():.3f}]")
            print(f"    Mean: {expr.mean():.3f}, Std: {expr.std():.3f}")
            
            # Check variance
            if expr.std() < 0.1:
                warnings.append(f"SSTR2 has low variance ({expr.std():.3f}) - may be hard to predict")
        else:
            errors.append("SSTR2 not found in single-cell data!")
            
        # Check key TFs in expression
        key_tfs_in_adata = [tf for tf in key_tfs if tf in adata_genes]
        print(f"  ✓ Key TFs in expression: {len(key_tfs_in_adata)}/{len(key_tfs)}")
        
        # Gene overlap with binding
        overlap = set([g.upper() for g in gene_names_binding]) & set(adata_genes)
        print(f"  ✓ Gene overlap with binding: {len(overlap)}/{len(gene_names_binding)}")
        
    except Exception as e:
        errors.append(f"Single-cell loading failed: {e}")
        print(f"  ✗ Error: {e}")
    
    # =========================================================================
    # 5. DATA QUALITY CHECKS
    # =========================================================================
    print("\n[5/6] DATA QUALITY CHECKS")
    print("-" * 50)
    
    try:
        # Check for NaN in all matrices
        matrices = {
            'binding': binding,
            'tf_tf_adj': tf_tf_adj,
            'cof_to_tf': cof_to_tf,
            'tf_tf_ppi': tf_tf_ppi,
            'tf_specificity': tf_spec
        }
        
        for name, mat in matrices.items():
            if np.isnan(mat).any():
                errors.append(f"{name} contains NaN!")
            elif np.isinf(mat).any():
                errors.append(f"{name} contains Inf!")
            else:
                print(f"  ✓ {name}: No NaN/Inf")
        
        # Check row normalization of PPI
        cof_row_sums = cof_to_tf.sum(axis=1)
        if (cof_row_sums > 0).any():
            max_sum = cof_row_sums.max()
            if max_sum > 1.01:
                warnings.append(f"Cofactor→TF not row-normalized: max sum = {max_sum:.4f}")
            else:
                print(f"  ✓ Cofactor→TF row-normalized (max sum: {max_sum:.4f})")
                
    except Exception as e:
        print(f"  ⚠ Quality check error: {e}")
    
    # =========================================================================
    # 6. BIOLOGICAL SANITY CHECKS
    # =========================================================================
    print("\n[6/6] BIOLOGICAL SANITY CHECKS (SSTR2 regulators)")
    print("-" * 50)
    
    try:
        # Known SSTR2 regulators from literature
        known_activators = ['NEUROD1', 'NEUROG2', 'ASCL1', 'SOX2', 'PAX6']
        known_repressors = ['REST', 'MYCN', 'MYC']
        
        print("  Known SSTR2 regulators (from literature):")
        print(f"    Activators: {known_activators}")
        print(f"    Repressors: {known_repressors}")
        
        # Check binding for these TFs to SSTR2
        sstr2_idx = gene_names_binding.index('SSTR2') if 'SSTR2' in gene_names_binding else -1
        if sstr2_idx >= 0:
            print("\n  ChIP binding to SSTR2 promoter:")
            for tf in known_activators + known_repressors:
                if tf in tf_names:
                    tf_idx = tf_names.index(tf)
                    binds = binding[sstr2_idx, tf_idx] > 0
                    print(f"    {tf}: {'YES' if binds else 'no'}")
                else:
                    print(f"    {tf}: (not in TF list)")
        
        print("\n  ⚡ After training, verify these TFs appear as top regulators!")
        
    except Exception as e:
        print(f"  ⚠ Sanity check error: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print("\n❌ ERRORS (must fix before training):")
        for e in errors:
            print(f"   • {e}")
        all_ok = False
    else:
        print("\n✓ No critical errors found!")
    
    if warnings:
        print("\n⚠ WARNINGS (may affect results):")
        for w in warnings:
            print(f"   • {w}")
    else:
        print("✓ No warnings!")
    
    print("\n" + "=" * 80)
    
    if all_ok:
        print("✓ ALL VALIDATIONS PASSED - Ready for training!")
    else:
        print("✗ FIX ERRORS BEFORE TRAINING!")
        
    print("=" * 80)
    
    return all_ok

if __name__ == '__main__':
    success = validate_all()
    sys.exit(0 if success else 1)


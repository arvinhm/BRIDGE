#!/usr/bin/env python3
"""
Analyze Rosetta PPI predictions to connect cofactors to DNA-binding TFs.

This completes the regulatory puzzle:
- DNA-binding TFs → Genes (ChIP-seq)
- TF-TF cascade (from ChIP data)
- DNA-binding TF motif specificity (JASPAR)
- Cofactor → DNA-binding TF (PPI predictions) ← THIS SCRIPT

For cofactors (TFs without JASPAR motifs), their effect on genes is
MEDIATED through their DNA-binding TF partners.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

def normalize_name(name: str) -> str:
    """Normalize protein/gene name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    # Remove common suffixes
    name = re.sub(r'_HUMAN$', '', name)
    name = re.sub(r'_MOUSE$', '', name)
    # Remove hyphens, underscores, dots
    name = name.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')
    return name


def load_ppi_predictions(ppi_file: str, min_prob: float = 0.8) -> pd.DataFrame:
    """Load high-confidence PPI predictions."""
    print(f"Loading PPI predictions from {ppi_file}...")
    
    # Read file, skip comment lines
    with open(ppi_file, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
    
    # Parse TSV
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(lines)), sep='\t')
    
    print(f"  Total PPIs: {len(df):,}")
    
    # Filter by AF2 probability (most reliable)
    df_filtered = df[df['AFprob'] >= min_prob].copy()
    print(f"  High confidence (AFprob >= {min_prob}): {len(df_filtered):,}")
    
    return df_filtered


def load_tf_lists(tf_specificity_dir: str):
    """Load JASPAR-matched and unmatched TFs."""
    matched_file = os.path.join(tf_specificity_dir, 'jaspar_matched_tfs.csv')
    unmatched_file = os.path.join(tf_specificity_dir, 'jaspar_unmatched_tfs.csv')
    
    # Load matched (DNA-binding) TFs
    dna_binding_tfs = set()
    if os.path.exists(matched_file):
        df = pd.read_csv(matched_file)
        # Try different column names
        col = 'tf' if 'tf' in df.columns else 'tf_name' if 'tf_name' in df.columns else df.columns[0]
        dna_binding_tfs = set(df[col].dropna().astype(str).str.upper())
        print(f"  DNA-binding TFs (JASPAR matched): {len(dna_binding_tfs)}")
    
    # Load unmatched (potential cofactors)
    cofactors = set()
    if os.path.exists(unmatched_file):
        df = pd.read_csv(unmatched_file)
        # Try different column names
        col = 'tf' if 'tf' in df.columns else 'tf_name' if 'tf_name' in df.columns else df.columns[0]
        cofactors = set(df[col].dropna().astype(str).str.upper())
        print(f"  Potential cofactors (no JASPAR match): {len(cofactors)}")
    
    return dna_binding_tfs, cofactors


def load_all_tfs(base_dir: str) -> list:
    """Load all TF names - PRIORITIZE mappings.pkl (same source as training!)."""
    import pickle
    
    # First try mappings.pkl (authoritative source)
    mappings_path = os.path.join(base_dir, 'data', 'tf_binding', 'mappings.pkl')
    if os.path.exists(mappings_path):
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        tfs = mappings.get('tf_names', [])
        if tfs:
            print(f"  Loaded {len(tfs)} TFs from {mappings_path} (training source)")
            return tfs
    
    # Fallback to results CSVs
    results_dir = os.path.join(base_dir, 'results')
    possible_paths = [
        os.path.join(results_dir, 'sclc_grn', 'tf_names.csv'),
        os.path.join(results_dir, 'sclc_grn_jaspar', 'tf_names.csv'),
        os.path.join(results_dir, 'grn_model_v5', 'tf_names.csv'),
    ]
    
    for tf_names_file in possible_paths:
        if os.path.exists(tf_names_file):
            df = pd.read_csv(tf_names_file, header=None)
            tfs = list(df[0].values)
            print(f"  Loaded {len(tfs)} TFs from {tf_names_file} (results fallback)")
            return tfs
    
    return []


def build_cofactor_tf_matrix(ppi_df: pd.DataFrame, 
                              all_tfs: list,
                              dna_binding_tfs: set,
                              cofactors: set) -> tuple:
    """
    Build matrix of Cofactor → DNA-binding TF interactions.
    
    Returns:
        cofactor_tf_matrix: (n_cofactors, n_dna_binding) interaction scores
        cofactor_names: list of cofactor names
        dna_binding_names: list of DNA-binding TF names
    """
    # Create normalized lookup
    tf_to_idx = {normalize_name(tf): i for i, tf in enumerate(all_tfs)}
    tf_names_upper = {tf.upper() for tf in all_tfs}
    
    # Filter to TFs in our list
    dna_binding_in_list = [tf for tf in dna_binding_tfs if tf in tf_names_upper]
    cofactors_in_list = [tf for tf in cofactors if tf in tf_names_upper]
    
    print(f"\n  TFs in our binding data:")
    print(f"    DNA-binding: {len(dna_binding_in_list)}")
    print(f"    Cofactors: {len(cofactors_in_list)}")
    
    # Build interaction matrix
    # For each cofactor, find its DNA-binding TF partners
    cofactor_partners = defaultdict(list)  # cofactor -> [(dna_tf, prob), ...]
    
    for _, row in ppi_df.iterrows():
        name1 = str(row['Name1']).upper() if pd.notna(row['Name1']) else ""
        name2 = str(row['Name2']).upper() if pd.notna(row['Name2']) else ""
        prob = float(row['AFprob']) if pd.notna(row['AFprob']) else 0
        
        # Check if this is a Cofactor-DNA_binding_TF interaction
        # Case 1: Name1 is cofactor, Name2 is DNA-binding
        if name1 in cofactors and name2 in dna_binding_tfs:
            cofactor_partners[name1].append((name2, prob))
        # Case 2: Name2 is cofactor, Name1 is DNA-binding
        elif name2 in cofactors and name1 in dna_binding_tfs:
            cofactor_partners[name2].append((name1, prob))
        # Case 3: Both are DNA-binding (TF-TF interaction)
        # Case 4: Both are cofactors (cofactor complex)
    
    # Report statistics
    cofactors_with_partners = [c for c, partners in cofactor_partners.items() if len(partners) > 0]
    print(f"\n  Cofactors with DNA-binding TF partners: {len(cofactors_with_partners)}")
    
    if cofactors_with_partners:
        avg_partners = np.mean([len(v) for v in cofactor_partners.values() if len(v) > 0])
        print(f"  Average DNA-binding partners per cofactor: {avg_partners:.1f}")
    
    return cofactor_partners, dna_binding_in_list, cofactors_in_list


def compute_cofactor_weights(cofactor_partners: dict,
                            dna_binding_tfs: list,
                            all_tfs: list,
                            existing_weights: np.ndarray = None) -> np.ndarray:
    """
    Compute specificity weights for cofactors based on their TF partners.
    
    Logic:
    - A cofactor with FEW, SPECIFIC DNA-binding TF partners → HIGH weight
    - A cofactor with MANY, DIVERSE partners → LOWER weight (less specific)
    - The weight also considers the specificity of partner TFs
    """
    tf_to_idx = {str(tf).upper(): i for i, tf in enumerate(all_tfs)}
    n_tfs = len(all_tfs)
    
    # Start with existing weights or ones
    # Handle size mismatch by creating new array if needed
    if existing_weights is not None:
        if len(existing_weights) == n_tfs:
            weights = existing_weights.copy()
        else:
            print(f"  Warning: weight size mismatch ({len(existing_weights)} vs {n_tfs}), using minimum")
            min_len = min(len(existing_weights), n_tfs)
            weights = np.ones(n_tfs)
            weights[:min_len] = existing_weights[:min_len]
    else:
        weights = np.ones(n_tfs)
    
    # Compute cofactor-specific weights
    cofactor_weights = {}
    
    for cofactor, partners in cofactor_partners.items():
        if not partners:
            continue
            
        # Number of partners (more specific if fewer)
        n_partners = len(partners)
        
        # Average interaction probability
        avg_prob = np.mean([p[1] for p in partners])
        
        # Get partner specificities (if available)
        partner_weights = []
        for partner_tf, prob in partners:
            if partner_tf in tf_to_idx:
                idx = tf_to_idx[partner_tf]
                if existing_weights is not None:
                    partner_weights.append(existing_weights[idx])
        
        # Specificity score:
        # - Fewer partners = higher (inverse of log)
        # - Higher avg probability = higher
        # - Partners with higher specificity = higher
        partner_specificity = np.mean(partner_weights) if partner_weights else 1.0
        
        # Formula: weight = prob * partner_specificity * (1 / log(n_partners + 1))
        specificity = avg_prob * partner_specificity * (1 / np.log(n_partners + 2))
        
        # Normalize to ~1.0 range (multiply by scaling factor)
        cofactor_weights[cofactor] = specificity * 1.5  # Scale factor
    
    # Apply to weights array
    for cofactor, spec_weight in cofactor_weights.items():
        if cofactor in tf_to_idx:
            idx = tf_to_idx[cofactor]
            # Combine with existing weight (if any)
            weights[idx] = 0.5 * weights[idx] + 0.5 * spec_weight
    
    # Normalize weights to mean=1
    weights = weights / weights.mean()
    
    return weights, cofactor_weights


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    ppi_dir = base_dir / 'final_predictions'
    results_dir = base_dir / 'results'
    tf_specificity_dir = base_dir / 'data' / 'tf_specificity'
    output_dir = tf_specificity_dir
    
    # Check for PPI file - use 80% for better coverage (more interactions)
    ppi_file = ppi_dir / 'final_predictions_80.tsv'  # 80% precision (more coverage)
    if not ppi_file.exists():
        ppi_file = ppi_dir / 'final_predictions_90.tsv'  # 90% precision fallback
    
    if not ppi_file.exists():
        print("ERROR: No PPI predictions file found!")
        print(f"  Expected: {ppi_dir / 'final_predictions_90.tsv'}")
        sys.exit(1)
    
    print("=" * 70)
    print("COFACTOR-TF PPI ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading TF lists...")
    dna_binding_tfs, cofactors = load_tf_lists(str(tf_specificity_dir))
    all_tfs = load_all_tfs(str(base_dir))  # Use base_dir to find mappings.pkl
    print(f"  Total TFs in binding matrix: {len(all_tfs)}")
    
    print("\n[2] Loading PPI predictions...")
    ppi_df = load_ppi_predictions(str(ppi_file), min_prob=0.5)  # Lower threshold for more coverage
    
    print("\n[3] Building Cofactor-TF interaction matrix...")
    cofactor_partners, dna_binding_in_list, cofactors_in_list = build_cofactor_tf_matrix(
        ppi_df, all_tfs, dna_binding_tfs, cofactors
    )
    
    print("\n[4] Loading existing JASPAR weights...")
    existing_weights = None
    jaspar_weights_file = tf_specificity_dir / 'tf_specificity_weights.npy'
    if jaspar_weights_file.exists():
        existing_weights = np.load(str(jaspar_weights_file))
        print(f"  Loaded {len(existing_weights)} weights from JASPAR analysis")
    
    print("\n[5] Computing combined specificity weights...")
    combined_weights, cofactor_weight_dict = compute_cofactor_weights(
        cofactor_partners, dna_binding_in_list, all_tfs, existing_weights
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined weights
    combined_weights_file = output_dir / 'tf_specificity_weights_ppi.npy'
    np.save(str(combined_weights_file), combined_weights)
    print(f"\n  Saved combined weights: {combined_weights_file}")
    
    # Save cofactor-TF interactions
    interactions = []
    for cofactor, partners in cofactor_partners.items():
        for partner_tf, prob in partners:
            interactions.append({
                'cofactor': cofactor,
                'dna_binding_tf': partner_tf,
                'af2_prob': prob
            })
    
    if interactions:
        interactions_df = pd.DataFrame(interactions)
        interactions_file = output_dir / 'cofactor_tf_interactions.csv'
        interactions_df.to_csv(str(interactions_file), index=False)
        print(f"  Saved interactions: {interactions_file}")
    
    # Save analysis summary
    summary_file = output_dir / 'ppi_analysis_stats.txt'
    with open(summary_file, 'w') as f:
        f.write(f"PPI COFACTOR-TF ANALYSIS\n")
        f.write(f"========================\n\n")
        f.write(f"DNA-binding TFs (JASPAR): {len(dna_binding_tfs)}\n")
        f.write(f"Cofactors (no JASPAR): {len(cofactors)}\n")
        f.write(f"High-confidence PPIs: {len(ppi_df):,}\n")
        f.write(f"\nCofactors with DNA-binding partners: {len(cofactor_partners)}\n")
        if cofactor_partners:
            f.write(f"Average partners per cofactor: {np.mean([len(v) for v in cofactor_partners.values()]):.1f}\n")
        f.write(f"\nCombined weight range: [{combined_weights.min():.3f}, {combined_weights.max():.3f}]\n")
        f.write(f"Combined weight mean: {combined_weights.mean():.3f}\n")
    print(f"  Saved summary: {summary_file}")
    
    # Print top cofactor interactions
    print("\n" + "=" * 70)
    print("TOP COFACTOR-TF INTERACTIONS")
    print("=" * 70)
    
    # Sort by number of partners
    top_cofactors = sorted(cofactor_partners.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    
    for cofactor, partners in top_cofactors:
        top_partners = sorted(partners, key=lambda x: x[1], reverse=True)[:5]
        partner_str = ", ".join([f"{p[0]}({p[1]:.2f})" for p in top_partners])
        print(f"  {cofactor}: {len(partners)} partners → {partner_str}")
    
    # Specific examples mentioned by user
    print("\n" + "=" * 70)
    print("EXAMPLE: RCOR2 (CoREST complex)")
    print("=" * 70)
    if 'RCOR2' in cofactor_partners:
        partners = cofactor_partners['RCOR2']
        print(f"  RCOR2 has {len(partners)} DNA-binding TF partners:")
        for tf, prob in sorted(partners, key=lambda x: x[1], reverse=True):
            print(f"    → {tf}: {prob:.3f}")
    else:
        print("  RCOR2 not found in high-confidence PPIs")
        # Check if it appears at all
        rcor2_any = ppi_df[(ppi_df['Name1'].str.upper() == 'RCOR2') | 
                          (ppi_df['Name2'].str.upper() == 'RCOR2')]
        if len(rcor2_any) > 0:
            print(f"  But found in {len(rcor2_any)} lower-confidence PPIs")
    
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION")
    print("=" * 70)
    
    # Compare DNA-binding vs cofactor weights
    dna_binding_indices = [i for i, tf in enumerate(all_tfs) if tf.upper() in dna_binding_tfs]
    cofactor_indices = [i for i, tf in enumerate(all_tfs) if tf.upper() in cofactors]
    
    if dna_binding_indices:
        dna_weights = combined_weights[dna_binding_indices]
        print(f"  DNA-binding TFs: mean={dna_weights.mean():.3f}, std={dna_weights.std():.3f}")
    
    if cofactor_indices:
        cof_weights = combined_weights[cofactor_indices]
        print(f"  Cofactors: mean={cof_weights.mean():.3f}, std={cof_weights.std():.3f}")
    
    print("\n  IMPORTANT: Cofactors are NOT diluted!")
    print("  Their weights are based on PPI partner specificity")
    print("  A cofactor with specific TF partners → HIGH weight")
    
    print("\n" + "=" * 70)
    print("DONE! Use tf_specificity_weights_ppi.npy for training")
    print("=" * 70)


if __name__ == '__main__':
    main()


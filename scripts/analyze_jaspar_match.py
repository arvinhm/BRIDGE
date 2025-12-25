#!/usr/bin/env python3
"""
Analyze JASPAR motif matching with our TF binding model.
Computes TF specificity weights based on:
1. JASPAR motif Information Content (higher = more specific motif)
2. TF-Gene binding matrix sparsity (fewer targets = more specific)

IMPROVED MATCHING:
- Case insensitive
- Removes hyphens, underscores, dots
- Handles heterodimers (e.g., RXRA::VDR matches both RXRA and VDR)
- Handles fusion proteins (e.g., EWSR1-FLI1 matches both EWSR1 and FLI1)
- Handles numbered variants (e.g., SOX2 matches SOX-2)

Output: TF specificity weights for model training
"""

import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from collections import defaultdict
import pickle
import math
from typing import Optional


def normalize_tf_name(name: str) -> str:
    """
    Normalize TF name for matching:
    - Uppercase
    - Remove hyphens, underscores, dots, spaces
    - Handle common variations
    """
    name = str(name).upper()
    # Remove hyphens, underscores, dots, spaces
    name = re.sub(r'[-_.\s]', '', name)
    return name


def get_tf_variants(name: str) -> set:
    """
    Generate multiple variants of a TF name for fuzzy matching.
    Returns a set of normalized names to try.
    """
    variants = set()
    
    # Original name (normalized)
    norm = normalize_tf_name(name)
    variants.add(norm)
    
    # Handle heterodimers (RXRA::VDR)
    if '::' in name:
        for component in name.split('::'):
            variants.add(normalize_tf_name(component))
    
    # Handle fusion proteins with hyphen (EWSR1-FLI1)
    if '-' in name:
        for component in name.split('-'):
            comp_norm = normalize_tf_name(component)
            if len(comp_norm) >= 2:  # Skip single letters
                variants.add(comp_norm)
    
    # Handle numbered variants (RCOR2 -> RCOR-2 -> RCOR2)
    # This is already handled by removing hyphens
    
    # Try adding/removing trailing numbers for family matching
    # e.g., SOX2 -> SOX (but we don't want to lose specificity)
    
    return variants


def extract_tf_name(header_value: str) -> str:
    """
    Extract TF name from various JASPAR header formats:
    - "Arnt" -> "Arnt"
    - "RXRA::VDR" -> "RXRA::VDR"
    - "MA0002.1.RUNX1" -> "RUNX1"
    - "MA0006.1.Ahr::Arnt" -> "Ahr::Arnt"
    """
    header_value = header_value.strip()
    
    # Check if format is "MA####.#.TF_NAME" (new format)
    if re.match(r'^MA\d+\.\d+\.', header_value):
        # Extract everything after "MA####.#."
        match = re.match(r'^MA\d+\.\d+\.(.+)$', header_value)
        if match:
            return match.group(1)
    
    return header_value


def parse_jaspar(jaspar_path: str) -> dict:
    """
    Parse JASPAR format file.
    
    ROBUST APPROACH:
    - Collect ALL motif versions for each TF (mouse Sox2, human SOX2, etc.)
    - Compute IC for each version
    - Store AVERAGE IC across all versions (more robust than single version)
    - Also store the version count for confidence weighting
    
    This handles:
    - Multiple species (mouse/human) with same TF name
    - Multiple JASPAR versions (MA0143.1, MA0143.2, etc.)
    - Heterodimers (separated into individual TFs)
    
    Format: {normalized_name: {'ids': [list], 'names': [list], 'matrices': [list], 
                               'avg_ic': float, 'max_ic': float, 'n_versions': int,
                               'is_heterodimer_only': bool}}
    """
    # First pass: collect ALL entries for each normalized TF name
    tf_collections = defaultdict(list)
    
    current_id = None
    current_name = None
    current_matrix = []
    entries_parsed = 0
    
    with open(jaspar_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous TF if exists
                if current_id is not None and len(current_matrix) == 4:
                    is_heterodimer = '::' in current_name
                    entry = {
                        'id': current_id,
                        'name': current_name,
                        'matrix': np.array(current_matrix),
                        'is_heterodimer': is_heterodimer,
                        'ic': compute_normalized_ic(np.array(current_matrix))
                    }
                    
                    # Add under all variant names
                    for variant in get_tf_variants(current_name):
                        tf_collections[variant].append(entry)
                    
                    entries_parsed += 1
                
                # Parse header
                parts = line[1:].split('\t')
                if len(parts) >= 2:
                    current_id = parts[0].strip()
                    raw_name = parts[1].strip()
                    current_name = extract_tf_name(raw_name)
                else:
                    current_id = parts[0].strip()
                    current_name = extract_tf_name(parts[0].strip())
                
                current_matrix = []
            
            elif line.startswith(('A', 'C', 'G', 'T')):
                match = re.search(r'\[([\d\s]+)\]', line)
                if match:
                    values = [int(x) for x in match.group(1).split()]
                    current_matrix.append(values)
        
        # Save last TF
        if current_id is not None and len(current_matrix) == 4:
            is_heterodimer = '::' in current_name
            entry = {
                'id': current_id,
                'name': current_name,
                'matrix': np.array(current_matrix),
                'is_heterodimer': is_heterodimer,
                'ic': compute_normalized_ic(np.array(current_matrix))
            }
            for variant in get_tf_variants(current_name):
                tf_collections[variant].append(entry)
            entries_parsed += 1
    
    # Second pass: aggregate entries for each TF
    tfs = {}
    for tf_name, entries in tf_collections.items():
        # Separate standalone vs heterodimer entries
        standalone = [e for e in entries if not e['is_heterodimer']]
        heterodimer = [e for e in entries if e['is_heterodimer']]
        
        # Prefer standalone entries if available
        if standalone:
            use_entries = standalone
            is_heterodimer_only = False
        else:
            use_entries = heterodimer
            is_heterodimer_only = True
        
        if not use_entries:
            continue
        
        # Compute aggregate statistics across ALL versions
        ics = [e['ic'] for e in use_entries]
        avg_ic = np.mean(ics)
        max_ic = np.max(ics)
        
        # Use the entry with highest IC as representative
        best_entry = max(use_entries, key=lambda x: x['ic'])
        
        tfs[tf_name] = {
            'id': best_entry['id'],
            'name': best_entry['name'],
            'matrix': best_entry['matrix'],
            'is_heterodimer': is_heterodimer_only,
            'avg_ic': avg_ic,
            'max_ic': max_ic,
            'n_versions': len(use_entries),
            'all_ids': [e['id'] for e in use_entries]
        }
    
    return tfs, entries_parsed


def compute_information_content(pfm: np.ndarray, pseudocount: float = 0.25) -> float:
    """
    Compute total Information Content (IC) of a Position Frequency Matrix.
    
    IC = sum over positions of: sum over nucleotides of f_i * log2(f_i / 0.25)
    
    Higher IC = more specific motif (less entropy, more defined binding preference)
    
    Returns: Total IC in bits (max = 2 * length for perfectly specific motif)
    """
    # Add pseudocount and normalize
    pfm = pfm.astype(float)
    pfm = pfm + pseudocount
    
    # Normalize each position
    ppm = pfm / pfm.sum(axis=0, keepdims=True)
    
    # Background frequency (uniform)
    bg = 0.25
    
    # Compute IC per position
    total_ic = 0.0
    for pos in range(ppm.shape[1]):
        for nuc in range(4):
            freq = ppm[nuc, pos]
            if freq > 0:
                total_ic += freq * math.log2(freq / bg)
    
    return total_ic


def compute_normalized_ic(pfm: np.ndarray, pseudocount: float = 0.25) -> float:
    """
    Compute normalized IC (0-1 scale, 1 = perfectly specific).
    """
    total_ic = compute_information_content(pfm, pseudocount)
    length = pfm.shape[1]
    max_ic = 2.0 * length  # Max IC for a perfectly specific motif
    return total_ic / max_ic


def main():
    script_dir = Path(__file__).parent.parent
    
    # Look for JASPAR file (prefer .jaspar over .txt)
    jaspar_path = script_dir / "JASPAR.jaspar"
    if not jaspar_path.exists():
        jaspar_path = script_dir / "JASPAR.txt"
    if not jaspar_path.exists():
        raise FileNotFoundError(f"JASPAR file not found in {script_dir}")
    
    tf_binding_dir = script_dir / "data" / "tf_binding"
    results_dir = script_dir / "results" / "sclc_grn"
    output_dir = script_dir / "data" / "tf_specificity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("JASPAR MOTIF MATCHING ANALYSIS (IMPROVED)")
    print("=" * 70)
    print(f"Using JASPAR file: {jaspar_path.name}")
    
    # 1. Parse JASPAR
    print("\n[1/5] Parsing JASPAR file...")
    jaspar_tfs, entries_parsed = parse_jaspar(str(jaspar_path))
    print(f"   JASPAR entries parsed: {entries_parsed}")
    print(f"   Unique normalized names (with variants): {len(jaspar_tfs)}")
    
    # 2. Load our TF names
    print("\n[2/5] Loading our TF binding model...")
    
    # PRIORITIZE mappings.pkl (same source as training!) to guarantee order match
    mappings_path = tf_binding_dir / "mappings.pkl"
    tf_names_path = results_dir / "tf_names.csv"
    
    if mappings_path.exists():
        # Use mappings.pkl - the authoritative source training uses
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        our_tfs = mappings.get('tf_names', [])
        print(f"   Loaded from: {mappings_path} (training source)")
    elif tf_names_path.exists():
        # Fallback to results CSV
        tf_df = pd.read_csv(tf_names_path)
        our_tfs = tf_df['tf'].tolist() if 'tf' in tf_df.columns else tf_df.iloc[:, 0].tolist()
        print(f"   Loaded from: {tf_names_path} (results fallback)")
    else:
        raise FileNotFoundError("Cannot find TF names in mappings.pkl or tf_names.csv")
    
    print(f"   Our TFs: {len(our_tfs)}")
    
    # 3. Load binding matrix for sparsity calculation
    print("\n[3/5] Loading TF-Gene binding matrix...")
    binding_path = tf_binding_dir / "binding_matrix.npy"
    if binding_path.exists():
        binding_matrix = np.load(binding_path)
        print(f"   Binding matrix: {binding_matrix.shape}")
        
        # Compute number of target genes per TF
        targets_per_tf = (binding_matrix > 0).sum(axis=0)
        n_genes = binding_matrix.shape[0]
    else:
        print("   WARNING: Binding matrix not found, using uniform weights")
        targets_per_tf = None
        n_genes = 18894  # Default from stats
    
    # 4. Match TFs and compute specificity
    print("\n[4/5] Matching TFs and computing specificity...")
    
    matched = []
    unmatched_ours = []
    
    for idx, tf in enumerate(our_tfs):
        tf_variants = get_tf_variants(tf)
        
        record = {
            'tf': tf,
            'idx': idx,
            'jaspar_match': None,
            'jaspar_id': None,
            'motif_ic': None,
            'motif_ic_norm': None,
            'n_targets': None,
            'target_fraction': None,
        }
        
        # Try to match in JASPAR using any variant
        jaspar_match = None
        for variant in tf_variants:
            if variant in jaspar_tfs:
                jaspar_match = jaspar_tfs[variant]
                break
        
        if jaspar_match:
            record['jaspar_match'] = jaspar_match['name']
            record['jaspar_id'] = jaspar_match['id']
            record['n_jaspar_versions'] = jaspar_match.get('n_versions', 1)
            record['all_jaspar_ids'] = ','.join(jaspar_match.get('all_ids', [jaspar_match['id']]))
            # Use AVERAGE IC across all versions (mouse + human) for robustness
            record['motif_ic_norm'] = jaspar_match.get('avg_ic', compute_normalized_ic(jaspar_match['matrix']))
            record['motif_ic_max'] = jaspar_match.get('max_ic', record['motif_ic_norm'])
            record['motif_ic'] = record['motif_ic_norm'] * 2 * jaspar_match['matrix'].shape[1]  # Approx total IC
        
        # Get binding sparsity
        if targets_per_tf is not None:
            record['n_targets'] = int(targets_per_tf[idx])
            record['target_fraction'] = targets_per_tf[idx] / n_genes
        
        if record['jaspar_match'] is not None:
            matched.append(record)
        else:
            unmatched_ours.append(record)
    
    # Summary
    print(f"\n   MATCHING RESULTS:")
    print(f"   ================")
    print(f"   Matched in JASPAR: {len(matched)} / {len(our_tfs)} ({100*len(matched)/len(our_tfs):.1f}%)")
    print(f"   Unmatched: {len(unmatched_ours)}")
    
    # Show some unmatched for debugging
    if len(unmatched_ours) > 0:
        print(f"\n   Sample unmatched TFs (first 20):")
        for rec in unmatched_ours[:20]:
            print(f"      {rec['tf']}")
    
    # 5. Compute combined specificity weights
    print("\n[5/5] Computing combined specificity weights...")
    
    all_records = matched + unmatched_ours
    
    # For unmatched TFs, use median IC from matched ones
    matched_ics = [r['motif_ic_norm'] for r in matched if r['motif_ic_norm'] is not None]
    median_ic = np.median(matched_ics) if matched_ics else 0.5
    print(f"   Median IC (for unmatched TFs): {median_ic:.3f}")
    
    # IMPORTANT: For unmatched TFs, we DON'T penalize them!
    # We give them the median IC so they are NOT diluted.
    # The weight will be based on binding sparsity only.
    
    # Compute specificity score for each TF
    for record in all_records:
        # Motif specificity (higher IC = more specific)
        if record['motif_ic_norm'] is not None:
            motif_spec = record['motif_ic_norm']
        else:
            # UNMATCHED: Use median IC (neutral, not penalized!)
            motif_spec = median_ic
        
        # Binding specificity (fewer targets = more specific)
        if record['target_fraction'] is not None:
            # Transform: high fraction -> low specificity
            binding_spec = 1.0 - record['target_fraction']
        else:
            binding_spec = 0.5  # Neutral
        
        # Combined weight: geometric mean (moderate blending)
        combined = np.sqrt(motif_spec * binding_spec)
        
        # Normalize to reasonable range [0.7, 1.3]
        # This ensures weight doesn't suppress TFs too much
        # and unmatched TFs are NOT penalized
        record['specificity_weight'] = 0.7 + 0.6 * combined
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Statistics
    print(f"\n   SPECIFICITY WEIGHT STATISTICS:")
    print(f"   ===============================")
    print(f"   Min: {df['specificity_weight'].min():.3f}")
    print(f"   Max: {df['specificity_weight'].max():.3f}")
    print(f"   Mean: {df['specificity_weight'].mean():.3f}")
    print(f"   Std: {df['specificity_weight'].std():.3f}")
    
    # Statistics by matched/unmatched
    matched_df = df[df['jaspar_match'].notna()]
    unmatched_df = df[df['jaspar_match'].isna()]
    
    print(f"\n   MATCHED TFs weight: mean={matched_df['specificity_weight'].mean():.3f}, std={matched_df['specificity_weight'].std():.3f}")
    print(f"   UNMATCHED TFs weight: mean={unmatched_df['specificity_weight'].mean():.3f}, std={unmatched_df['specificity_weight'].std():.3f}")
    
    # Top 10 most specific TFs
    print(f"\n   TOP 10 MOST SPECIFIC TFs:")
    top10 = df.nlargest(10, 'specificity_weight')
    for _, row in top10.iterrows():
        if pd.notna(row['jaspar_match']):
            n_ver = row.get('n_jaspar_versions', 1)
            jaspar_status = f"JASPAR x{int(n_ver)}" if n_ver > 1 else "JASPAR"
        else:
            jaspar_status = "binding-only"
        print(f"   {row['tf']:15s}  weight={row['specificity_weight']:.3f}  "
              f"IC={row['motif_ic_norm'] or median_ic:.3f}  targets={row['n_targets'] or 'N/A'}  ({jaspar_status})")
    
    # Top 10 least specific (promiscuous) TFs
    print(f"\n   TOP 10 LEAST SPECIFIC (promiscuous) TFs:")
    bottom10 = df.nsmallest(10, 'specificity_weight')
    for _, row in bottom10.iterrows():
        jaspar_status = "JASPAR" if pd.notna(row['jaspar_match']) else "binding-only"
        print(f"   {row['tf']:15s}  weight={row['specificity_weight']:.3f}  "
              f"IC={row['motif_ic_norm'] or median_ic:.3f}  targets={row['n_targets'] or 'N/A'}  ({jaspar_status})")
    
    # 6. Save results
    print("\n[6/6] Saving results...")
    
    # Save full DataFrame
    df.to_csv(output_dir / "tf_specificity_analysis.csv", index=False)
    print(f"   Saved: {output_dir / 'tf_specificity_analysis.csv'}")
    
    # Save just the weights as numpy array (aligned to TF order)
    weights = df.sort_values('idx')['specificity_weight'].values
    np.save(output_dir / "tf_specificity_weights.npy", weights)
    print(f"   Saved: {output_dir / 'tf_specificity_weights.npy'}")
    
    # Save matched TFs list (for reference)
    matched_export = df[df['jaspar_match'].notna()][['tf', 'jaspar_id', 'jaspar_match', 'motif_ic', 'motif_ic_norm']]
    matched_export.to_csv(output_dir / "jaspar_matched_tfs.csv", index=False)
    print(f"   Saved: {output_dir / 'jaspar_matched_tfs.csv'}")
    
    # Save unmatched TFs list (for debugging)
    unmatched_export = df[df['jaspar_match'].isna()][['tf', 'n_targets', 'specificity_weight']]
    unmatched_export.to_csv(output_dir / "jaspar_unmatched_tfs.csv", index=False)
    print(f"   Saved: {output_dir / 'jaspar_unmatched_tfs.csv'}")
    
    # Save summary stats
    with open(output_dir / "specificity_stats.txt", 'w') as f:
        f.write(f"Total TFs: {len(our_tfs)}\n")
        f.write(f"JASPAR entries: {entries_parsed}\n")
        f.write(f"JASPAR matched: {len(matched)}\n")
        f.write(f"Match rate: {100*len(matched)/len(our_tfs):.1f}%\n")
        f.write(f"Weight range: [{df['specificity_weight'].min():.3f}, {df['specificity_weight'].max():.3f}]\n")
        f.write(f"Weight mean: {df['specificity_weight'].mean():.3f}\n")
        f.write(f"Weight std: {df['specificity_weight'].std():.3f}\n")
        f.write(f"\nMatched TFs weight: mean={matched_df['specificity_weight'].mean():.3f}, std={matched_df['specificity_weight'].std():.3f}\n")
        f.write(f"Unmatched TFs weight: mean={unmatched_df['specificity_weight'].mean():.3f}, std={unmatched_df['specificity_weight'].std():.3f}\n")
        f.write(f"\nNOTE: Unmatched TFs are NOT penalized - they use median IC from matched TFs.\n")
        f.write(f"      Weighting is primarily based on binding sparsity (fewer targets = higher weight).\n")
    print(f"   Saved: {output_dir / 'specificity_stats.txt'}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  - {len(matched)} TFs matched to JASPAR ({100*len(matched)/len(our_tfs):.1f}%)")
    print(f"  - {len(unmatched_ours)} TFs use binding-sparsity-only weighting (NOT penalized)")
    print(f"  - Weight range: [{df['specificity_weight'].min():.3f}, {df['specificity_weight'].max():.3f}]")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    main()

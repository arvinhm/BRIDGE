#!/usr/bin/env python3
"""
================================================================================
OMNIPATH SIGNALING NETWORK PREPROCESSING
================================================================================

Processes OmniPath all_signaling_edges.tsv to create:
1. Gene→Gene adjacency matrix (activation/inhibition)
2. Gene→TF adjacency matrix (how genes regulate TFs)
3. Reference-weighted edge strengths

Key Features:
- Converts UniProt IDs to gene symbols
- Handles bidirectional edges (feedback loops)
- Weights edges by number of references (with ceiling)
- Case-insensitive gene matching
- Sparse matrix representation for efficiency

Usage:
    python scripts/preprocess_omnipath.py \
        --input omnipath_all_signaling_edges.tsv \
        --output data/omnipath/ \
        --gene-list data/tf_binding/mappings.pkl

Author: BRIDGE Team
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict


# =============================================================================
# UNIPROT TO GENE SYMBOL MAPPING
# =============================================================================

def load_uniprot_mapping(mapping_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load UniProt to Gene Symbol mapping.
    
    If no mapping file provided, uses a built-in common mapping
    and falls back to extracting from OmniPath data.
    """
    mapping = {}
    
    if mapping_file and Path(mapping_file).exists():
        # Load external mapping (e.g., from UniProt ID mapping service)
        df = pd.read_csv(mapping_file, sep='\t')
        if 'From' in df.columns and 'To' in df.columns:
            mapping = dict(zip(df['From'], df['To']))
        elif 'Entry' in df.columns and 'Gene Names (primary)' in df.columns:
            mapping = dict(zip(df['Entry'], df['Gene Names (primary)']))
    
    return mapping


def fetch_uniprot_to_gene(uniprot_ids: Set[str], batch_size: int = 100) -> Dict[str, str]:
    """
    Fetch UniProt to Gene Symbol mappings using UniProt API.
    
    Uses the new UniProt REST API with proper pagination.
    """
    import urllib.request
    import urllib.parse
    import time
    
    mapping = {}
    uniprot_list = list(uniprot_ids)
    
    print(f"  Fetching gene symbols for {len(uniprot_list)} UniProt IDs...")
    
    # Try UniProt ID mapping API - new format
    try:
        for i in range(0, len(uniprot_list), batch_size):
            batch = uniprot_list[i:i+batch_size]
            
            # Build query string - simpler approach
            query = '+OR+'.join([f'accession:{uid}' for uid in batch])
            url = f'https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,gene_primary&format=tsv&size={batch_size}'
            
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Python/BRIDGE-GRN')
                
                with urllib.request.urlopen(req, timeout=60) as response:
                    lines = response.read().decode('utf-8').strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        for line in lines[1:]:
                            parts = line.split('\t')
                            if len(parts) >= 2 and parts[1]:
                                mapping[parts[0]] = parts[1].upper()
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                # Try alternative API endpoint
                pass
            
            if (i // batch_size) % 20 == 0 and i > 0:
                print(f"    Processed {min(i + batch_size, len(uniprot_list))}/{len(uniprot_list)} ({len(mapping)} mapped)")
                
    except Exception as e:
        print(f"  Warning: UniProt API failed ({e}).")
    
    return mapping


def try_mygene_mapping(uniprot_ids: Set[str]) -> Dict[str, str]:
    """
    Try using mygene library for UniProt to Gene Symbol mapping.
    Much more reliable than direct API calls.
    """
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        
        print("  Using mygene library for ID mapping...")
        uniprot_list = list(uniprot_ids)
        
        # Query in batches
        mapping = {}
        batch_size = 1000
        
        for i in range(0, len(uniprot_list), batch_size):
            batch = uniprot_list[i:i+batch_size]
            results = mg.querymany(batch, scopes='uniprot', fields='symbol', species='human', verbose=False)
            
            for r in results:
                if 'symbol' in r and r['symbol']:
                    mapping[r['query']] = r['symbol'].upper()
            
            if i % 5000 == 0 and i > 0:
                print(f"    Processed {i}/{len(uniprot_list)} ({len(mapping)} mapped)")
        
        print(f"  Mapped {len(mapping)}/{len(uniprot_ids)} IDs via mygene")
        return mapping
        
    except ImportError:
        print("  mygene not installed. Install with: pip install mygene")
        return {}
    except Exception as e:
        print(f"  mygene failed: {e}")
        return {}


def create_fallback_mapping(omnipath_df: pd.DataFrame, gene_names: List[str], tf_names: List[str]) -> Dict[str, str]:
    """
    Create fallback mapping using bioservices or download from UniProt.
    
    As a last resort, download the human proteome mapping file.
    """
    import urllib.request
    import os
    
    mapping = {}
    gene_set = set(g.upper() for g in gene_names)
    tf_set = set(t.upper() for t in tf_names)
    all_genes = gene_set | tf_set
    
    # Try to download human ID mapping from UniProt
    cache_file = 'data/omnipath/uniprot_human_mapping.tsv'
    
    if not os.path.exists(cache_file):
        print("  Downloading UniProt human ID mapping (this may take a minute)...")
        try:
            # Download reviewed human proteins with gene names
            url = 'https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true+AND+organism_id:9606&fields=accession,gene_primary&format=tsv'
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Python/BRIDGE-GRN')
            
            with urllib.request.urlopen(req, timeout=300) as response:
                content = response.read().decode('utf-8')
                
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    f.write(content)
                    
            print(f"  Saved mapping to {cache_file}")
            
        except Exception as e:
            print(f"  Download failed: {e}")
            return {}
    
    # Parse the mapping file
    if os.path.exists(cache_file):
        print(f"  Loading cached mapping from {cache_file}")
        with open(cache_file, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 2 and parts[1]:
                    gene = parts[1].upper()
                    if gene in all_genes:
                        mapping[parts[0]] = gene
        
        print(f"  Found {len(mapping)} UniProt IDs matching our genes")
    
    return mapping


# =============================================================================
# OMNIPATH PROCESSING
# =============================================================================

def load_omnipath(filepath: str) -> pd.DataFrame:
    """Load and parse OmniPath signaling edges."""
    print(f"Loading OmniPath from {filepath}...")
    
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    
    print(f"  Total edges: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # Parse key columns
    required_cols = ['source', 'target', 'is_directed', 'is_stimulation', 'is_inhibition']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Count edge types
    n_stimulation = df['is_stimulation'].sum() if 'is_stimulation' in df.columns else 0
    n_inhibition = df['is_inhibition'].sum() if 'is_inhibition' in df.columns else 0
    n_directed = df['is_directed'].sum() if 'is_directed' in df.columns else 0
    
    print(f"  Directed edges: {n_directed:,}")
    print(f"  Stimulation (activation): {n_stimulation:,}")
    print(f"  Inhibition: {n_inhibition:,}")
    
    # Parse number of references
    if 'n_references' in df.columns:
        df['ref_count'] = df['n_references'].fillna(1).astype(int)
    elif 'references' in df.columns:
        df['ref_count'] = df['references'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) else 1
        )
    else:
        df['ref_count'] = 1
    
    print(f"  Reference counts: min={df['ref_count'].min()}, max={df['ref_count'].max()}, median={df['ref_count'].median():.0f}")
    
    return df


def compute_edge_weights(ref_counts: np.ndarray, 
                         base_weight: float = 0.2,
                         max_weight: float = 0.9,
                         saturation: int = 8) -> np.ndarray:
    """
    Compute edge weights from reference counts.
    
    Formula: weight = base_weight + (max_weight - base_weight) * (1 - exp(-ref_count / saturation))
    
    This ensures:
    - Edges with 0-1 references get base_weight
    - Edges with many references asymptote to max_weight
    - Saturation parameter controls how quickly weights increase
    
    Args:
        ref_counts: Number of references per edge
        base_weight: Minimum weight for any edge
        max_weight: Maximum weight (asymptotic limit)
        saturation: Reference count at which weight reaches ~63% of max
    """
    # Sigmoid-like saturation
    weights = base_weight + (max_weight - base_weight) * (1 - np.exp(-ref_counts / saturation))
    return weights


def build_signaling_matrices(
    omnipath_df: pd.DataFrame,
    gene_names: List[str],
    tf_names: List[str],
    uniprot_to_gene: Dict[str, str],
    base_weight: float = 0.5,
    max_weight: float = 1.0,
    saturation: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Build signaling adjacency matrices from OmniPath data.
    
    Returns:
        gene_gene_adj: (n_genes, n_genes) - Gene→Gene signaling
        gene_tf_adj: (n_genes, n_tfs) - Gene→TF regulation
        tf_gene_adj: (n_tfs, n_genes) - TF→Gene (for comparison with ChIP)
        stats: Dictionary with statistics
    """
    n_genes = len(gene_names)
    n_tfs = len(tf_names)
    
    # Create case-insensitive lookups
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    tf_to_idx = {t.upper(): i for i, t in enumerate(tf_names)}
    tf_set = set(t.upper() for t in tf_names)
    
    # Initialize matrices (use lists for efficiency, convert later)
    gene_gene_edges = []  # (source_idx, target_idx, weight, is_activation)
    gene_tf_edges = []
    tf_gene_edges = []
    tf_tf_edges = []
    
    # Track statistics
    stats = {
        'total_edges': len(omnipath_df),
        'mapped_edges': 0,
        'unmapped_source': 0,
        'unmapped_target': 0,
        'gene_gene': 0,
        'gene_tf': 0,
        'tf_gene': 0,
        'tf_tf': 0,
        'activations': 0,
        'inhibitions': 0,
        'bidirectional': 0
    }
    
    # Track bidirectional edges
    edge_set = set()
    
    print(f"\nBuilding signaling matrices...")
    print(f"  Genes: {n_genes}, TFs: {n_tfs}")
    
    for idx, row in omnipath_df.iterrows():
        source_uniprot = str(row['source']).strip()
        target_uniprot = str(row['target']).strip()
        
        # Map to gene symbols
        source_gene = uniprot_to_gene.get(source_uniprot, '').upper()
        target_gene = uniprot_to_gene.get(target_uniprot, '').upper()
        
        if not source_gene:
            stats['unmapped_source'] += 1
            continue
        if not target_gene:
            stats['unmapped_target'] += 1
            continue
        
        # Determine edge type
        is_activation = row.get('is_stimulation', False) == True
        is_inhibition = row.get('is_inhibition', False) == True
        
        # Skip if neither activation nor inhibition
        if not is_activation and not is_inhibition:
            continue
        
        # Compute weight from references
        ref_count = row.get('ref_count', 1)
        weight = compute_edge_weights(np.array([ref_count]), base_weight, max_weight, saturation)[0]
        
        # Sign: +1 for activation, -1 for inhibition
        sign = 1.0 if is_activation else -1.0
        signed_weight = sign * weight
        
        # Check for bidirectional
        edge_key = (source_gene, target_gene)
        reverse_key = (target_gene, source_gene)
        if reverse_key in edge_set:
            stats['bidirectional'] += 1
        edge_set.add(edge_key)
        
        # Classify edge type and add to appropriate list
        source_is_tf = source_gene in tf_set
        target_is_tf = target_gene in tf_set
        source_is_gene = source_gene in gene_to_idx
        target_is_gene = target_gene in gene_to_idx
        
        if source_is_gene and target_is_gene:
            source_idx = gene_to_idx[source_gene]
            target_idx = gene_to_idx[target_gene]
            
            if source_is_tf and target_is_tf:
                # TF → TF
                tf_tf_edges.append((tf_to_idx[source_gene], tf_to_idx[target_gene], signed_weight))
                stats['tf_tf'] += 1
            elif source_is_tf:
                # TF → Gene
                tf_gene_edges.append((tf_to_idx[source_gene], target_idx, signed_weight))
                stats['tf_gene'] += 1
            elif target_is_tf:
                # Gene → TF (NEW!)
                gene_tf_edges.append((source_idx, tf_to_idx[target_gene], signed_weight))
                stats['gene_tf'] += 1
            else:
                # Gene → Gene
                gene_gene_edges.append((source_idx, target_idx, signed_weight))
                stats['gene_gene'] += 1
            
            stats['mapped_edges'] += 1
            if is_activation:
                stats['activations'] += 1
            else:
                stats['inhibitions'] += 1
        
        if idx % 50000 == 0 and idx > 0:
            print(f"    Processed {idx:,}/{len(omnipath_df):,} edges...")
    
    # Convert to matrices
    print(f"\nConverting to matrices...")
    
    # Gene → Gene
    gene_gene_adj = np.zeros((n_genes, n_genes), dtype=np.float32)
    for src, tgt, w in gene_gene_edges:
        gene_gene_adj[src, tgt] = w  # Note: overwrites if duplicate, could also add
    
    # Gene → TF
    gene_tf_adj = np.zeros((n_genes, n_tfs), dtype=np.float32)
    for src, tgt, w in gene_tf_edges:
        gene_tf_adj[src, tgt] = w
    
    # TF → Gene (from OmniPath, separate from ChIP-Atlas)
    tf_gene_adj_omnipath = np.zeros((n_tfs, n_genes), dtype=np.float32)
    for src, tgt, w in tf_gene_edges:
        tf_gene_adj_omnipath[src, tgt] = w
    
    # TF → TF (from OmniPath, to blend with existing)
    tf_tf_adj_omnipath = np.zeros((n_tfs, n_tfs), dtype=np.float32)
    for src, tgt, w in tf_tf_edges:
        tf_tf_adj_omnipath[src, tgt] = w
    
    print(f"\nMatrix Statistics:")
    print(f"  Gene->Gene: {np.count_nonzero(gene_gene_adj):,} edges")
    print(f"    Activations: {np.sum(gene_gene_adj > 0):,}")
    print(f"    Inhibitions: {np.sum(gene_gene_adj < 0):,}")
    print(f"  Gene->TF: {np.count_nonzero(gene_tf_adj):,} edges")
    print(f"  TF->Gene (OmniPath): {np.count_nonzero(tf_gene_adj_omnipath):,} edges")
    print(f"  TF->TF (OmniPath): {np.count_nonzero(tf_tf_adj_omnipath):,} edges")
    
    return gene_gene_adj, gene_tf_adj, tf_gene_adj_omnipath, tf_tf_adj_omnipath, stats


def apply_topk_pruning(adj_matrix: np.ndarray, topk: int) -> np.ndarray:
    """
    Keep only top-k outgoing edges per source node (row) by absolute weight.
    
    This reduces memory usage and improves training stability by focusing
    on the most important connections.
    
    Uses argpartition for O(n) complexity instead of argsort O(n log n).
    
    Args:
        adj_matrix: (n_source, n_target) adjacency matrix
        topk: Number of edges to keep per source node
        
    Returns:
        Pruned adjacency matrix with same shape
    """
    if topk is None or topk <= 0:
        return adj_matrix
    
    n_source, n_target = adj_matrix.shape
    
    # Don't prune if matrix is smaller than topk
    if n_target <= topk:
        return adj_matrix
    
    pruned = np.zeros_like(adj_matrix)
    
    for i in range(n_source):
        row = adj_matrix[i]
        n_nonzero = np.count_nonzero(row)
        if n_nonzero == 0:
            continue
        
        if n_nonzero <= topk:
            pruned[i] = row
        else:
            # Use argpartition for O(n) complexity
            abs_row = np.abs(row)
            topk_indices = np.argpartition(abs_row, -topk)[-topk:]
            pruned[i, topk_indices] = row[topk_indices]
    
    return pruned


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Preprocess OmniPath signaling network')
    parser.add_argument('--input', type=str, default='omnipath_all_signaling_edges.tsv',
                        help='Path to OmniPath TSV file')
    parser.add_argument('--output', type=str, default='data/omnipath/',
                        help='Output directory')
    parser.add_argument('--gene-list', type=str, default='data/tf_binding/mappings.pkl',
                        help='Path to gene/TF mappings pickle')
    parser.add_argument('--uniprot-mapping', type=str, default=None,
                        help='Optional: Path to UniProt→Gene mapping file')
    parser.add_argument('--base-weight', type=float, default=0.5,
                        help='Base weight for edges')
    parser.add_argument('--max-weight', type=float, default=1.0,
                        help='Maximum weight for edges')
    parser.add_argument('--saturation', type=int, default=10,
                        help='Reference count for weight saturation')
    parser.add_argument('--topk-per-gene', type=int, default=None,
                        help='Keep only top-k outgoing edges per gene (memory efficiency)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OMNIPATH SIGNALING NETWORK PREPROCESSING")
    print("=" * 70)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load gene/TF names
    print(f"\n[1/4] Loading gene and TF lists...")
    mappings_path = Path(args.gene_list)
    
    if mappings_path.suffix == '.pkl':
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        gene_names = mappings.get('gene_names', [])
        tf_names = mappings.get('tf_names', [])
    else:
        # Try loading from results directory
        base_dir = mappings_path.parent.parent
        gene_names = pd.read_csv(base_dir / 'gene_names.csv', header=None)[0].tolist()
        tf_names = pd.read_csv(base_dir / 'tf_names.csv', header=None)[0].tolist()
    
    print(f"  Genes: {len(gene_names)}")
    print(f"  TFs: {len(tf_names)}")
    
    # Load OmniPath
    print(f"\n[2/4] Loading OmniPath data...")
    omnipath_df = load_omnipath(args.input)
    
    # Get UniProt IDs
    all_uniprots = set(omnipath_df['source'].unique()) | set(omnipath_df['target'].unique())
    print(f"  Unique UniProt IDs: {len(all_uniprots):,}")
    
    # Load or create UniProt->Gene mapping
    print(f"\n[3/4] Mapping UniProt IDs to gene symbols...")
    
    uniprot_to_gene = {}
    
    if args.uniprot_mapping and Path(args.uniprot_mapping).exists():
        uniprot_to_gene = load_uniprot_mapping(args.uniprot_mapping)
        print(f"  Loaded {len(uniprot_to_gene):,} mappings from file")
    
    # If not enough mappings, try mygene library
    if len(uniprot_to_gene) < len(all_uniprots) * 0.3:
        mygene_mapping = try_mygene_mapping(all_uniprots)
        uniprot_to_gene.update(mygene_mapping)
    
    # If still not enough, try downloading from UniProt
    if len(uniprot_to_gene) < len(all_uniprots) * 0.3:
        print("  Trying fallback mapping...")
        fallback_mapping = create_fallback_mapping(omnipath_df, gene_names, tf_names)
        uniprot_to_gene.update(fallback_mapping)
    
    # Final attempt: direct UniProt API
    if len(uniprot_to_gene) < len(all_uniprots) * 0.3:
        print("  Trying UniProt REST API...")
        api_mapping = fetch_uniprot_to_gene(all_uniprots - set(uniprot_to_gene.keys()))
        uniprot_to_gene.update(api_mapping)
    
    if len(uniprot_to_gene) < len(all_uniprots) * 0.1:
        print(f"  WARNING: Only mapped {len(uniprot_to_gene)}/{len(all_uniprots)} IDs")
        print(f"  Install mygene for better mapping: pip install mygene")
    
    # Save mapping for reference
    mapping_df = pd.DataFrame([
        {'uniprot_id': k, 'gene_symbol': v}
        for k, v in uniprot_to_gene.items()
    ])
    mapping_df.to_csv(output_dir / 'uniprot_to_gene_mapping.csv', index=False)
    print(f"  Saved mapping to {output_dir / 'uniprot_to_gene_mapping.csv'}")
    
    # Build matrices
    print(f"\n[4/4] Building signaling matrices...")
    gene_gene_adj, gene_tf_adj, tf_gene_adj, tf_tf_adj, stats = build_signaling_matrices(
        omnipath_df,
        gene_names,
        tf_names,
        uniprot_to_gene,
        base_weight=args.base_weight,
        max_weight=args.max_weight,
        saturation=args.saturation
    )
    
    # Apply top-k pruning if specified (for memory efficiency)
    if args.topk_per_gene:
        print(f"\n  Applying top-k pruning (k={args.topk_per_gene})...")
        orig_gg = np.count_nonzero(gene_gene_adj)
        orig_gt = np.count_nonzero(gene_tf_adj)
        
        gene_gene_adj = apply_topk_pruning(gene_gene_adj, args.topk_per_gene)
        gene_tf_adj = apply_topk_pruning(gene_tf_adj, args.topk_per_gene)
        
        print(f"    Gene->Gene: {orig_gg:,} -> {np.count_nonzero(gene_gene_adj):,} edges")
        print(f"    Gene->TF: {orig_gt:,} -> {np.count_nonzero(gene_tf_adj):,} edges")
    
    # Save matrices
    print(f"\nSaving outputs to {output_dir}/...")
    
    np.save(output_dir / 'gene_gene_signaling.npy', gene_gene_adj)
    np.save(output_dir / 'gene_tf_signaling.npy', gene_tf_adj)
    np.save(output_dir / 'tf_gene_signaling_omnipath.npy', tf_gene_adj)
    np.save(output_dir / 'tf_tf_signaling_omnipath.npy', tf_tf_adj)
    
    # Save sparse versions for efficiency
    sparse.save_npz(output_dir / 'gene_gene_signaling_sparse.npz', 
                    sparse.csr_matrix(gene_gene_adj))
    sparse.save_npz(output_dir / 'gene_tf_signaling_sparse.npz',
                    sparse.csr_matrix(gene_tf_adj))
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / 'omnipath_stats.csv', index=False)
    
    with open(output_dir / 'omnipath_stats.txt', 'w') as f:
        f.write("OMNIPATH PREPROCESSING STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:,}\n")
    
    print(f"\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - gene_gene_signaling.npy: ({gene_gene_adj.shape}) Gene->Gene network")
    print(f"  - gene_tf_signaling.npy: ({gene_tf_adj.shape}) Gene->TF regulation")
    print(f"  - tf_gene_signaling_omnipath.npy: TF->Gene from OmniPath")
    print(f"  - tf_tf_signaling_omnipath.npy: TF->TF from OmniPath")
    print(f"  - uniprot_to_gene_mapping.csv: ID mapping reference")
    print(f"  - omnipath_stats.csv: Processing statistics")
    
    print(f"\nKey Statistics:")
    print(f"  Mapped edges: {stats['mapped_edges']:,} / {stats['total_edges']:,} ({100*stats['mapped_edges']/stats['total_edges']:.1f}%)")
    print(f"  Gene->Gene: {stats['gene_gene']:,} edges")
    print(f"  Gene->TF: {stats['gene_tf']:,} edges (NEW! Upstream regulation)")
    print(f"  Bidirectional: {stats['bidirectional']:,} feedback loops")


if __name__ == '__main__':
    main()


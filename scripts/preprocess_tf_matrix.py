#!/usr/bin/env python3
"""
Preprocess TF-Gene Binding Matrix from ChIP-Atlas

Processes transcription_factor_gene_matrix.csv and prepares:
1. Binary/continuous TF-Gene binding matrix
2. TF-TF regulatory adjacency matrix
3. Gene and TF name mappings
4. Pathway annotations

Usage:
    python scripts/preprocess_tf_matrix.py --input TF_pathways/transcription_factor_gene_matrix.csv --output data/tf_binding/
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tf_gene_matrix(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load TF-Gene binding matrix from CSV.
    
    The CSV format is:
    - Row 1: TF names as column headers
    - Column 1: Gene names as row index
    - Values: Binding scores or gene names (need parsing)
    
    Args:
        csv_path: Path to transcription_factor_gene_matrix.csv
        
    Returns:
        Tuple of (binding_df, gene_names, tf_names)
    """
    print(f"Loading TF-Gene matrix from {csv_path}...")
    
    # First, read the header to understand structure
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
    
    # TF names are in the header
    tf_names = header[1:]  # Skip first column (gene column)
    print(f"  Found {len(tf_names)} TFs")
    
    # Load the full matrix
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    
    # Check if values are gene names (string) or scores (numeric)
    sample_val = df.iloc[0, 0]
    is_string_values = isinstance(sample_val, str)
    
    if is_string_values:
        print("  Matrix contains gene names - converting to binary binding")
        # Convert string gene names to binary (1 if TF binds to gene at that position)
        # Gene names are extracted from the VALUES, not row indices
        binding_matrix, gene_names = process_string_matrix(df, tf_names)
    else:
        print("  Matrix contains numeric scores")
        binding_matrix = df.astype(float)
        # Ensure gene names are strings
        gene_names = [str(g) for g in df.index]
    
    print(f"  Found {len(gene_names)} genes")
    
    return binding_matrix, gene_names, tf_names


def process_string_matrix(df: pd.DataFrame, tf_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process matrix where values are target gene names.
    
    For each TF column, values indicate which gene is targeted.
    We convert this to a binary matrix indicating TF->Gene binding.
    
    IMPORTANT: Gene names are the VALUES in the matrix, not the row index!
    
    Args:
        df: DataFrame with gene names as values
        tf_names: List of TF names
        
    Returns:
        Tuple of (Binary binding matrix (genes x TFs), list of gene names)
    """
    # First, collect ALL unique gene names from the VALUES
    all_genes = set()
    for tf in tf_names:
        if tf in df.columns:
            # Get all non-null values (these are target gene names)
            targets = df[tf].dropna()
            for target in targets:
                gene_str = str(target).strip()
                if gene_str and gene_str.lower() != 'nan':
                    all_genes.add(gene_str)
    
    # Convert to sorted list for consistent ordering
    gene_names = sorted(list(all_genes))
    print(f"  Extracted {len(gene_names)} unique gene names from matrix values")
    print(f"  Gene names sample: {gene_names[:5]}")
    
    n_genes = len(gene_names)
    n_tfs = len(tf_names)
    
    # Gene name to index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Create empty binary matrix
    binding_matrix = np.zeros((n_genes, n_tfs), dtype=np.float32)
    
    # For each TF column
    for tf_idx, tf in enumerate(tqdm(tf_names, desc="Processing TFs")):
        if tf not in df.columns:
            continue
            
        # Get target genes for this TF
        targets = df[tf].dropna()
        
        for target in targets:
            target_str = str(target).strip()
            if target_str in gene_to_idx:
                gene_idx = gene_to_idx[target_str]
                binding_matrix[gene_idx, tf_idx] = 1.0
                
    result = pd.DataFrame(
        binding_matrix,
        index=gene_names,
        columns=tf_names
    )
    
    return result, gene_names


def normalize_binding_matrix(
    binding_matrix: pd.DataFrame,
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize binding matrix values.
    
    Args:
        binding_matrix: Raw binding matrix
        method: 'minmax', 'zscore', or 'none'
        
    Returns:
        Normalized matrix
    """
    if method == 'none':
        return binding_matrix
        
    values = binding_matrix.values
    
    if method == 'minmax':
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = values.mean()
        std = values.std()
        if std > 0:
            values = (values - mean) / std
            
    return pd.DataFrame(
        values,
        index=binding_matrix.index,
        columns=binding_matrix.columns
    )


# Known DNA-binding TF families (for filtering cofactors)
# These are prefixes/patterns for true transcription factors
DNA_BINDING_TF_FAMILIES = {
    'ZNF', 'ZFP', 'KLF', 'SP', 'EGR', 'WT1',  # Zinc fingers
    'FOX', 'FOXA', 'FOXB', 'FOXC', 'FOXD', 'FOXE', 'FOXF', 'FOXG', 'FOXH', 'FOXI', 'FOXJ', 'FOXK', 'FOXL', 'FOXM', 'FOXN', 'FOXO', 'FOXP', 'FOXQ', 'FOXR', 'FOXS',  # Forkhead
    'SOX', 'SRY',  # HMG box
    'PAX',  # Paired box
    'HOX', 'HOXA', 'HOXB', 'HOXC', 'HOXD', 'CDX', 'EVX', 'GBX', 'GSX', 'PDX', 'MNX',  # Homeobox
    'POU', 'OCT', 'PIT', 'BRN',  # POU domain
    'GATA',  # GATA
    'NF', 'NFAT', 'NFKB', 'NFE', 'NFIA', 'NFIB', 'NFIC', 'NFIX',  # NF family
    'STAT',  # STAT
    'SMAD',  # SMAD
    'CREB', 'ATF', 'JUN', 'FOS', 'MAF', 'BACH',  # bZIP
    'MYC', 'MAX', 'MAD', 'MXI', 'MLX', 'MLXIP',  # bHLH-LZ
    'E2F', 'DP',  # E2F
    'ETS', 'ELK', 'ELF', 'ERF', 'ERG', 'FLI', 'GAB', 'PEA', 'SPI', 'TEL',  # ETS
    'RUNX', 'CBFA', 'CBFB',  # RUNX
    'TEAD', 'TEF',  # TEAD
    'YY1', 'YY2',  # YY
    'MYB', 'MYBL',  # MYB
    'E2A', 'TCF', 'LEF', 'HEB', 'TCF3', 'TCF4', 'TCF7', 'TCF12',  # bHLH / TCF-LEF
    'IRF',  # IRF
    'REL', 'RELA', 'RELB', 'NFKB1', 'NFKB2',  # REL/NF-kB
    'NR', 'ESR', 'AR', 'PGR', 'GR', 'MR', 'VDR', 'RAR', 'RXR', 'PPARA', 'PPARD', 'PPARG', 'LXR', 'FXR', 'PXR', 'CAR', 'HNF4', 'COUP', 'ERR', 'NUR', 'TR', 'THRA', 'THRB',  # Nuclear receptors
    'TP53', 'TP63', 'TP73', 'P53', 'P63', 'P73',  # p53 family
    'NOTCH', 'RBPJ', 'CSL',  # Notch pathway TFs
    'ASCL', 'NEUROD', 'NEUROG', 'ATOH', 'OLIG', 'TWIST', 'HAND', 'ID', 'HES', 'HEY',  # bHLH
    'TBX', 'EOMES', 'TBXT', 'TBR',  # T-box
    'SIX', 'DACH', 'EYA',  # SIX-EYA-DACH
    'LHX', 'ISL', 'LMO', 'LDB',  # LIM
    'REST', 'NRSF',  # REST
    'CTCF',  # CTCF
    'BACH', 'NFE2', 'NRF',  # CNC-bZIP
}

# Known cofactors/enzymes to EXCLUDE (not DNA-binding TFs)
COFACTORS_TO_EXCLUDE = {
    'POLR2A', 'POLR2B', 'POLR2C', 'POLR2D', 'POLR2E', 'POLR2F', 'POLR2G', 'POLR2H', 'POLR2I', 'POLR2J', 'POLR2K', 'POLR2L',  # RNA Pol II
    'POLR1', 'POLR3',  # Other polymerases
    'TBP', 'TAF', 'TFIIA', 'TFIIB', 'TFIID', 'TFIIE', 'TFIIF', 'TFIIH',  # General TFs
    'KAT', 'KAT2A', 'KAT2B', 'KAT5', 'KAT6A', 'KAT6B', 'KAT7', 'KAT8',  # Histone acetyltransferases
    'HDAC', 'SIRT',  # Histone deacetylases
    'KMT', 'SETD', 'DOT1L', 'EZH', 'SUV39', 'MLL', 'KMT2',  # Histone methyltransferases
    'KDM', 'JMJD', 'LSD',  # Histone demethylases
    'BRD', 'BPTF', 'SMARCA', 'SMARCB', 'SMARCC', 'SMARCD', 'SMARCE',  # Chromatin remodelers
    'RAG1', 'RAG2',  # Recombinases
    'CRTC', 'CRTC1', 'CRTC2', 'CRTC3',  # CREB coactivators
    'NCOA', 'NCOR', 'SRC',  # Nuclear receptor coactivators
    'CBP', 'EP300', 'P300', 'CREBBP',  # HATs/coactivators
    'SAP', 'SAP130', 'SAP30',  # Sin3 complex
    'SIN3A', 'SIN3B',  # Sin3 complex
    'BCORL', 'BCOR',  # BCL6 corepressor
    'BAHCC', 'BAH',  # BAH domain proteins
    'ARID', 'ARID1A', 'ARID1B', 'ARID2', 'ARID3A', 'ARID3B', 'ARID4A', 'ARID4B', 'ARID5A', 'ARID5B',  # ARID (some are TFs, some aren't)
    'CDK', 'CDKN',  # Kinases/inhibitors
    'RB1', 'RBL1', 'RBL2',  # Pocket proteins (interact with TFs but aren't TFs)
    'MED', 'MED1', 'MED12', 'MED13', 'MED14', 'MED15', 'MED16',  # Mediator complex
    'GTF', 'GTF2A', 'GTF2B', 'GTF2E', 'GTF2F', 'GTF2H',  # General TFs
}


def is_dna_binding_tf(tf_name: str) -> bool:
    """Check if a TF is likely a DNA-binding transcription factor."""
    tf_upper = tf_name.upper()
    
    # Check exclusion list first
    for cofactor in COFACTORS_TO_EXCLUDE:
        if tf_upper.startswith(cofactor):
            return False
    
    # Check if matches any known DNA-binding family
    for family in DNA_BINDING_TF_FAMILIES:
        if tf_upper.startswith(family):
            return True
    
    # Default: include (may have false positives)
    return True


def build_tf_tf_adjacency(
    binding_matrix: pd.DataFrame,
    tf_names: List[str],
    gene_names: List[str],
    filter_dna_binding: bool = False
) -> np.ndarray:
    """
    Build TF-TF regulatory adjacency matrix.
    
    If TF1 binds to the gene of TF2, then TF1 regulates TF2.
    
    Args:
        binding_matrix: TF-Gene binding matrix
        tf_names: List of TF names
        gene_names: List of gene names
        filter_dna_binding: If True, filter to DNA-binding TFs only
        
    Returns:
        TF-TF adjacency matrix (n_tfs x n_tfs)
        
    NOTE: Your TF list may include cofactors (KAT6B, CRTC2, RAG2, POLR2C, etc.)
    which have broad chromatin association but aren't DNA-binding TFs.
    Set filter_dna_binding=True to exclude these, or use a curated TF list
    from AnimalTFDB/TFClass/hTFtarget.
    """
    print("Building TF-TF adjacency matrix...")
    
    if filter_dna_binding:
        dna_binding_mask = [is_dna_binding_tf(tf) for tf in tf_names]
        n_dna_binding = sum(dna_binding_mask)
        print(f"  Filtering to DNA-binding TFs: {n_dna_binding}/{len(tf_names)}")
    
    n_tfs = len(tf_names)
    adjacency = np.zeros((n_tfs, n_tfs), dtype=np.float32)
    
    # TF to gene index mapping - convert to strings first
    tf_to_gene = {}
    gene_name_lower = {str(g).lower(): i for i, g in enumerate(gene_names)}
    
    for tf_idx, tf in enumerate(tf_names):
        tf_lower = str(tf).lower()
        if tf_lower in gene_name_lower:
            tf_to_gene[tf_idx] = gene_name_lower[tf_lower]
            
    print(f"  {len(tf_to_gene)} TFs found as genes")
    
    # Build adjacency
    for tf1_idx in range(n_tfs):
        for tf2_idx, gene_idx in tf_to_gene.items():
            if tf1_idx != tf2_idx:
                # TF1 binds to gene of TF2
                binding_score = binding_matrix.iloc[gene_idx, tf1_idx]
                if binding_score > 0:
                    adjacency[tf1_idx, tf2_idx] = binding_score
                    
    # Normalize rows
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency = adjacency / row_sums
    
    return adjacency


def load_pathway_annotations(
    pathway_dir: str,
    tf_names: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Load pathway annotations from TF_Pathways directory.
    
    Args:
        pathway_dir: Path to TF_Pathways directory
        tf_names: List of TF names
        
    Returns:
        Dict mapping TF -> pathway type -> pathways
    """
    print(f"Loading pathway annotations from {pathway_dir}...")
    
    pathway_dict = {}
    pathway_dir = Path(pathway_dir)
    
    for tf in tqdm(tf_names, desc="Loading pathways"):
        pathway_file = pathway_dir / f"{tf}_pathway_analysis.csv"
        
        if not pathway_file.exists():
            continue
            
        try:
            df = pd.read_csv(pathway_file)
            
            tf_pathways = {}
            for domain in df['domain'].unique():
                domain_df = df[df['domain'] == domain]
                tf_pathways[domain] = domain_df['term.name'].tolist()
                
            pathway_dict[tf] = tf_pathways
            
        except Exception as e:
            print(f"  Error loading {tf}: {e}")
            
    print(f"  Loaded pathways for {len(pathway_dict)} TFs")
    
    return pathway_dict


def save_processed_data(
    output_dir: str,
    binding_matrix: pd.DataFrame,
    tf_tf_adjacency: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    pathway_annotations: Optional[Dict] = None
):
    """
    Save all processed data.
    
    Args:
        output_dir: Output directory
        binding_matrix: TF-Gene binding matrix
        tf_tf_adjacency: TF-TF adjacency
        gene_names: Gene names
        tf_names: TF names
        pathway_annotations: Optional pathway data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving processed data to {output_dir}...")
    
    # Save binding matrix
    binding_matrix.to_csv(output_dir / 'binding_matrix.csv')
    print(f"  Saved binding_matrix.csv ({binding_matrix.shape})")
    
    # Save as numpy for faster loading
    np.save(output_dir / 'binding_matrix.npy', binding_matrix.values)
    print(f"  Saved binding_matrix.npy")
    
    # Save TF-TF adjacency
    np.save(output_dir / 'tf_tf_adjacency.npy', tf_tf_adjacency)
    print(f"  Saved tf_tf_adjacency.npy ({tf_tf_adjacency.shape})")
    
    # Save gene and TF mappings
    mappings = {
        'gene_names': gene_names,
        'tf_names': tf_names,
        'gene_to_idx': {g: i for i, g in enumerate(gene_names)},
        'tf_to_idx': {t: i for i, t in enumerate(tf_names)},
        'idx_to_gene': {i: g for i, g in enumerate(gene_names)},
        'idx_to_tf': {i: t for i, t in enumerate(tf_names)}
    }
    
    with open(output_dir / 'mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    print(f"  Saved mappings.pkl")
    
    # Save pathway annotations
    if pathway_annotations:
        with open(output_dir / 'pathway_annotations.pkl', 'wb') as f:
            pickle.dump(pathway_annotations, f)
        print(f"  Saved pathway_annotations.pkl")
        
    # Save summary stats
    stats = {
        'n_genes': len(gene_names),
        'n_tfs': len(tf_names),
        'binding_sparsity': (binding_matrix.values == 0).mean(),
        'tf_tf_edges': (tf_tf_adjacency > 0).sum()
    }
    
    with open(output_dir / 'stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
            
    print(f"  Stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess TF-Gene binding data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to transcription_factor_gene_matrix.csv')
    parser.add_argument('--pathway-dir', type=str, default=None,
                        help='Path to TF_Pathways directory')
    parser.add_argument('--output', type=str, default='data/tf_binding',
                        help='Output directory')
    parser.add_argument('--normalize', type=str, default='minmax',
                        choices=['minmax', 'zscore', 'none'],
                        help='Normalization method')
    
    args = parser.parse_args()
    
    # Load TF-Gene matrix
    binding_df, gene_names, tf_names = load_tf_gene_matrix(args.input)
    
    # Normalize
    binding_df = normalize_binding_matrix(binding_df, args.normalize)
    
    # Build TF-TF adjacency
    tf_tf_adj = build_tf_tf_adjacency(binding_df, tf_names, gene_names)
    
    # Load pathway annotations
    pathway_annotations = None
    if args.pathway_dir and Path(args.pathway_dir).exists():
        pathway_annotations = load_pathway_annotations(args.pathway_dir, tf_names)
        
    # Save
    save_processed_data(
        args.output,
        binding_df,
        tf_tf_adj,
        gene_names,
        tf_names,
        pathway_annotations
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()


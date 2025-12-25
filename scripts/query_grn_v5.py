#!/usr/bin/env python3
"""
================================================================================
Query GRN Model v5 - Explore Gene Regulation
================================================================================

Query the trained GRN model to find:
1. Which TFs regulate a target gene (activators and repressors)
2. What happens when a TF is perturbed (all affected genes)
3. Regulatory pathways and cascades

Usage:
    # Find regulators of CD3D
    python scripts/query_grn_v5.py --model results/grn_model_v5 --gene CD3D
    
    # Find targets of a TF
    python scripts/query_grn_v5.py --model results/grn_model_v5 --tf STAT1
    
    # Interactive mode
    python scripts/query_grn_v5.py --model results/grn_model_v5 --interactive
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# For visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_grn_model(model_dir: str) -> Dict:
    """Load trained GRN model and metadata."""
    model_dir = Path(model_dir)
    
    # Load checkpoint
    checkpoint_path = model_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / 'final_model.pt'
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load or compute GRN matrix
    grn_path = model_dir / 'grn_matrix.npy'
    if grn_path.exists():
        print(f"Loading GRN matrix from {grn_path}...")
        grn_matrix = np.load(grn_path)
    else:
        print("Computing GRN matrix from model weights...")
        # Extract from model state dict
        state = checkpoint['model_state_dict']
        direct_effects = state['direct_effects'].numpy()
        grn_matrix = direct_effects  # Simplified - full computation would need model
        
    # Load gene names
    gene_names_path = model_dir / 'gene_names.csv'
    if gene_names_path.exists():
        gene_df = pd.read_csv(gene_names_path)
        gene_names = list(gene_df['gene'].values)
    else:
        gene_names = [f"Gene_{i}" for i in range(grn_matrix.shape[0])]
        
    # Load TF names
    tf_names_path = model_dir / 'tf_names.csv'
    if tf_names_path.exists():
        tf_df = pd.read_csv(tf_names_path)
        tf_names = list(tf_df['tf'].values)
    else:
        tf_names = [f"TF_{i}" for i in range(grn_matrix.shape[1])]
        
    # Load cell type names
    celltype_path = model_dir / 'celltype_names.csv'
    if celltype_path.exists():
        ct_df = pd.read_csv(celltype_path)
        celltype_names = list(ct_df['celltype'].values)
    else:
        celltype_names = []
        
    # Load config
    config_path = model_dir / 'model_config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = checkpoint.get('config', {})
        
    # Load metrics
    metrics = checkpoint.get('metrics', {})
    
    return {
        'grn_matrix': grn_matrix,
        'gene_names': gene_names,
        'tf_names': tf_names,
        'celltype_names': celltype_names,
        'config': config,
        'metrics': metrics
    }


def find_gene_index(gene_name: str, gene_names: List[str]) -> Optional[int]:
    """Find gene index (case-insensitive)."""
    gene_upper = gene_name.upper()
    for i, g in enumerate(gene_names):
        if str(g).upper() == gene_upper:
            return i
    return None


def find_tf_index(tf_name: str, tf_names: List[str]) -> Optional[int]:
    """Find TF index (case-insensitive)."""
    tf_upper = tf_name.upper()
    for i, t in enumerate(tf_names):
        if str(t).upper() == tf_upper:
            return i
    return None


def query_gene_regulators(
    gene_name: str,
    grn_matrix: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    top_k: int = 20
) -> Dict:
    """
    Find TFs that regulate a target gene.
    
    Returns activators (positive effect) and repressors (negative effect).
    """
    gene_idx = find_gene_index(gene_name, gene_names)
    
    if gene_idx is None:
        # Try partial match
        matches = [g for g in gene_names if gene_name.upper() in str(g).upper()]
        if matches:
            print(f"Gene '{gene_name}' not found exactly. Did you mean: {matches[:5]}?")
        return {'error': f"Gene '{gene_name}' not found"}
        
    # Get TF effects on this gene
    effects = grn_matrix[gene_idx, :]  # (n_tfs,)
    
    # Sort by absolute effect
    sorted_indices = np.argsort(np.abs(effects))[::-1]
    
    # Separate activators and repressors
    activators = []
    repressors = []
    
    for idx in sorted_indices:
        effect = effects[idx]
        tf_name = tf_names[idx]
        
        if effect > 0.001:  # Threshold for significance
            activators.append((tf_name, float(effect)))
        elif effect < -0.001:
            repressors.append((tf_name, float(effect)))
            
    return {
        'gene': gene_name,
        'gene_idx': gene_idx,
        'activators': activators[:top_k],
        'repressors': repressors[:top_k],
        'total_activators': len(activators),
        'total_repressors': len(repressors)
    }


def query_tf_targets(
    tf_name: str,
    grn_matrix: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    top_k: int = 20
) -> Dict:
    """
    Find genes regulated by a TF.
    
    Returns activated genes and repressed genes.
    """
    tf_idx = find_tf_index(tf_name, tf_names)
    
    if tf_idx is None:
        matches = [t for t in tf_names if tf_name.upper() in str(t).upper()]
        if matches:
            print(f"TF '{tf_name}' not found exactly. Did you mean: {matches[:5]}?")
        return {'error': f"TF '{tf_name}' not found"}
        
    # Get this TF's effect on all genes
    effects = grn_matrix[:, tf_idx]  # (n_genes,)
    
    # Sort by absolute effect
    sorted_indices = np.argsort(np.abs(effects))[::-1]
    
    # Separate activated and repressed genes
    activated = []
    repressed = []
    
    for idx in sorted_indices:
        effect = effects[idx]
        gene = gene_names[idx]
        
        if effect > 0.001:
            activated.append((gene, float(effect)))
        elif effect < -0.001:
            repressed.append((gene, float(effect)))
            
    return {
        'tf': tf_name,
        'tf_idx': tf_idx,
        'activated_genes': activated[:top_k],
        'repressed_genes': repressed[:top_k],
        'total_activated': len(activated),
        'total_repressed': len(repressed)
    }


def query_upregulation_strategy(
    gene_name: str,
    grn_matrix: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    top_k: int = 10
) -> Dict:
    """
    Find strategies to UPREGULATE a target gene.
    
    Returns:
    - TFs to overexpress (activators)
    - TFs to knockout (repressors)
    """
    result = query_gene_regulators(gene_name, grn_matrix, gene_names, tf_names, top_k=50)
    
    if 'error' in result:
        return result
        
    strategies = {
        'gene': gene_name,
        'to_upregulate': {
            'overexpress_these_TFs': result['activators'][:top_k],
            'knockout_these_TFs': result['repressors'][:top_k]
        },
        'explanation': f"To increase {gene_name} expression:\n"
                      f"  - Overexpress top activators (positive regulators)\n"
                      f"  - Knockout top repressors (negative regulators)"
    }
    
    return strategies


def query_downregulation_strategy(
    gene_name: str,
    grn_matrix: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    top_k: int = 10
) -> Dict:
    """
    Find strategies to DOWNREGULATE a target gene.
    
    Returns:
    - TFs to knockout (activators)
    - TFs to overexpress (repressors)
    """
    result = query_gene_regulators(gene_name, grn_matrix, gene_names, tf_names, top_k=50)
    
    if 'error' in result:
        return result
        
    strategies = {
        'gene': gene_name,
        'to_downregulate': {
            'knockout_these_TFs': result['activators'][:top_k],
            'overexpress_these_TFs': result['repressors'][:top_k]
        },
        'explanation': f"To decrease {gene_name} expression:\n"
                      f"  - Knockout top activators (positive regulators)\n"
                      f"  - Overexpress top repressors (negative regulators)"
    }
    
    return strategies


def print_gene_regulation(result: Dict):
    """Pretty print gene regulation results."""
    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return
        
    print(f"\n{'='*70}")
    print(f"  REGULATION OF: {result['gene']}")
    print(f"{'='*70}")
    
    print(f"\n  ACTIVATORS (TFs that INCREASE {result['gene']}):")
    print(f"  " + "-"*50)
    if result['activators']:
        for i, (tf, effect) in enumerate(result['activators'], 1):
            bar = "+" * min(int(effect * 100), 20)
            print(f"  {i:2d}. {tf:15s}  effect: +{effect:.4f}  {bar}")
    else:
        print("    (none found)")
        
    print(f"\n  REPRESSORS (TFs that DECREASE {result['gene']}):")
    print(f"  " + "-"*50)
    if result['repressors']:
        for i, (tf, effect) in enumerate(result['repressors'], 1):
            bar = "-" * min(int(abs(effect) * 100), 20)
            print(f"  {i:2d}. {tf:15s}  effect: {effect:.4f}  {bar}")
    else:
        print("    (none found)")
        
    print(f"\n  Summary: {result['total_activators']} activators, {result['total_repressors']} repressors")
    print(f"{'='*70}\n")


def print_tf_targets(result: Dict):
    """Pretty print TF target results."""
    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return
        
    print(f"\n{'='*70}")
    print(f"  TARGETS OF TF: {result['tf']}")
    print(f"{'='*70}")
    
    print(f"\n  ACTIVATED GENES (expression INCREASES when {result['tf']} is active):")
    print(f"  " + "-"*50)
    if result['activated_genes']:
        for i, (gene, effect) in enumerate(result['activated_genes'], 1):
            bar = "+" * min(int(effect * 100), 20)
            print(f"  {i:2d}. {gene:15s}  effect: +{effect:.4f}  {bar}")
    else:
        print("    (none found)")
        
    print(f"\n  REPRESSED GENES (expression DECREASES when {result['tf']} is active):")
    print(f"  " + "-"*50)
    if result['repressed_genes']:
        for i, (gene, effect) in enumerate(result['repressed_genes'], 1):
            bar = "-" * min(int(abs(effect) * 100), 20)
            print(f"  {i:2d}. {gene:15s}  effect: {effect:.4f}  {bar}")
    else:
        print("    (none found)")
        
    print(f"\n  Summary: {result['total_activated']} activated, {result['total_repressed']} repressed")
    print(f"{'='*70}\n")


def print_strategy(result: Dict, direction: str = 'up'):
    """Pretty print regulation strategy."""
    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return
        
    print(f"\n{'='*70}")
    if direction == 'up':
        print(f"  STRATEGY TO UPREGULATE: {result['gene']}")
    else:
        print(f"  STRATEGY TO DOWNREGULATE: {result['gene']}")
    print(f"{'='*70}")
    
    if direction == 'up':
        data = result['to_upregulate']
        
        print(f"\n  OPTION 1: OVEREXPRESS these TFs (activators):")
        print(f"  " + "-"*50)
        for i, (tf, effect) in enumerate(data['overexpress_these_TFs'], 1):
            print(f"  {i:2d}. Overexpress {tf:15s}  (effect: +{effect:.4f})")
            
        print(f"\n  OPTION 2: KNOCKOUT these TFs (repressors):")
        print(f"  " + "-"*50)
        for i, (tf, effect) in enumerate(data['knockout_these_TFs'], 1):
            print(f"  {i:2d}. Knockout {tf:15s}  (removes repression: {effect:.4f})")
    else:
        data = result['to_downregulate']
        
        print(f"\n  OPTION 1: KNOCKOUT these TFs (activators):")
        print(f"  " + "-"*50)
        for i, (tf, effect) in enumerate(data['knockout_these_TFs'], 1):
            print(f"  {i:2d}. Knockout {tf:15s}  (removes activation: +{effect:.4f})")
            
        print(f"\n  OPTION 2: OVEREXPRESS these TFs (repressors):")
        print(f"  " + "-"*50)
        for i, (tf, effect) in enumerate(data['overexpress_these_TFs'], 1):
            print(f"  {i:2d}. Overexpress {tf:15s}  (effect: {effect:.4f})")
            
    print(f"\n{result.get('explanation', '')}")
    print(f"{'='*70}\n")


def visualize_gene_regulation(
    result: Dict,
    output_path: str = 'gene_regulation.png',
    top_k: int = 15
):
    """Create visualization of gene regulation."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return
        
    if 'error' in result:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    gene_name = result['gene']
    
    # Activators
    ax1 = axes[0]
    activators = result['activators'][:top_k]
    if activators:
        tfs = [a[0] for a in activators]
        effects = [a[1] for a in activators]
        colors = ['#2ecc71'] * len(tfs)
        
        y_pos = np.arange(len(tfs))
        ax1.barh(y_pos, effects, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(tfs)
        ax1.invert_yaxis()
        ax1.set_xlabel('Effect Size')
        ax1.set_title(f'Activators of {gene_name}\n(Overexpress to increase)')
        
    # Repressors
    ax2 = axes[1]
    repressors = result['repressors'][:top_k]
    if repressors:
        tfs = [r[0] for r in repressors]
        effects = [abs(r[1]) for r in repressors]
        colors = ['#e74c3c'] * len(tfs)
        
        y_pos = np.arange(len(tfs))
        ax2.barh(y_pos, effects, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(tfs)
        ax2.invert_yaxis()
        ax2.set_xlabel('Effect Size (absolute)')
        ax2.set_title(f'Repressors of {gene_name}\n(Knockout to increase)')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def interactive_mode(model_data: Dict):
    """Interactive query mode."""
    print("\n" + "="*70)
    print("  INTERACTIVE GRN QUERY MODE")
    print("="*70)
    print("\nCommands:")
    print("  gene <name>     - Find regulators of a gene (e.g., 'gene CD3D')")
    print("  tf <name>       - Find targets of a TF (e.g., 'tf STAT1')")
    print("  up <gene>       - Strategy to upregulate a gene")
    print("  down <gene>     - Strategy to downregulate a gene")
    print("  list genes      - List all genes")
    print("  list tfs        - List all TFs")
    print("  search <term>   - Search for gene/TF names")
    print("  quit            - Exit")
    print("="*70 + "\n")
    
    grn = model_data['grn_matrix']
    genes = model_data['gene_names']
    tfs = model_data['tf_names']
    
    while True:
        try:
            cmd = input("GRN> ").strip()
        except EOFError:
            break
            
        if not cmd:
            continue
            
        parts = cmd.split()
        action = parts[0].lower()
        
        if action == 'quit' or action == 'exit' or action == 'q':
            break
            
        elif action == 'gene' and len(parts) > 1:
            gene_name = parts[1]
            result = query_gene_regulators(gene_name, grn, genes, tfs)
            print_gene_regulation(result)
            
        elif action == 'tf' and len(parts) > 1:
            tf_name = parts[1]
            result = query_tf_targets(tf_name, grn, genes, tfs)
            print_tf_targets(result)
            
        elif action == 'up' and len(parts) > 1:
            gene_name = parts[1]
            result = query_upregulation_strategy(gene_name, grn, genes, tfs)
            print_strategy(result, 'up')
            
        elif action == 'down' and len(parts) > 1:
            gene_name = parts[1]
            result = query_downregulation_strategy(gene_name, grn, genes, tfs)
            print_strategy(result, 'down')
            
        elif action == 'list':
            if len(parts) > 1 and parts[1].lower() == 'genes':
                print(f"\nGenes ({len(genes)} total):")
                print(", ".join(genes[:50]) + "...")
            elif len(parts) > 1 and parts[1].lower() == 'tfs':
                print(f"\nTFs ({len(tfs)} total):")
                print(", ".join(tfs[:50]) + "...")
                
        elif action == 'search' and len(parts) > 1:
            term = parts[1].upper()
            matching_genes = [g for g in genes if term in str(g).upper()]
            matching_tfs = [t for t in tfs if term in str(t).upper()]
            print(f"\nMatching genes ({len(matching_genes)}): {matching_genes[:20]}")
            print(f"Matching TFs ({len(matching_tfs)}): {matching_tfs[:20]}")
            
        else:
            print("Unknown command. Type 'help' for usage.")


def main():
    parser = argparse.ArgumentParser(description='Query GRN Model v5')
    parser.add_argument('--model', type=str, required=True, help='Model directory')
    parser.add_argument('--gene', type=str, help='Query regulators of this gene')
    parser.add_argument('--tf', type=str, help='Query targets of this TF')
    parser.add_argument('--up', type=str, help='Strategy to upregulate this gene')
    parser.add_argument('--down', type=str, help='Strategy to downregulate this gene')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top results')
    parser.add_argument('--save-plot', type=str, help='Save visualization to file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model_data = load_grn_model(args.model)
    
    grn = model_data['grn_matrix']
    genes = model_data['gene_names']
    tfs = model_data['tf_names']
    
    print(f"\nLoaded GRN: {grn.shape[0]} genes x {grn.shape[1]} TFs")
    print(f"Cell types used: {model_data['celltype_names']}")
    print(f"Model metrics: {model_data['metrics']}")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model_data)
        return
        
    # Query gene regulators
    if args.gene:
        result = query_gene_regulators(args.gene, grn, genes, tfs, args.top_k)
        print_gene_regulation(result)
        
        if args.save_plot:
            visualize_gene_regulation(result, args.save_plot, args.top_k)
            
    # Query TF targets
    if args.tf:
        result = query_tf_targets(args.tf, grn, genes, tfs, args.top_k)
        print_tf_targets(result)
        
    # Upregulation strategy
    if args.up:
        result = query_upregulation_strategy(args.up, grn, genes, tfs, args.top_k)
        print_strategy(result, 'up')
        
    # Downregulation strategy
    if args.down:
        result = query_downregulation_strategy(args.down, grn, genes, tfs, args.top_k)
        print_strategy(result, 'down')
        
    # Default: show help
    if not any([args.gene, args.tf, args.up, args.down]):
        print("\nNo query specified. Use --gene, --tf, --up, --down, or --interactive")
        print("\nExamples:")
        print(f"  python {sys.argv[0]} --model {args.model} --gene CD3D")
        print(f"  python {sys.argv[0]} --model {args.model} --tf STAT1")
        print(f"  python {sys.argv[0]} --model {args.model} --up CD3D")
        print(f"  python {sys.argv[0]} --model {args.model} --interactive")


if __name__ == '__main__':
    main()


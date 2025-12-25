#!/usr/bin/env python3
"""
Visualize Gene Regulatory Networks

Creates publication-quality visualizations for GRN analysis:
1. Network graphs (TF-Gene regulatory network)
2. Heatmaps (TF activity, expression changes)
3. Sankey diagrams (pathway flow)
4. UMAP plots (cell state embedding)

Usage:
    python scripts/visualize_grn.py \
        --results results/analysis_SSTR2/results.json \
        --output results/analysis_SSTR2/figures/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palettes
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'bg': '#FAFAFA',
    'text': '#2C3E50',
    'ko': '#E74C3C',
    'oe': '#27AE60',
    'tf': '#9B59B6',
    'gene': '#3498DB'
}


def create_grn_network_plot(
    edges: List[Dict],
    target_gene: str,
    output_path: Path,
    max_nodes: int = 30
):
    """
    Create network visualization of GRN.
    
    Args:
        edges: List of edge dicts with source, target, weight
        target_gene: Central target gene
        output_path: Output file path
        max_nodes: Maximum nodes to show
    """
    try:
        import networkx as nx
    except ImportError:
        print("Warning: networkx not available, skipping network plot")
        return
        
    # Build graph
    G = nx.DiGraph()
    
    # Add target node
    G.add_node(target_gene, node_type='target')
    
    # Add edges
    edge_weights = []
    for edge in edges[:max_nodes]:
        source = edge.get('source', edge.get('tf', ''))
        target = edge.get('target', target_gene)
        weight = edge.get('weight', edge.get('score', 0.5))
        
        if source and target:
            G.add_node(source, node_type='tf')
            G.add_edge(source, target, weight=weight)
            edge_weights.append(weight)
            
    if len(G.nodes) < 2:
        print("Not enough nodes for network plot")
        return
        
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Center target gene
    pos[target_gene] = np.array([0, 0])
    
    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor(COLORS['bg'])
    
    # Draw edges with varying width
    if edge_weights:
        max_weight = max(edge_weights) if edge_weights else 1
        for (u, v, d) in G.edges(data=True):
            weight = d.get('weight', 0.5)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                width=2 + weight / max_weight * 3,
                alpha=0.3 + weight / max_weight * 0.5,
                edge_color=COLORS['secondary'],
                arrows=True,
                arrowsize=15,
                ax=ax
            )
            
    # Draw TF nodes
    tf_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'tf']
    nx.draw_networkx_nodes(
        G, pos, nodelist=tf_nodes,
        node_color=COLORS['tf'],
        node_size=800,
        alpha=0.8,
        ax=ax
    )
    
    # Draw target node (larger)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[target_gene],
        node_color=COLORS['accent'],
        node_size=2000,
        alpha=0.9,
        ax=ax
    )
    
    # Labels
    nx.draw_networkx_labels(
        G, pos, font_size=9, font_weight='bold',
        font_color=COLORS['text'], ax=ax
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['accent'], label=f'Target: {target_gene}'),
        mpatches.Patch(color=COLORS['tf'], label='Transcription Factors'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title(f'Gene Regulatory Network for {target_gene}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_strategy_heatmap(
    ko_strategies: List[Tuple[str, float]],
    oe_strategies: List[Tuple[str, float]],
    target_gene: str,
    output_path: Path,
    top_n: int = 15
):
    """
    Create heatmap of perturbation strategies.
    
    Args:
        ko_strategies: List of (gene, effect) for KO
        oe_strategies: List of (gene, effect) for OE
        target_gene: Target gene name
        output_path: Output path
        top_n: Number of top strategies to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # KO strategies
    ko_genes = [s[0] for s in ko_strategies[:top_n]]
    ko_effects = [s[1] for s in ko_strategies[:top_n]]
    
    colors_ko = [COLORS['success'] if e > 0 else COLORS['ko'] for e in ko_effects]
    ax1.barh(range(len(ko_genes)), ko_effects, color=colors_ko, alpha=0.8)
    ax1.set_yticks(range(len(ko_genes)))
    ax1.set_yticklabels(ko_genes, fontsize=9)
    ax1.set_xlabel('Effect on Target', fontsize=11)
    ax1.set_title(f'Knockdown Strategies for {target_gene}', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color=COLORS['text'], linestyle='-', linewidth=0.5)
    ax1.invert_yaxis()
    
    # OE strategies
    oe_genes = [s[0] for s in oe_strategies[:top_n]]
    oe_effects = [s[1] for s in oe_strategies[:top_n]]
    
    colors_oe = [COLORS['success'] if e > 0 else COLORS['ko'] for e in oe_effects]
    ax2.barh(range(len(oe_genes)), oe_effects, color=colors_oe, alpha=0.8)
    ax2.set_yticks(range(len(oe_genes)))
    ax2.set_yticklabels(oe_genes, fontsize=9)
    ax2.set_xlabel('Effect on Target', fontsize=11)
    ax2.set_title(f'Overexpression Strategies for {target_gene}', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color=COLORS['text'], linestyle='-', linewidth=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_tf_binding_heatmap(
    tf_regulators: List[Tuple[str, float]],
    target_gene: str,
    output_path: Path,
    top_n: int = 20
):
    """
    Create heatmap of TF binding scores from ChIP-Atlas.
    
    Args:
        tf_regulators: List of (tf, binding_score)
        target_gene: Target gene name
        output_path: Output path
        top_n: Number of TFs to show
    """
    tfs = [r[0] for r in tf_regulators[:top_n]]
    scores = [r[1] for r in tf_regulators[:top_n]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', COLORS['tf']])
    
    # Bar plot
    bars = ax.barh(range(len(tfs)), scores, color=COLORS['tf'], alpha=0.8)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
                
    ax.set_yticks(range(len(tfs)))
    ax.set_yticklabels(tfs, fontsize=10)
    ax.set_xlabel('Binding Score (ChIP-Atlas)', fontsize=11)
    ax.set_title(f'Top TF Regulators of {target_gene}', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_pathway_enrichment_plot(
    pathways: Dict[str, List[str]],
    output_path: Path,
    max_pathways: int = 10
):
    """
    Create pathway enrichment visualization.
    
    Args:
        pathways: Dict mapping domain to list of pathways
        output_path: Output path
        max_pathways: Max pathways per domain
    """
    n_domains = len(pathways)
    if n_domains == 0:
        print("No pathway data available")
        return
        
    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 8))
    
    if n_domains == 1:
        axes = [axes]
        
    colors_list = [COLORS['secondary'], COLORS['success'], COLORS['warning'], 
                   COLORS['tf'], COLORS['accent']]
    
    for idx, (domain, paths) in enumerate(pathways.items()):
        ax = axes[idx]
        paths = paths[:max_pathways]
        
        # Truncate long names
        paths_short = [p[:40] + '...' if len(p) > 40 else p for p in paths]
        
        y_pos = range(len(paths_short))
        ax.barh(y_pos, range(len(paths_short), 0, -1), 
                color=colors_list[idx % len(colors_list)], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(paths_short, fontsize=9)
        ax.set_xlabel('Rank', fontsize=10)
        ax.set_title(domain, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        
    plt.suptitle('Pathway Enrichment Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_figure(
    results: Dict[str, Any],
    output_path: Path
):
    """
    Create summary figure combining multiple visualizations.
    
    Args:
        results: Analysis results dict
        output_path: Output path
    """
    fig = plt.figure(figsize=(16, 12))
    
    target = results.get('target_gene', 'Unknown')
    direction = results.get('desired_direction', 'increase')
    
    # Title
    fig.suptitle(f'GRN Analysis Summary: {target} ({direction.upper()})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. TF Regulators (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    tf_regs = results.get('tf_regulators', [])[:10]
    if tf_regs:
        tfs = [r[0] for r in tf_regs]
        scores = [r[1] for r in tf_regs]
        ax1.barh(range(len(tfs)), scores, color=COLORS['tf'], alpha=0.8)
        ax1.set_yticks(range(len(tfs)))
        ax1.set_yticklabels(tfs, fontsize=8)
        ax1.set_xlabel('Binding Score')
        ax1.invert_yaxis()
    ax1.set_title('Top TF Regulators', fontweight='bold')
    
    # 2. KO Strategies (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ko_strats = results.get('ko_strategies', [])[:10]
    if ko_strats:
        genes = [s[0] for s in ko_strats]
        effects = [s[1] for s in ko_strats]
        colors = [COLORS['success'] if e > 0 else COLORS['ko'] for e in effects]
        ax2.barh(range(len(genes)), effects, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(genes)))
        ax2.set_yticklabels(genes, fontsize=8)
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.invert_yaxis()
    ax2.set_title('Top KO Strategies', fontweight='bold')
    
    # 3. OE Strategies (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    oe_strats = results.get('oe_strategies', [])[:10]
    if oe_strats:
        genes = [s[0] for s in oe_strats]
        effects = [s[1] for s in oe_strats]
        colors = [COLORS['success'] if e > 0 else COLORS['ko'] for e in effects]
        ax3.barh(range(len(genes)), effects, color=colors, alpha=0.8)
        ax3.set_yticks(range(len(genes)))
        ax3.set_yticklabels(genes, fontsize=8)
        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.invert_yaxis()
    ax3.set_title('Top OE Strategies', fontweight='bold')
    
    # 4. Network info (bottom, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    network = results.get('network', {})
    info_text = f"""
    Network Statistics:
    - Target Gene: {target}
    - Total Nodes: {network.get('n_nodes', 'N/A')}
    - Total Edges: {network.get('n_edges', 'N/A')}
    
    Analysis Direction: {direction.upper()} {target}
    
    Top 3 Recommendations:
    """
    
    # Add top recommendations
    if ko_strats and direction == 'increase':
        info_text += f"\n    1. Knock out {ko_strats[0][0]} (effect: {ko_strats[0][1]:.4f})"
        if len(ko_strats) > 1:
            info_text += f"\n    2. Knock out {ko_strats[1][0]} (effect: {ko_strats[1][1]:.4f})"
    if oe_strats and direction == 'increase':
        if len(oe_strats) > 0:
            info_text += f"\n    3. Overexpress {oe_strats[0][0]} (effect: {oe_strats[0][1]:.4f})"
            
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=COLORS['bg'], alpha=0.8))
    ax4.axis('off')
    ax4.set_title('Analysis Summary', fontweight='bold')
    
    # 5. Pathway info (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    pathways = results.get('pathways', {})
    if pathways:
        pathway_text = "Enriched Pathways:\n\n"
        for domain, paths in list(pathways.items())[:2]:
            pathway_text += f"{domain}:\n"
            for p in paths[:3]:
                pathway_text += f"  - {p[:30]}...\n" if len(p) > 30 else f"  - {p}\n"
        ax5.text(0.1, 0.5, pathway_text, transform=ax5.transAxes, fontsize=9,
                 verticalalignment='center')
    ax5.axis('off')
    ax5.set_title('Pathway Enrichment', fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GRN analysis results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results.json from query_target.py')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
        
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target = results.get('target_gene', 'Unknown')
    
    print(f"\nGenerating visualizations for {target}...")
    
    # 1. TF binding heatmap
    tf_regs = results.get('tf_regulators', [])
    if tf_regs:
        create_tf_binding_heatmap(
            tf_regs, target,
            output_dir / 'tf_binding_heatmap.png'
        )
        
    # 2. Strategy heatmap
    ko_strats = results.get('ko_strategies', [])
    oe_strats = results.get('oe_strategies', [])
    if ko_strats or oe_strats:
        create_strategy_heatmap(
            ko_strats, oe_strats, target,
            output_dir / 'strategy_heatmap.png'
        )
        
    # 3. Network plot
    network = results.get('network', {})
    edges = network.get('edges', [])
    if edges:
        create_grn_network_plot(
            edges, target,
            output_dir / 'grn_network.png'
        )
        
    # 4. Pathway enrichment
    pathways = results.get('pathways', {})
    if pathways:
        create_pathway_enrichment_plot(
            pathways,
            output_dir / 'pathway_enrichment.png'
        )
        
    # 5. Summary figure
    create_summary_figure(
        results,
        output_dir / 'summary_figure.png'
    )
    
    print(f"\nAll figures saved to {output_dir}")


if __name__ == '__main__':
    main()


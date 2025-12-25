#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: DISCOVERY MODE - EXPLORE UNKNOWN REGULATORY CONNECTIONS
================================================================================

Phase 2 of BRIDGE: After training on known network (OmniPath + ChIP-Atlas),
this script explores potentially novel regulatory connections.

Key Features:
1. Start with optimized known network
2. For target gene(s), identify high-confidence regulators
3. Expand neighborhood to test unknown connections
4. Report discoveries with confidence scores

Workflow:
```
Phase 1 (train_v7.py)           Phase 2 (this script)
┌──────────────────────┐        ┌────────────────────────────────────┐
│ Learn known network  │   →    │ Query: "What regulates SSTR2?"     │
│ (OmniPath + ChIP)    │        │                                    │
└──────────────────────┘        │ Known: TF_A → SSTR2 (high conf)    │
                                │ Known: Gene_B → TF_A (medium conf) │
                                │                                    │
                                │ DISCOVERY MODE:                    │
                                │ Test Gene_C → ? → SSTR2            │
                                │ (Gene_C correlated but not in DB)  │
                                │                                    │
                                │ Novel: Gene_C → TF_A → SSTR2 !     │
                                └────────────────────────────────────┘
```

Usage:
    python scripts/discover_regulators.py \
        --gene SSTR2 \
        --model results/sclc_v7/ \
        --singlecell data/sclc/preprocessed.h5ad \
        --omnipath data/omnipath/ \
        --output results/discovery_SSTR2/ \
        --top-k 50 \
        --discovery-threshold 0.1

Author: BRIDGE Team
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DISCOVERY ENGINE
# =============================================================================

class RegulatoryDiscoveryEngine:
    """
    Engine for discovering novel regulatory connections.
    
    Combines:
    - Known network (from training)
    - Expression correlations
    - Perturbation analysis
    """
    
    def __init__(self, model_dir: str, singlecell_path: str,
                 omnipath_dir: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.singlecell_path = singlecell_path
        self.omnipath_dir = Path(omnipath_dir) if omnipath_dir else None
        
        self._load_data()
        self._load_model()
        self._load_known_network()
        
    def _load_data(self):
        """Load expression and gene names."""
        import scanpy as sc
        
        print("Loading single-cell data...")
        adata = sc.read_h5ad(self.singlecell_path)
        
        if len(adata) < 500:
            self.expression = adata.X
            if hasattr(self.expression, 'toarray'):
                self.expression = self.expression.toarray()
        else:
            # Aggregate
            from collections import defaultdict
            ct_expr = defaultdict(list)
            for i in range(len(adata)):
                ct = adata.obs.iloc[i].get('celltype', str(i))
                ct_expr[ct].append(adata.X[i])
            celltypes = list(ct_expr.keys())
            self.expression = np.array([
                np.mean(ct_expr[ct], axis=0) for ct in celltypes
            ])
        
        self.gene_names = list(adata.var_names)
        self.n_genes = len(self.gene_names)
        self.n_celltypes = self.expression.shape[0]
        
        # Create lookup
        self.gene_to_idx = {g.upper(): i for i, g in enumerate(self.gene_names)}
        
        print(f"  Expression: {self.expression.shape}")
        
    def _load_model(self):
        """Load trained signaling networks."""
        print("Loading trained model...")
        
        # TF names
        tf_path = self.model_dir / 'tf_names.csv'
        if tf_path.exists():
            self.tf_names = pd.read_csv(tf_path, header=None)[0].tolist()
        else:
            self.tf_names = []
        
        self.n_tfs = len(self.tf_names)
        self.tf_to_idx = {t.upper(): i for i, t in enumerate(self.tf_names)}
        self.tf_set = set(t.upper() for t in self.tf_names)
        
        # Load networks
        self.networks = {}
        
        for name in ['tf_to_gene', 'gene_to_gene', 'gene_to_tf', 'tf_to_tf']:
            path = self.model_dir / f'{name}.npy'
            if path.exists():
                self.networks[name] = np.load(path)
                print(f"  {name}: {self.networks[name].shape}")
            else:
                print(f"  {name}: not found")
        
        # Also check for GRN matrix (backwards compatibility)
        grn_path = self.model_dir / 'grn_matrix_CORRECT.npy'
        if grn_path.exists():
            self.grn_matrix = np.load(grn_path)
        elif (self.model_dir / 'grn_matrix.npy').exists():
            self.grn_matrix = np.load(self.model_dir / 'grn_matrix.npy')
        else:
            self.grn_matrix = self.networks.get('tf_to_gene', np.zeros((self.n_genes, self.n_tfs)))
        
    def _load_known_network(self):
        """Load known edges from OmniPath."""
        self.known_edges = set()
        
        if self.omnipath_dir and (self.omnipath_dir / 'gene_gene_signaling.npy').exists():
            gene_gene = np.load(self.omnipath_dir / 'gene_gene_signaling.npy')
            src, dst = np.nonzero(gene_gene)
            for i in range(len(src)):
                self.known_edges.add((self.gene_names[src[i]].upper(), 
                                     self.gene_names[dst[i]].upper()))
            print(f"  Known Gene→Gene edges: {len(self.known_edges):,}")
        
        if self.omnipath_dir and (self.omnipath_dir / 'gene_tf_signaling.npy').exists():
            gene_tf = np.load(self.omnipath_dir / 'gene_tf_signaling.npy')
            src, dst = np.nonzero(gene_tf)
            for i in range(len(src)):
                if dst[i] < self.n_tfs:
                    self.known_edges.add((self.gene_names[src[i]].upper(),
                                         self.tf_names[dst[i]].upper()))
    
    def get_target_regulators(self, target_gene: str, top_k: int = 50) -> pd.DataFrame:
        """
        Get known and learned regulators for a target gene.
        
        Returns DataFrame with:
        - TF/Gene name
        - Effect (from learned network)
        - Is known (in OmniPath/ChIP)
        - Confidence score
        """
        target_upper = target_gene.upper()
        
        if target_upper not in self.gene_to_idx:
            raise ValueError(f"Gene {target_gene} not found")
        
        target_idx = self.gene_to_idx[target_upper]
        
        results = []
        
        # TF regulators
        if self.grn_matrix is not None and target_idx < self.grn_matrix.shape[0]:
            tf_effects = self.grn_matrix[target_idx, :]
            
            for i, tf in enumerate(self.tf_names):
                if i >= len(tf_effects):
                    break
                    
                effect = tf_effects[i]
                is_known = (tf.upper(), target_upper) in self.known_edges
                
                # Confidence based on effect magnitude and consistency
                confidence = np.abs(effect)
                
                results.append({
                    'regulator': tf,
                    'type': 'TF',
                    'effect': effect,
                    'direction': 'activator' if effect > 0 else 'repressor',
                    'is_known': is_known,
                    'confidence': confidence,
                })
        
        # Gene regulators (from gene→gene network)
        if 'gene_to_gene' in self.networks:
            gene_gene = self.networks['gene_to_gene']
            if target_idx < gene_gene.shape[1]:
                gene_effects = gene_gene[:, target_idx]
                
                for i, gene in enumerate(self.gene_names):
                    if i == target_idx:
                        continue
                    if gene.upper() in self.tf_set:
                        continue  # Already counted as TF
                    
                    effect = gene_effects[i]
                    if np.abs(effect) < 1e-6:
                        continue
                    
                    is_known = (gene.upper(), target_upper) in self.known_edges
                    
                    results.append({
                        'regulator': gene,
                        'type': 'Gene',
                        'effect': effect,
                        'direction': 'activator' if effect > 0 else 'repressor',
                        'is_known': is_known,
                        'confidence': np.abs(effect),
                    })
        
        df = pd.DataFrame(results)
        if len(df) == 0:
            return df
        
        # Sort by confidence
        df = df.sort_values('confidence', ascending=False).head(top_k)
        
        # Normalize confidence to 0-1
        if df['confidence'].max() > 0:
            df['confidence'] = df['confidence'] / df['confidence'].max()
        
        return df.reset_index(drop=True)
    
    def discover_novel_regulators(self, target_gene: str, 
                                   discovery_threshold: float = 0.1,
                                   max_distance: int = 2) -> pd.DataFrame:
        """
        Discover potentially novel regulators not in known network.
        
        Strategy:
        1. Find genes strongly correlated with target
        2. Check if they're NOT in known network
        3. Test if adding them as regulators improves prediction
        4. Rank by discovery confidence
        """
        target_upper = target_gene.upper()
        target_idx = self.gene_to_idx[target_upper]
        
        print(f"\nDISCOVERY MODE: Finding novel regulators for {target_gene}")
        print("=" * 60)
        
        # Step 1: Find expression correlations
        target_expr = self.expression[:, target_idx]
        correlations = []
        
        for i, gene in enumerate(self.gene_names):
            if i == target_idx:
                continue
            
            gene_expr = self.expression[:, i]
            if np.std(gene_expr) < 1e-6:
                continue
            
            r, p = stats.pearsonr(target_expr, gene_expr)
            correlations.append({
                'gene': gene,
                'gene_idx': i,
                'correlation': r,
                'p_value': p,
            })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df[corr_df['p_value'] < 0.05]  # Significant only
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
        
        print(f"  Significant correlations: {len(corr_df)}")
        
        # Step 2: Identify novel candidates (not in known network)
        known_regulators = set()
        for edge in self.known_edges:
            if edge[1] == target_upper:
                known_regulators.add(edge[0])
        
        novel_candidates = []
        
        for _, row in corr_df.head(200).iterrows():  # Check top 200
            gene = row['gene']
            gene_upper = gene.upper()
            
            if gene_upper in known_regulators:
                continue  # Skip known
            if gene_upper == target_upper:
                continue
            
            # Check if this gene could regulate target through TFs
            potential_paths = self._find_regulatory_paths(
                gene_upper, target_upper, max_distance
            )
            
            # Compute discovery score
            discovery_score = self._compute_discovery_score(
                row['gene_idx'], target_idx, row['correlation']
            )
            
            if discovery_score > discovery_threshold:
                novel_candidates.append({
                    'gene': gene,
                    'correlation': row['correlation'],
                    'p_value': row['p_value'],
                    'discovery_score': discovery_score,
                    'potential_paths': len(potential_paths),
                    'path_examples': '; '.join(potential_paths[:3]),
                    'is_tf': gene_upper in self.tf_set,
                })
        
        novel_df = pd.DataFrame(novel_candidates)
        if len(novel_df) > 0:
            novel_df = novel_df.sort_values('discovery_score', ascending=False)
        
        print(f"  Novel candidates (score > {discovery_threshold}): {len(novel_df)}")
        
        return novel_df
    
    def _find_regulatory_paths(self, source: str, target: str, 
                               max_distance: int) -> List[str]:
        """Find potential regulatory paths from source to target."""
        paths = []
        
        # Direct TF regulation
        if source in self.tf_set:
            paths.append(f"{source} →TF→ {target}")
        
        # Through TF cascade
        if 'gene_to_tf' in self.networks:
            gene_tf = self.networks['gene_to_tf']
            source_idx = self.gene_to_idx.get(source, -1)
            
            if source_idx >= 0 and source_idx < gene_tf.shape[0]:
                # Find TFs this gene regulates
                tf_effects = gene_tf[source_idx, :]
                top_tfs = np.argsort(np.abs(tf_effects))[-5:]
                
                target_idx = self.gene_to_idx.get(target, -1)
                
                for tf_idx in top_tfs:
                    if tf_idx >= self.n_tfs:
                        continue
                    tf_name = self.tf_names[tf_idx]
                    
                    # Check if this TF regulates target
                    if target_idx >= 0 and target_idx < self.grn_matrix.shape[0]:
                        if tf_idx < self.grn_matrix.shape[1]:
                            tf_target_effect = self.grn_matrix[target_idx, tf_idx]
                            if np.abs(tf_target_effect) > 0.01:
                                paths.append(f"{source} → {tf_name} →TF→ {target}")
        
        return paths
    
    def _compute_discovery_score(self, source_idx: int, target_idx: int,
                                  correlation: float) -> float:
        """
        Compute discovery confidence score.
        
        Combines:
        - Expression correlation
        - Network propagation consistency
        - Variance explained
        """
        # Base: correlation strength
        corr_score = np.abs(correlation)
        
        # Check if source can explain target variance through the network
        source_expr = self.expression[:, source_idx]
        target_expr = self.expression[:, target_idx]
        
        # Simple regression to estimate effect
        if np.std(source_expr) > 1e-6:
            slope = np.cov(source_expr, target_expr)[0, 1] / np.var(source_expr)
            predicted = source_expr * slope
            r2 = 1 - np.var(target_expr - predicted) / np.var(target_expr)
            r2 = max(0, r2)
        else:
            r2 = 0
        
        # Combine scores
        discovery_score = 0.6 * corr_score + 0.4 * np.sqrt(r2)
        
        return discovery_score
    
    def generate_discovery_report(self, target_gene: str, output_dir: str,
                                   top_k: int = 50, discovery_threshold: float = 0.1):
        """Generate comprehensive discovery report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get known regulators
        print("\n" + "=" * 70)
        print(f"REGULATORY DISCOVERY REPORT: {target_gene}")
        print("=" * 70)
        
        known_df = self.get_target_regulators(target_gene, top_k)
        
        print(f"\nKNOWN REGULATORS (from trained network):")
        print("-" * 50)
        print(f"  Total: {len(known_df)}")
        print(f"  In database: {known_df['is_known'].sum()}")
        print(f"  TFs: {(known_df['type'] == 'TF').sum()}")
        print(f"  Genes: {(known_df['type'] == 'Gene').sum()}")
        
        # Discover novel
        novel_df = self.discover_novel_regulators(target_gene, discovery_threshold)
        
        print(f"\nNOVEL DISCOVERIES:")
        print("-" * 50)
        if len(novel_df) > 0:
            print(f"  Candidates: {len(novel_df)}")
            print(f"  With TF paths: {(novel_df['potential_paths'] > 0).sum()}")
            print("\n  Top 10 Novel Regulators:")
            for i, row in novel_df.head(10).iterrows():
                status = "TF" if row['is_tf'] else "Gene"
                print(f"    {i+1}. {row['gene']} ({status})")
                print(f"       Score: {row['discovery_score']:.3f}, r={row['correlation']:.3f}")
                if row['path_examples']:
                    print(f"       Paths: {row['path_examples']}")
        else:
            print("  No novel candidates found above threshold")
        
        # Save CSVs
        known_df.to_csv(output_dir / f'known_regulators_{target_gene}.csv', index=False)
        novel_df.to_csv(output_dir / f'novel_discoveries_{target_gene}.csv', index=False)
        
        # Generate figure
        self._create_discovery_figure(target_gene, known_df, novel_df, output_dir)
        
        print(f"\n✓ Report saved to: {output_dir}")
        
        return known_df, novel_df
    
    def _create_discovery_figure(self, target_gene: str, known_df: pd.DataFrame,
                                  novel_df: pd.DataFrame, output_dir: Path):
        """Create discovery visualization."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Panel A: Known regulators by confidence
        ax_a = fig.add_subplot(gs[0, 0])
        if len(known_df) > 0:
            top_known = known_df.head(20)
            colors = ['#06d6a0' if e > 0 else '#ef476f' for e in top_known['effect']]
            ax_a.barh(range(len(top_known)), top_known['confidence'], color=colors, alpha=0.8)
            ax_a.set_yticks(range(len(top_known)))
            ax_a.set_yticklabels(top_known['regulator'], fontsize=8)
            ax_a.invert_yaxis()
            ax_a.set_xlabel('Confidence')
            ax_a.set_title(f'Top Known Regulators of {target_gene}')
            
            # Mark database entries
            for i, is_known in enumerate(top_known['is_known']):
                if is_known:
                    ax_a.annotate('📚', (top_known['confidence'].iloc[i], i),
                                 fontsize=8, ha='left', va='center')
        
        # Panel B: Known vs Novel pie
        ax_b = fig.add_subplot(gs[0, 1])
        n_known_db = known_df['is_known'].sum() if len(known_df) > 0 else 0
        n_learned = len(known_df) - n_known_db if len(known_df) > 0 else 0
        n_novel = len(novel_df)
        
        sizes = [n_known_db, n_learned, n_novel]
        labels = [f'In Database\n({n_known_db})', 
                  f'Learned\n({n_learned})', 
                  f'Novel Discovery\n({n_novel})']
        colors = ['#3a86ff', '#8338ec', '#ff006e']
        
        ax_b.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, explode=(0, 0, 0.1))
        ax_b.set_title('Regulator Categories')
        
        # Panel C: Novel discoveries
        ax_c = fig.add_subplot(gs[0, 2])
        if len(novel_df) > 0:
            top_novel = novel_df.head(15)
            colors = ['#ff006e' if is_tf else '#fb5607' for is_tf in top_novel['is_tf']]
            ax_c.barh(range(len(top_novel)), top_novel['discovery_score'], color=colors, alpha=0.8)
            ax_c.set_yticks(range(len(top_novel)))
            ax_c.set_yticklabels(top_novel['gene'], fontsize=8)
            ax_c.invert_yaxis()
            ax_c.set_xlabel('Discovery Score')
            ax_c.set_title('Novel Regulatory Discoveries')
        else:
            ax_c.text(0.5, 0.5, 'No novel discoveries\nabove threshold',
                     ha='center', va='center', fontsize=12)
            ax_c.axis('off')
        
        # Panel D: Effect sizes (known)
        ax_d = fig.add_subplot(gs[1, 0])
        if len(known_df) > 0:
            activators = known_df[known_df['effect'] > 0]['effect']
            repressors = known_df[known_df['effect'] < 0]['effect']
            
            ax_d.hist(activators, bins=30, alpha=0.7, color='#06d6a0', label='Activators')
            ax_d.hist(repressors, bins=30, alpha=0.7, color='#ef476f', label='Repressors')
            ax_d.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax_d.set_xlabel('Effect Size')
            ax_d.set_ylabel('Count')
            ax_d.set_title('Distribution of Regulatory Effects')
            ax_d.legend()
        
        # Panel E: Correlation vs Discovery Score
        ax_e = fig.add_subplot(gs[1, 1])
        if len(novel_df) > 0:
            colors = ['#ff006e' if is_tf else '#fb5607' for is_tf in novel_df['is_tf']]
            ax_e.scatter(novel_df['correlation'].abs(), novel_df['discovery_score'],
                        c=colors, alpha=0.6, s=50)
            ax_e.set_xlabel('|Correlation|')
            ax_e.set_ylabel('Discovery Score')
            ax_e.set_title('Correlation vs Discovery Score')
            
            # Add gene labels for top candidates
            for _, row in novel_df.head(5).iterrows():
                ax_e.annotate(row['gene'], (abs(row['correlation']), row['discovery_score']),
                             fontsize=7, alpha=0.8)
        
        # Panel F: Summary text
        ax_f = fig.add_subplot(gs[1, 2])
        ax_f.axis('off')
        
        summary = f"""
DISCOVERY SUMMARY: {target_gene}
{'='*40}

KNOWN NETWORK:
  • Total regulators: {len(known_df)}
  • In database (OmniPath/ChIP): {n_known_db}
  • Learned from data: {n_learned}
  • TF regulators: {(known_df['type'] == 'TF').sum() if len(known_df) > 0 else 0}
  • Gene regulators: {(known_df['type'] == 'Gene').sum() if len(known_df) > 0 else 0}

NOVEL DISCOVERIES:
  • Candidates tested: {len(novel_df)}
  • With regulatory paths: {(novel_df['potential_paths'] > 0).sum() if len(novel_df) > 0 else 0}
  
TOP NOVEL REGULATORS:
"""
        if len(novel_df) > 0:
            for i, row in novel_df.head(5).iterrows():
                summary += f"  {i+1}. {row['gene']} (score: {row['discovery_score']:.2f})\n"
        else:
            summary += "  None found above threshold\n"
        
        summary += f"""
INTERPRETATION:
  Green bars = Activators (↑ gene → ↑ {target_gene})
  Red bars = Repressors (↑ gene → ↓ {target_gene})
  📚 = Known in database
  
Novel discoveries suggest NEW regulatory
connections not in OmniPath/ChIP-Atlas.
Validate experimentally!
"""
        
        ax_f.text(0.05, 0.95, summary, transform=ax_f.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f7f7f7', alpha=0.9))
        
        plt.suptitle(f'Regulatory Discovery Report: {target_gene}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig.savefig(output_dir / f'discovery_report_{target_gene}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved figure: discovery_report_{target_gene}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BRIDGE 2.0: Discovery Mode')
    parser.add_argument('--gene', type=str, required=True, help='Target gene')
    parser.add_argument('--model', type=str, required=True, help='Trained model directory')
    parser.add_argument('--singlecell', type=str, required=True, help='Single-cell h5ad')
    parser.add_argument('--omnipath', type=str, default=None, help='OmniPath directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--top-k', type=int, default=50, help='Top K known regulators')
    parser.add_argument('--discovery-threshold', type=float, default=0.1,
                        help='Threshold for novel discoveries')
    
    args = parser.parse_args()
    
    engine = RegulatoryDiscoveryEngine(
        model_dir=args.model,
        singlecell_path=args.singlecell,
        omnipath_dir=args.omnipath
    )
    
    known_df, novel_df = engine.generate_discovery_report(
        target_gene=args.gene,
        output_dir=args.output,
        top_k=args.top_k,
        discovery_threshold=args.discovery_threshold
    )


if __name__ == '__main__':
    main()


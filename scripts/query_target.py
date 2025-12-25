#!/usr/bin/env python3
"""
Query Target Gene Regulators

Given a trained Monster GRN Model, query for:
1. What TFs regulate a target gene?
2. What genes/pathways should be targeted to increase/decrease expression?
3. Generate regulatory network visualization

Usage:
    python scripts/query_target.py \
        --model results/grn_model/best_model.pt \
        --gene SSTR2 \
        --direction increase \
        --output results/SSTR2_analysis/
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.monster_grn import MonsterGRNModel, create_monster_grn


class GRNQueryEngine:
    """
    Query engine for trained GRN model.
    
    Provides methods to:
    - Find TF regulators for any gene
    - Simulate perturbation effects
    - Generate regulatory network graphs
    - Identify optimal perturbation strategies
    """
    
    def __init__(
        self,
        model_path: str,
        gene_names: List[str],
        tf_names: Optional[List[str]] = None,
        tf_gene_binding: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize query engine.
        
        Args:
            model_path: Path to trained model checkpoint
            gene_names: List of gene names
            tf_names: Optional TF names
            tf_gene_binding: Optional TF-Gene binding matrix
            device: Device for inference
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.gene_names = gene_names
        self.gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
        self.n_genes = len(gene_names)
        
        self.tf_names = tf_names or []
        self.tf_to_idx = {t.upper(): i for i, t in enumerate(tf_names)} if tf_names else {}
        
        self.tf_gene_binding = tf_gene_binding
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> MonsterGRNModel:
        """Load model from checkpoint."""
        print(f"Loading model from {model_path}...")
        
        model_dir = Path(model_path).parent
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get('config')
        
        # If config is None, try loading from model_config.json
        if config is None:
            config_file = model_dir / 'model_config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  Loaded config from {config_file}")
            else:
                config = {}
        
        # Extract config values with fallbacks
        n_genes = config.get('n_genes', self.n_genes)
        n_tfs = config.get('n_tfs', len(self.tf_names) if self.tf_names else 100)
        n_batches = config.get('n_batches', 1)
        hidden_dim = config.get('hidden_dim', 512)
        num_transformer_layers = config.get('num_transformer_layers', 2)
        num_gcn_layers = config.get('num_gcn_layers', 3)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)
        
        # Create model with exact config
        model = create_monster_grn(
            n_genes=n_genes,
            n_tfs=n_tfs,
            n_batches=n_batches,
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_gcn_layers=num_gcn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        
        # Cache binding matrix if available
        if self.tf_gene_binding is not None:
            binding_tensor = torch.FloatTensor(self.tf_gene_binding).to(self.device)
            model.set_binding_matrix(binding_tensor)
        
        print(f"  Loaded model: {n_genes} genes, {n_tfs} TFs, hidden={hidden_dim}")
        
        return model
        
    def find_gene_index(self, gene: str) -> int:
        """Find index of gene by name."""
        gene_upper = gene.upper()
        
        if gene_upper in self.gene_to_idx:
            return self.gene_to_idx[gene_upper]
            
        # Try partial match
        for name, idx in self.gene_to_idx.items():
            if gene_upper in name or name in gene_upper:
                print(f"  Matched '{gene}' to '{self.gene_names[idx]}'")
                return idx
                
        raise ValueError(f"Gene '{gene}' not found")
        
    def get_tf_regulators(
        self,
        gene: str,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top TF regulators for a gene from ChIP-Atlas data.
        
        Args:
            gene: Gene name
            top_k: Number of top TFs to return
            
        Returns:
            List of (tf_name, binding_score) tuples
        """
        if self.tf_gene_binding is None:
            return []
            
        gene_idx = self.find_gene_index(gene)
        
        # Get binding scores for this gene
        binding_scores = self.tf_gene_binding[gene_idx]
        
        # Sort by score
        sorted_indices = np.argsort(binding_scores)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            if binding_scores[idx] > 0:
                results.append((self.tf_names[idx], float(binding_scores[idx])))
                
        return results
        
    @torch.no_grad()
    def simulate_perturbation(
        self,
        perturb_gene: str,
        direction: str = 'knockdown',
        baseline_expression: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate effect of perturbing a gene.
        
        Args:
            perturb_gene: Gene to perturb
            direction: 'knockdown' or 'overexpression'
            baseline_expression: Optional baseline expression (default: zeros)
            
        Returns:
            Dict with predicted effects
        """
        gene_idx = self.find_gene_index(perturb_gene)
        
        # Create baseline
        if baseline_expression is None:
            baseline = torch.zeros(1, self.n_genes, device=self.device)
        else:
            baseline = torch.FloatTensor(baseline_expression).unsqueeze(0).to(self.device)
            
        # Create perturbed
        perturbed = baseline.clone()
        if direction == 'knockdown':
            perturbed[0, gene_idx] = -2.0  # Low expression
        else:
            perturbed[0, gene_idx] = 2.0  # High expression
            
        # Get model predictions
        baseline_out = self.model(baseline, return_embeddings=True)
        perturbed_out = self.model(perturbed, return_embeddings=True)
        
        # Compute effect
        if 'predicted_expression' in baseline_out:
            baseline_pred = baseline_out['predicted_expression'].cpu().numpy()[0]
            perturbed_pred = perturbed_out['predicted_expression'].cpu().numpy()[0]
        elif 'reconstructed' in baseline_out:
            baseline_pred = baseline_out['reconstructed'].cpu().numpy()[0]
            perturbed_pred = perturbed_out['reconstructed'].cpu().numpy()[0]
        else:
            # Fallback to embedding-based prediction
            baseline_pred = baseline.cpu().numpy()[0]
            perturbed_pred = perturbed.cpu().numpy()[0]
            
        effect = perturbed_pred - baseline_pred
        
        return {
            'baseline': baseline_pred,
            'perturbed': perturbed_pred,
            'effect': effect,
            'perturbed_gene': perturb_gene,
            'direction': direction
        }
        
    def find_regulators(
        self,
        target_gene: str,
        desired_direction: str = 'increase',
        top_k: int = 20,
        expression_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Find genes/TFs that regulate target gene.
        
        For each potential regulator, simulate its perturbation and
        check effect on target gene.
        
        Args:
            target_gene: Gene to regulate
            desired_direction: 'increase' or 'decrease'
            top_k: Number of top regulators to return
            expression_data: Optional mean expression for baseline
            
        Returns:
            Dict with regulator analysis
        """
        target_idx = self.find_gene_index(target_gene)
        
        print(f"\nFinding regulators for {target_gene} ({desired_direction})...")
        
        # Strategy 1: Use ChIP-Atlas TF binding
        tf_regulators = self.get_tf_regulators(target_gene, top_k=50)
        
        # Strategy 2: Simulate perturbations for all genes
        ko_effects = []  # (gene, effect_on_target)
        oe_effects = []
        
        # Set baseline
        baseline = expression_data if expression_data is not None else None
        
        # Test each gene as perturbation
        print("Simulating perturbations...")
        for gene_idx in tqdm(range(min(self.n_genes, 5000))):  # Limit for speed
            gene_name = self.gene_names[gene_idx]
            
            # Knockdown
            try:
                result = self.simulate_perturbation(gene_name, 'knockdown', baseline)
                ko_effect = result['effect'][target_idx]
                ko_effects.append((gene_name, ko_effect))
            except Exception:
                pass
                
            # Overexpression
            try:
                result = self.simulate_perturbation(gene_name, 'overexpression', baseline)
                oe_effect = result['effect'][target_idx]
                oe_effects.append((gene_name, oe_effect))
            except Exception:
                pass
                
        # Sort by effect
        # Effect = how target gene changes when we perturb
        # Positive effect = target increases; Negative effect = target decreases
        if desired_direction == 'increase':
            # KO: want positive effect (KO repressor -> target increases)
            ko_sorted = sorted(ko_effects, key=lambda x: x[1], reverse=True)[:top_k]
            # OE: want positive effect (OE activator -> target increases)
            oe_sorted = sorted(oe_effects, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            # KO: want negative effect (KO activator -> target decreases)
            ko_sorted = sorted(ko_effects, key=lambda x: x[1])[:top_k]
            # OE: want negative effect (OE repressor -> target decreases)
            oe_sorted = sorted(oe_effects, key=lambda x: x[1])[:top_k]
            
        return {
            'target_gene': target_gene,
            'desired_direction': desired_direction,
            'tf_regulators': tf_regulators,
            'ko_strategies': ko_sorted,
            'oe_strategies': oe_sorted
        }
        
    def build_regulatory_network(
        self,
        target_gene: str,
        max_depth: int = 3,
        min_score: float = 0.1
    ) -> Dict[str, Any]:
        """
        Build regulatory network graph for a target gene.
        
        Traces upstream regulators through TF cascade.
        
        Args:
            target_gene: Gene to analyze
            max_depth: Maximum depth to trace
            min_score: Minimum binding score to include
            
        Returns:
            Dict with network structure
        """
        print(f"\nBuilding regulatory network for {target_gene}...")
        
        nodes = set()
        edges = []
        
        def trace_upstream(gene: str, depth: int):
            if depth > max_depth:
                return
                
            nodes.add(gene)
            
            # Get TF regulators
            regulators = self.get_tf_regulators(gene, top_k=10)
            
            for tf, score in regulators:
                if score >= min_score:
                    edges.append({
                        'source': tf,
                        'target': gene,
                        'weight': score,
                        'type': 'tf_binding'
                    })
                    
                    if tf not in nodes:
                        trace_upstream(tf, depth + 1)
                        
        trace_upstream(target_gene, 0)
        
        return {
            'target': target_gene,
            'nodes': list(nodes),
            'edges': edges,
            'n_nodes': len(nodes),
            'n_edges': len(edges)
        }
        
    def generate_pathway_analysis(
        self,
        regulators: Dict[str, Any],
        pathway_annotations: Optional[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        Analyze pathways associated with regulators.
        
        Args:
            regulators: Output from find_regulators
            pathway_annotations: TF pathway annotations
            
        Returns:
            Dict mapping pathway types to enriched pathways
        """
        if not pathway_annotations:
            return {}
            
        pathway_counts = {}
        
        # Count pathways from TF regulators
        for tf, score in regulators.get('tf_regulators', []):
            if tf in pathway_annotations:
                for pathway_type, pathways in pathway_annotations[tf].items():
                    if pathway_type not in pathway_counts:
                        pathway_counts[pathway_type] = {}
                    for pathway in pathways:
                        pathway_counts[pathway_type][pathway] = pathway_counts[pathway_type].get(pathway, 0) + score
                        
        # Sort and return top pathways
        result = {}
        for pathway_type, counts in pathway_counts.items():
            sorted_pathways = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            result[pathway_type] = [p[0] for p in sorted_pathways]
            
        return result


def generate_report(
    results: Dict[str, Any],
    output_dir: Path
):
    """Generate HTML report for analysis results."""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GRN Analysis: {results['target_gene']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .strategy {{ background: #ecf0f1; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .strategy.ko {{ border-left: 4px solid #e74c3c; }}
        .strategy.oe {{ border-left: 4px solid #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .score {{ font-weight: bold; color: #2980b9; }}
        .pathway {{ background: #e8f4f8; padding: 8px 12px; margin: 4px; display: inline-block; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gene Regulatory Network Analysis: {results['target_gene']}</h1>
        <p><strong>Objective:</strong> {results['desired_direction'].upper()} expression of {results['target_gene']}</p>
        
        <h2>Top TF Regulators (ChIP-Atlas)</h2>
        <table>
            <tr><th>Transcription Factor</th><th>Binding Score</th></tr>
"""
    
    for tf, score in results.get('tf_regulators', [])[:15]:
        html += f'            <tr><td>{tf}</td><td class="score">{score:.4f}</td></tr>\n'
        
    html += """
        </table>
        
        <h2>Knockdown Strategies</h2>
        <p>Genes to knock down to achieve desired effect:</p>
"""
    
    for gene, effect in results.get('ko_strategies', [])[:10]:
        html += f'        <div class="strategy ko"><strong>{gene}</strong>: Effect = {effect:.4f}</div>\n'
        
    html += """
        
        <h2>Overexpression Strategies</h2>
        <p>Genes to overexpress to achieve desired effect:</p>
"""
    
    for gene, effect in results.get('oe_strategies', [])[:10]:
        html += f'        <div class="strategy oe"><strong>{gene}</strong>: Effect = {effect:.4f}</div>\n'
        
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_dir / 'analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Report saved to {output_dir / 'analysis_report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Query GRN for target gene regulators')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--gene', type=str, required=True,
                        help='Target gene to analyze')
    parser.add_argument('--direction', type=str, default='increase',
                        choices=['increase', 'decrease'],
                        help='Desired direction of change')
    parser.add_argument('--tf-binding', type=str, default=None,
                        help='Path to TF binding directory')
    parser.add_argument('--output', type=str, default='results/analysis',
                        help='Output directory')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top regulators to return')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load gene names
    model_dir = Path(args.model).parent
    gene_names_file = model_dir / 'gene_names.csv'
    
    if gene_names_file.exists():
        gene_names = pd.read_csv(gene_names_file)['gene'].tolist()
    else:
        print("Warning: gene_names.csv not found, using dummy gene names")
        gene_names = [f"Gene_{i}" for i in range(12328)]
        
    # Load TF binding if provided
    tf_names = None
    tf_gene_binding = None
    pathway_annotations = None
    
    if args.tf_binding:
        tf_binding_dir = Path(args.tf_binding)
        
        if (tf_binding_dir / 'binding_matrix.npy').exists():
            tf_gene_binding = np.load(tf_binding_dir / 'binding_matrix.npy')
            
        if (tf_binding_dir / 'mappings.pkl').exists():
            with open(tf_binding_dir / 'mappings.pkl', 'rb') as f:
                mappings = pickle.load(f)
            tf_names = mappings.get('tf_names', [])
            
        if (tf_binding_dir / 'pathway_annotations.pkl').exists():
            with open(tf_binding_dir / 'pathway_annotations.pkl', 'rb') as f:
                pathway_annotations = pickle.load(f)
                
    # Create query engine
    engine = GRNQueryEngine(
        model_path=args.model,
        gene_names=gene_names,
        tf_names=tf_names,
        tf_gene_binding=tf_gene_binding,
        device=args.device
    )
    
    # Find regulators
    results = engine.find_regulators(
        target_gene=args.gene,
        desired_direction=args.direction,
        top_k=args.top_k
    )
    
    # Build regulatory network
    network = engine.build_regulatory_network(args.gene)
    results['network'] = network
    
    # Pathway analysis
    if pathway_annotations:
        pathways = engine.generate_pathway_analysis(results, pathway_annotations)
        results['pathways'] = pathways
        
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in results.items()
        }, f, indent=2, default=str)
        
    # Generate report
    generate_report(results, output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Analysis for {args.gene} ({args.direction})")
    print(f"{'='*60}")
    
    print(f"\nTop TF Regulators:")
    for tf, score in results.get('tf_regulators', [])[:10]:
        print(f"  {tf}: {score:.4f}")
        
    print(f"\nTop Knockdown Targets:")
    for gene, effect in results.get('ko_strategies', [])[:5]:
        print(f"  KO {gene}: effect = {effect:.4f}")
        
    print(f"\nTop Overexpression Targets:")
    for gene, effect in results.get('oe_strategies', [])[:5]:
        print(f"  OE {gene}: effect = {effect:.4f}")
        
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()


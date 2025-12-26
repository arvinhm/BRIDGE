#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: GENE-SPECIFIC FINE-TUNING WITH NEIGHBORHOOD ANALYSIS
================================================================================

USES EXACTLY the same architecture as train_v7.py:
- IntegratedGRNModelV7 with Gene→Gene, Gene→TF, TF→TF, TF→Gene
- OmniPath integration
- Cofactor gating
- Context gating (FiLM)
- Multi-TF mode

ADDS:
- Extra loss weight on target gene
- Freezes TF identity (embeddings, cascade, specificity weights)
- Only tunes gene-level effects
- Neighborhood analysis: Gene→TF→Target path scoring

Usage:
    python scripts/finetune_gene.py \
        --gene SSTR2 \
        --model results/sclc_v7/best_model.pt \
        --singlecell data/sclc/preprocessed.h5ad \
        --tf-binding data/tf_binding/ \
        --omnipath data/omnipath/ \
        --ppi data/ppi/ \
        --output results/SSTR2_finetuned/ \
        --epochs 50

Author: BRIDGE Team
"""

import os
import sys
import json
import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORT EVERYTHING FROM train_v7 - KEEP EXACT SAME CODE
# =============================================================================
from scripts.train_v7 import (
    # Data loading (from train_v6 via train_v7)
    load_singlecell_data,
    load_tf_binding,
    compute_tf_activity_linger,
    compute_baseline_metrics,
    
    # OmniPath loading
    load_omnipath,
    
    # Model - THE v7 ARCHITECTURE with Gene→Gene, Gene→TF
    IntegratedGRNModelV7,
    
    # Parent model from v6
    IntegratedGRNModel,
    
    # Dataset
    MultiTFDataset,
    
    # Loss function
    DeltaFocusedLoss,
    
    # Trainer - v7 trainer with prior penalty
    SignalingGRNTrainer,
    IntegratedGRNTrainer,
)


# =============================================================================
# GENE-WEIGHTED LOSS (Extends DeltaFocusedLoss)
# =============================================================================

class GeneWeightedLoss(DeltaFocusedLoss):
    """
    EXACT same loss as train_v6/v7, but with extra weight on target gene.
    
    This encourages the model to predict the target gene's delta more accurately.
    """
    
    def __init__(
        self, 
        target_gene_idx: int, 
        gene_weight: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_gene_idx = target_gene_idx
        self.gene_weight = gene_weight
        
    def forward(
        self, 
        pred_delta: torch.Tensor, 
        actual_delta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with extra weight on target gene.
        
        Args:
            pred_delta: (batch, n_genes) predicted expression changes
            actual_delta: (batch, n_genes) actual expression changes
            
        Returns:
            Dict with loss components including target gene metrics
        """
        # Get base loss from parent (EXACT same as train_v6/v7)
        losses = super().forward(pred_delta, actual_delta)
        
        # Extract target gene predictions
        target_pred = pred_delta[:, self.target_gene_idx]
        target_actual = actual_delta[:, self.target_gene_idx]
        
        # Target gene MSE (weighted extra)
        target_mse = F.mse_loss(target_pred, target_actual)
        
        # Target gene correlation - numerically stable formula
        pred_c = target_pred - target_pred.mean()
        actual_c = target_actual - target_actual.mean()
        
        correlation = (pred_c * actual_c).mean() / (pred_c.std() * actual_c.std() + 1e-8)
        correlation = torch.clamp(correlation, -1, 1)
        
        # Target gene direction accuracy
        sign_match = ((target_pred > 0) == (target_actual > 0)).float().mean()
        
        # Add weighted target loss to total
        losses['total'] = losses['total'] + self.gene_weight * target_mse
        losses['target_mse'] = target_mse
        losses['target_corr'] = correlation
        losses['target_sign'] = sign_match
        
        return losses


# =============================================================================
# GENE-SPECIFIC TRAINER (Extends SignalingGRNTrainer from v7)
# =============================================================================

class GeneSpecificTrainer(SignalingGRNTrainer):
    """
    Trainer for gene-specific fine-tuning.
    
    INHERITS ALL functionality from SignalingGRNTrainer (train_v7):
    - train_epoch_multi_tf with prior penalty
    - validate_multi_tf
    - compute_prior_penalty
    
    ADDS:
    - Gene-weighted loss
    - Extra tracking for target gene metrics
    """
    
    def __init__(
        self,
        model: IntegratedGRNModelV7,
        device: torch.device,
        target_gene_idx: int,
        gene_weight: float = 10.0,
        learning_rate: float = 1e-4,
        use_amp: bool = True,
        use_chip_reg: bool = True,
        prior_penalty_weight: float = 0.01
    ):
        # Call parent constructor (SignalingGRNTrainer -> IntegratedGRNTrainer)
        # Note: SignalingGRNTrainer stores prior_penalty_weight itself
        super().__init__(
            model=model,
            device=device,
            learning_rate=learning_rate,
            use_amp=use_amp,
            use_chip_reg=use_chip_reg,
            prior_penalty_weight=prior_penalty_weight
        )
        
        self.target_gene_idx = target_gene_idx
        self.gene_weight = gene_weight
        
        # Replace loss function with gene-weighted version
        self.loss_fn = GeneWeightedLoss(
            target_gene_idx=target_gene_idx,
            gene_weight=gene_weight,
            use_chip_reg=use_chip_reg
        )
    
    def train_epoch_multi_tf(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Training epoch for gene-specific fine-tuning.
        
        SAME as parent but tracks target gene metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_prior = 0.0
        total_target_corr = 0.0
        total_target_sign = 0.0
        all_pred_deltas = []
        all_actual_deltas = []
        all_target_preds = []
        all_target_actuals = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}")
        
        for batch in pbar:
            # Correct keys from MultiTFDataset (train_v6/v7)
            tf_change = batch['tf_activity_change'].to(self.device)
            source_expr = batch['source_expression'].to(self.device)
            actual_delta = batch['delta_expression'].to(self.device)
            
            self.optimizer.zero_grad()
            
            from torch.cuda.amp import autocast
            with autocast(enabled=self.use_amp):
                pred_delta, grn_matrix = self.model.predict_delta_multi_tf(
                    tf_change, source_expr
                )
                
                losses = self.loss_fn(pred_delta, actual_delta)
                
                # Add L2 prior penalty (inherited from SignalingGRNTrainer)
                prior_penalty = self.compute_prior_penalty()
                total_loss_batch = losses['total'] + prior_penalty
            
            if self.use_amp:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_prior += prior_penalty.item()
            total_target_corr += losses['target_corr'].item()
            total_target_sign += losses['target_sign'].item()
            
            # Collect for metrics
            if len(all_pred_deltas) < 50:
                all_pred_deltas.append(pred_delta.detach().cpu())
                all_actual_deltas.append(actual_delta.detach().cpu())
            
            # Always collect target gene for accurate metrics
            all_target_preds.append(pred_delta[:, self.target_gene_idx].detach().cpu())
            all_target_actuals.append(actual_delta[:, self.target_gene_idx].detach().cpu())
            
            pbar.set_postfix({
                'loss': f'{losses["total"].item():.4f}',
                'target_r': f'{losses["target_corr"].item():.3f}',
                'sign': f'{losses["target_sign"].item():.1%}'
            })
        
        # Compute metrics
        n_batches = len(dataloader)
        
        # Full delta correlation
        all_pred = torch.cat(all_pred_deltas, dim=0)
        all_actual = torch.cat(all_actual_deltas, dim=0)
        delta_pcc = self._batch_correlation(all_pred, all_actual)
        
        # Target gene correlation (using ALL batches)
        all_target_p = torch.cat(all_target_preds)
        all_target_a = torch.cat(all_target_actuals)
        target_pcc = scipy_stats.pearsonr(
            all_target_p.numpy(), 
            all_target_a.numpy()
        )[0] if len(all_target_p) > 10 else 0.0
        
        return {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches,
            'prior_penalty': total_prior / n_batches,
            'delta_pcc': delta_pcc,
            'target_pcc': target_pcc,
            'target_sign': total_target_sign / n_batches
        }
    
    def validate_multi_tf(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validation with target gene metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_pred_deltas = []
        all_actual_deltas = []
        all_target_preds = []
        all_target_actuals = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="  [VAL]", leave=True)
            
            for batch in pbar:
                tf_change = batch['tf_activity_change'].to(self.device)
                source_expr = batch['source_expression'].to(self.device)
                actual_delta = batch['delta_expression'].to(self.device)
                
                from torch.cuda.amp import autocast
                with autocast(enabled=self.use_amp):
                    pred_delta, _ = self.model.predict_delta_multi_tf(
                        tf_change, source_expr
                    )
                    losses = self.loss_fn(pred_delta, actual_delta)
                
                total_loss += losses['total'].item()
                all_pred_deltas.append(pred_delta.cpu())
                all_actual_deltas.append(actual_delta.cpu())
                all_target_preds.append(pred_delta[:, self.target_gene_idx].cpu())
                all_target_actuals.append(actual_delta[:, self.target_gene_idx].cpu())
                
                pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})
        
        # Compute metrics
        all_pred = torch.cat(all_pred_deltas, dim=0)
        all_actual = torch.cat(all_actual_deltas, dim=0)
        
        delta_pcc = self._batch_correlation(all_pred, all_actual)
        dir_acc = self._compute_direction_accuracy(all_pred, all_actual)
        
        # Target gene metrics
        all_target_p = torch.cat(all_target_preds)
        all_target_a = torch.cat(all_target_actuals)
        target_pcc = scipy_stats.pearsonr(
            all_target_p.numpy(),
            all_target_a.numpy()
        )[0] if len(all_target_p) > 10 else 0.0
        
        target_dir = ((all_target_p > 0) == (all_target_a > 0)).float().mean().item()
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_delta_pcc': delta_pcc,
            'val_dir_acc': dir_acc,
            'val_target_pcc': target_pcc,
            'val_target_dir': target_dir
        }


# =============================================================================
# NEIGHBORHOOD ANALYSIS
# =============================================================================

def run_neighborhood_analysis(
    model: IntegratedGRNModelV7,
    target_gene: str,
    target_gene_idx: int,
    gene_names: np.ndarray,
    tf_names: np.ndarray,
    expression: np.ndarray,
    output_dir: Path,
    top_k: int = 20
):
    """
    Compute neighborhood analysis: Gene→TF→Target path scores and sensitivity.
    
    Produces:
    1. CSV of top Gene→TF→Target path contributors
    2. CSV of local gene sensitivities (autograd)
    3. Network visualization showing which genes (via which TFs) affect target
    """
    print("\n" + "=" * 70)
    print(f"NEIGHBORHOOD ANALYSIS FOR {target_gene}")
    print("=" * 70)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get GRN matrix
    with torch.no_grad():
        grn = model.get_grn_matrix().cpu().numpy()  # (n_genes, n_tfs)
    
    print(f"  GRN matrix: {grn.shape}")
    
    # Get Gene→TF modulation matrix if available
    gene_to_tf = None
    if getattr(model, 'use_gene_tf_modulation', False):
        with torch.no_grad():
            # Combine learned and prior: gene_to_tf is (n_tfs, n_genes)
            g2t_learned = model.gene_to_tf.cpu().numpy()
            g2t_prior = model.gene_tf_prior.T.cpu().numpy()
            gene_to_tf = (g2t_learned + 0.5 * g2t_prior).T  # -> (n_genes, n_tfs)
        print(f"  Gene→TF matrix: {gene_to_tf.shape}")
    
    # =========================================================================
    # 1. Path Scores: Gene→TF→Target
    # =========================================================================
    print("\n  [1/3] Computing Gene→TF→Target path scores...")
    
    if gene_to_tf is not None:
        # TF effects on target gene
        tf_target = grn[target_gene_idx, :]  # (n_tfs,) - signed effects
        tf_target_abs = np.abs(tf_target)
        
        # Path scores: how much does each gene affect target via each TF
        # path_score[g, t] = |gene_to_tf[g, t]| × |grn[target, t]|
        path_scores = np.abs(gene_to_tf) * tf_target_abs[None, :]  # (n_genes, n_tfs)
        
        # Aggregate: total path score per gene
        gene_path_scores = np.nansum(path_scores, axis=1)  # (n_genes,)
        
        # Top genes by path score
        top_gene_idx = np.argsort(gene_path_scores)[-top_k:][::-1]
        top_tfs_idx = np.argsort(tf_target_abs)[-top_k:][::-1]
        
        # Save top genes
        path_df = pd.DataFrame({
            'gene': gene_names[top_gene_idx],
            'path_score': gene_path_scores[top_gene_idx],
            'gene_idx': top_gene_idx
        })
        path_df.to_csv(output_dir / f'{target_gene}_top_gene_path_scores.csv', index=False)
        print(f"    Saved: {target_gene}_top_gene_path_scores.csv")
        
        # Save top TFs affecting target
        tf_df = pd.DataFrame({
            'tf': tf_names[top_tfs_idx],
            'effect_on_target': tf_target[top_tfs_idx],
            'tf_idx': top_tfs_idx
        })
        tf_df.to_csv(output_dir / f'{target_gene}_top_tf_effects.csv', index=False)
        print(f"    Saved: {target_gene}_top_tf_effects.csv")
    else:
        print("    Gene→TF modulation not available, skipping path analysis")
        top_gene_idx = np.arange(min(top_k, len(gene_names)))
        top_tfs_idx = np.argsort(np.abs(grn[target_gene_idx, :]))[-top_k:][::-1]
        path_scores = None
    
    # =========================================================================
    # 2. Local Sensitivity via Autograd
    # =========================================================================
    print("\n  [2/3] Computing gene sensitivity via autograd...")
    
    # Representative TF change: mean abs across dataset
    tf_change_ref = torch.ones(1, grn.shape[1], device=device)
    
    # Representative source expression: mean across cell types
    source_expr_ref = torch.from_numpy(expression.mean(axis=0)).float().unsqueeze(0).to(device)
    source_expr_ref.requires_grad_(True)
    
    # Forward pass with gradient tracking
    model.eval()
    pred_delta, _ = model.predict_delta_multi_tf(tf_change_ref, source_expr_ref)
    target_delta = pred_delta[0, target_gene_idx]
    
    # Backward to get sensitivity
    target_delta.backward()
    gene_sensitivity = source_expr_ref.grad.detach().cpu().numpy().ravel()
    
    # Save sensitivity
    sens_df = pd.DataFrame({
        'gene': gene_names,
        'sensitivity': gene_sensitivity
    }).sort_values('sensitivity', key=abs, ascending=False)
    sens_df.to_csv(output_dir / f'{target_gene}_gene_sensitivity.csv', index=False)
    print(f"    Saved: {target_gene}_gene_sensitivity.csv")
    
    # =========================================================================
    # 3. Neighborhood Graph Visualization
    # =========================================================================
    print("\n  [3/3] Creating neighborhood graph...")
    
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Target node
        G.add_node(target_gene, kind='target')
        
        # Add top gene nodes
        for gi in top_gene_idx[:top_k]:
            g_name = gene_names[gi]
            if g_name != target_gene:
                G.add_node(g_name, kind='gene')
        
        # Add top TF nodes
        for ti in top_tfs_idx[:top_k]:
            G.add_node(tf_names[ti], kind='tf')
        
        # Add edges: Gene→TF (weighted by path contribution)
        if path_scores is not None:
            # Threshold: keep only top edges
            threshold = np.percentile(np.abs(path_scores[path_scores != 0]), 95)
            
            for gi in top_gene_idx[:top_k]:
                g_name = gene_names[gi]
                if g_name == target_gene:
                    continue
                for ti in top_tfs_idx[:top_k]:
                    t_name = tf_names[ti]
                    w = gene_to_tf[gi, ti]
                    if abs(w) > threshold * 0.1:  # Keep stronger edges
                        G.add_edge(g_name, t_name, weight=w, role='gene->tf')
        
        # Add edges: TF→Target
        for ti in top_tfs_idx[:top_k]:
            t_name = tf_names[ti]
            w = grn[target_gene_idx, ti]
            if abs(w) > 0.001:  # Non-trivial effect
                G.add_edge(t_name, target_gene, weight=w, role='tf->target')
        
        # Layout
        pos = nx.kamada_kawai_layout(G)
        
        # Edge styling
        edge_colors = []
        edge_widths = []
        max_weight = max(abs(d.get('weight', 0.1)) for _, _, d in G.edges(data=True)) + 1e-8
        for u, v, d in G.edges(data=True):
            if d.get('role') == 'gene->tf':
                edge_colors.append('#2ca02c' if d.get('weight', 0) > 0 else '#d62728')
            else:
                edge_colors.append('#1f77b4' if d.get('weight', 0) > 0 else '#ff7f0e')
            edge_widths.append(max(0.5, 3.0 * abs(d.get('weight', 0.1)) / max_weight))
        
        # Node styling
        node_colors = []
        for n, d in G.nodes(data=True):
            kind = d.get('kind', 'gene')
            if kind == 'target':
                node_colors.append('#fb8072')
            elif kind == 'tf':
                node_colors.append('#bebada')
            else:
                node_colors.append('#8dd3c7')
        
        # Draw
        plt.figure(figsize=(14, 10))
        nx.draw_networkx(
            G, pos,
            with_labels=True,
            node_size=800,
            node_color=node_colors,
            arrows=True,
            width=edge_widths,
            edge_color=edge_colors,
            font_size=8,
            font_weight='bold',
            arrowsize=15
        )
        
        plt.title(f'{target_gene} Neighborhood: Gene→TF→Target Routes (post-finetune)', fontsize=14)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#fb8072', label='Target Gene'),
            Patch(facecolor='#bebada', label='TFs'),
            Patch(facecolor='#8dd3c7', label='Upstream Genes'),
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f'{target_gene}_neighborhood.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {target_gene}_neighborhood.png")
        print(f"    Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
    except ImportError as e:
        print(f"    Skipping graph visualization (missing dependency: {e})")
    
    print("\n  Neighborhood analysis complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BRIDGE 2.0: Gene-Specific Fine-Tuning (uses train_v7 architecture)'
    )
    
    # Required arguments
    parser.add_argument('--gene', type=str, required=True,
                        help='Target gene symbol (e.g., SSTR2)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained v7 model checkpoint')
    parser.add_argument('--singlecell', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--tf-binding', type=str, required=True,
                        help='TF binding directory')
    
    # OmniPath (SAME as train_v7)
    parser.add_argument('--omnipath', type=str, default=None,
                        help='OmniPath directory (optional but recommended)')
    parser.add_argument('--ppi', type=str, default=None,
                        help='PPI directory')
    
    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    
    # Training params
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gene-weight', type=float, default=10.0,
                        help='Extra weight on target gene loss')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--val-every', type=int, default=1)
    
    # OmniPath flags (SAME as train_v7)
    parser.add_argument('--no-gene-propagation', action='store_true',
                        help='Disable gene→gene propagation')
    parser.add_argument('--no-gene-tf-modulation', action='store_true',
                        help='Disable gene→TF modulation')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_gene = args.gene.upper()
    
    print("=" * 70)
    print(f"BRIDGE 2.0: GENE-SPECIFIC FINE-TUNING FOR {target_gene}")
    print("(Uses EXACT same architecture as train_v7)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Gene weight: {args.gene_weight}x")
    
    # =========================================================================
    # [1/7] Load Single-Cell Data (SAME AS train_v7)
    # =========================================================================
    print("\n[1/7] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(
        args.singlecell, args.celltype_key
    )
    n_celltypes, n_genes = expression.shape
    print(f"  Expression: {expression.shape}")
    
    # Find target gene
    gene_names_upper = [g.upper() for g in gene_names]
    if target_gene not in gene_names_upper:
        print(f"  Available genes containing '{target_gene[:4]}': ", end="")
        matches = [g for g in gene_names if target_gene[:4].upper() in g.upper()]
        print(matches[:10])
        raise ValueError(f"Target gene {target_gene} not found!")
    
    target_gene_idx = gene_names_upper.index(target_gene)
    print(f"  Target gene: {target_gene} at index {target_gene_idx}")
    print(f"  Expression range: [{expression[:, target_gene_idx].min():.3f}, {expression[:, target_gene_idx].max():.3f}]")
    
    # =========================================================================
    # [2/7] Load TF Binding (SAME AS train_v7)
    # =========================================================================
    print("\n[2/7] Loading TF binding data...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(args.tf_binding, gene_names)
    n_tfs = len(tf_names)
    print(f"  Aligned {binding_matrix.shape[0]}/{n_genes} genes")
    print(f"  TFs: {n_tfs}")
    print(f"  Binding: {binding_matrix.shape}")
    
    # =========================================================================
    # [3/7] Compute TF Activity (SAME AS train_v7)
    # =========================================================================
    print("\n[3/7] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    print(f"  TF activity: {tf_activity.shape}")
    
    # =========================================================================
    # [4/7] Load OmniPath (SAME AS train_v7)
    # =========================================================================
    print("\n[4/7] Loading OmniPath signaling network...")
    if args.omnipath:
        # load_omnipath signature: (omnipath_dir, n_genes, n_tfs, gene_names, tf_names)
        gene_gene, gene_tf, tf_tf_omnipath = load_omnipath(
            args.omnipath, len(gene_names), len(tf_names),
            gene_names=gene_names, tf_names=tf_names
        )
        # Blend TF-TF from ChIP and OmniPath
        if tf_tf_omnipath is not None:
            tf_tf_combined = tf_tf_adj + 0.3 * tf_tf_omnipath
        else:
            tf_tf_combined = tf_tf_adj
        has_omnipath = True
    else:
        gene_gene = np.zeros((len(gene_names), len(gene_names)), dtype=np.float32)
        gene_tf = np.zeros((len(gene_names), len(tf_names)), dtype=np.float32)
        tf_tf_combined = tf_tf_adj
        has_omnipath = False
        print("  No OmniPath data provided")
    
    # =========================================================================
    # [5/7] Create Datasets (SAME AS train_v7)
    # =========================================================================
    print("\n[5/7] Creating datasets...")
    
    # Cell-type pair split (hold out some for validation)
    n_holdout = max(2, n_celltypes // 6)
    np.random.seed(42)
    holdout = set(np.random.choice(n_celltypes, n_holdout, replace=False))
    train_idx = [i for i in range(n_celltypes) if i not in holdout]
    
    train_pairs = [(i, j) for i in train_idx for j in train_idx if i != j]
    val_pairs = [(i, j) for i in holdout for j in holdout if i != j]
    
    print(f"  Train cell-type pairs: {len(train_pairs)}")
    print(f"  Val cell-type pairs: {len(val_pairs)}")
    
    train_dataset = MultiTFDataset(
        expression_means=expression,
        tf_activity=tf_activity,
        celltype_names=list(celltypes),
        allowed_pairs=train_pairs
    )
    val_dataset = MultiTFDataset(
        expression_means=expression,
        tf_activity=tf_activity,
        celltype_names=list(celltypes),
        allowed_pairs=val_pairs
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_dataset) > args.batch_size
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # =========================================================================
    # [6/7] Load Pre-trained Model (SAME ARCHITECTURE AS train_v7)
    # =========================================================================
    print("\n[6/7] Loading pre-trained model...")
    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        loaded_epoch = checkpoint.get('epoch', 'unknown')
        loaded_delta = checkpoint.get('best_val_delta_pcc', 'unknown')
        print(f"  Loaded checkpoint: epoch={loaded_epoch}, delta_pcc={loaded_delta}")
    else:
        state_dict = checkpoint
        print("  Loaded raw state dict")
    
    # Create model (EXACT SAME as train_v7)
    model = IntegratedGRNModelV7(
        n_genes=n_genes,
        n_tfs=n_tfs,
        hidden_dim=args.hidden_dim,
        use_gene_propagation=has_omnipath and not args.no_gene_propagation,
        use_gene_tf_modulation=has_omnipath and not args.no_gene_tf_modulation
    )
    
    # Load PPI cofactor data (SAME as train_v7)
    cof_to_tf = None
    cof_indices = None
    if args.ppi:
        ppi_dir = Path(args.ppi)
        if (ppi_dir / 'cof_to_tf_adj.npy').exists():
            cof_to_tf = np.load(ppi_dir / 'cof_to_tf_adj.npy')
            cof_indices = np.load(ppi_dir / 'cofactor_gene_indices.npy')
    
    # Set all priors (SAME as train_v7)
    model.set_priors(
        chip_binding=torch.from_numpy(binding_matrix),
        tf_tf_adjacency=torch.from_numpy(tf_tf_combined),
        cofactor_tf_adjacency=torch.from_numpy(cof_to_tf) if cof_to_tf is not None else None,
        cofactor_gene_indices=torch.from_numpy(cof_indices) if cof_indices is not None else None
    )
    
    # Set OmniPath priors (SAME as train_v7)
    if has_omnipath:
        model.set_omnipath_priors(
            gene_gene=torch.from_numpy(gene_gene),
            gene_tf=torch.from_numpy(gene_tf)
        )
    
    # Load pre-trained weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for new components)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # =========================================================================
    # FREEZE TF IDENTITY, ONLY TUNE GENE-LEVEL EFFECTS
    # =========================================================================
    print("\n  Configuring fine-tuning...")
    
    # Freeze TF-level components (preserve learned TF relationships)
    finetune_params = []
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # FREEZE: TF embeddings, cascade, specificity weights, and global network params
        # These represent learned TF relationships we want to preserve
        if any(x in name for x in [
            'tf_embeddings', 'tf_cascade', 'tf_specificity_weights',
            'gene_propagation', 'gene_to_tf', 'gene_prop_weight',
            'gene_tf_modulation_strength', 'gene_gene_prior', 'gene_tf_prior'
        ]):
            param.requires_grad = False
            frozen_count += 1
        # FINE-TUNE: Effect decoder (TF→gene), direct effects, context gating
        elif any(x in name for x in [
            'effect_decoder', 'direct_effects', 'context',
            'chip_mask', '_leakage_raw'
        ]):
            param.requires_grad = True
            finetune_params.append(param)
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"  Fine-tuning: {trainable_count} groups, {sum(p.numel() for p in finetune_params):,} params")
    print(f"  Frozen: {frozen_count} groups")
    
    # =========================================================================
    # [7/7] Create Trainer and Train
    # =========================================================================
    print("\n[7/7] Setting up training...")
    
    trainer = GeneSpecificTrainer(
        model=model,
        device=device,
        target_gene_idx=target_gene_idx,
        gene_weight=args.gene_weight,
        learning_rate=args.lr,
        use_amp=True,
        prior_penalty_weight=0.01
    )
    
    # Replace optimizer with one that only updates finetune_params
    trainer.optimizer = torch.optim.AdamW(finetune_params, lr=args.lr, weight_decay=0.01)
    
    print(f"  Gene propagation: {'ON' if model.use_gene_propagation else 'OFF'}")
    print(f"  Gene→TF modulation: {'ON' if model.use_gene_tf_modulation else 'OFF'}")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"FINE-TUNING FOR {target_gene}")
    print("=" * 70)
    
    best_target_pcc = -1.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = trainer.train_epoch_multi_tf(train_loader, epoch)
        
        print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} "
              f"| {target_gene}_r={train_metrics['target_pcc']:.4f} "
              f"| sign={train_metrics['target_sign']:.1%}")
        
        # Validate
        if epoch % args.val_every == 0:
            val_metrics = trainer.validate_multi_tf(val_loader)
            
            print(f"  [VAL] Delta_PCC={val_metrics['val_delta_pcc']:.4f} "
                  f"| {target_gene}_r={val_metrics['val_target_pcc']:.4f} "
                  f"| Dir={val_metrics['val_target_dir']:.1%}")
            
            # Save best based on target gene PCC
            if val_metrics['val_target_pcc'] > best_target_pcc:
                best_target_pcc = val_metrics['val_target_pcc']
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'target_gene': target_gene,
                    'target_gene_idx': target_gene_idx,
                    'best_target_pcc': best_target_pcc,
                    'val_delta_pcc': val_metrics['val_delta_pcc'],
                    'config': model.config
                }, output_dir / 'best_model.pt')
                
                print(f"  >> New best! {target_gene}_PCC={best_target_pcc:.4f}")
            
            # Record history
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_target_pcc': train_metrics['target_pcc'],
                'train_target_sign': train_metrics['target_sign'],
                'val_delta_pcc': val_metrics['val_delta_pcc'],
                'val_target_pcc': val_metrics['val_target_pcc'],
                'val_target_dir': val_metrics['val_target_dir']
            })
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save GRN matrix
    grn_matrix = model.get_grn_matrix().detach().cpu().numpy()
    np.save(output_dir / 'grn_matrix.npy', grn_matrix)
    print(f"  Saved: grn_matrix.npy {grn_matrix.shape}")
    
    # Save signaling networks (v7 specific)
    if hasattr(model, 'get_signaling_networks'):
        networks = model.get_signaling_networks()
        for name, net in networks.items():
            if net is not None:
                np.save(output_dir / f'{name}.npy', net.cpu().numpy())
                print(f"  Saved: {name}.npy")
    
    # Save metadata
    gene_names_arr = np.array(gene_names)
    tf_names_arr = np.array(tf_names)
    
    pd.DataFrame({'gene': gene_names}).to_csv(
        output_dir / 'gene_names.csv', index=False, header=False
    )
    pd.DataFrame({'tf': tf_names}).to_csv(
        output_dir / 'tf_names.csv', index=False, header=False
    )
    pd.DataFrame({'celltype': celltypes}).to_csv(
        output_dir / 'celltype_names.csv', index=False, header=False
    )
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    
    # Save history
    pd.DataFrame(history).to_csv(output_dir / 'finetune_history.csv', index=False)
    
    # Save config
    config = {
        'target_gene': target_gene,
        'target_gene_idx': target_gene_idx,
        'gene_weight': args.gene_weight,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_target_pcc': float(best_target_pcc),
        'n_genes': n_genes,
        'n_tfs': n_tfs,
        'has_omnipath': has_omnipath,
        'model_version': 'v7'
    }
    with open(output_dir / 'finetune_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nBest: Epoch {best_epoch} with {target_gene}_PCC={best_target_pcc:.4f}")
    print(f"Saved to: {output_dir}")
    
    # =========================================================================
    # NEIGHBORHOOD ANALYSIS
    # =========================================================================
    run_neighborhood_analysis(
        model=model,
        target_gene=target_gene,
        target_gene_idx=target_gene_idx,
        gene_names=gene_names_arr,
        tf_names=tf_names_arr,
        expression=expression,
        output_dir=output_dir
    )
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print(f"\nTo analyze:")
    print(f"python scripts/analyze_target_full.py \\")
    print(f"    --gene {target_gene} \\")
    print(f"    --model {output_dir} \\")
    print(f"    --singlecell {args.singlecell} \\")
    print(f"    --ppi {args.ppi if args.ppi else 'data/ppi'} \\")
    print(f"    --celltype-key {args.celltype_key} \\")
    print(f"    --output analysis/{target_gene}_finetuned/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: COMPLETE MOLECULAR SIGNALING NETWORK TRAINING
================================================================================

EXTENSION of train_v6.py - Uses EXACTLY the same IntegratedGRNModel architecture
with ADDITIONAL dimensions for:
- Gene→Gene signaling (OmniPath)
- Gene→TF regulation (OmniPath) 

All components from train_v6 are PRESERVED:
- IntegratedGRNModel with TFRegulatoryEncoder, SingleCellDataEncoder
- Cell-type PAIR holdout validation
- DELTA prediction focus
- JASPAR/PPI integration
- Multi-TF training mode
- Cofactor gating

NEW ADDITIONS:
- Gene→Gene propagation network (learned from OmniPath prior)
- Gene→TF modulation (upstream regulation)
- Two-phase training (known network → discovery)

Usage:
    python scripts/train_v7.py \
        --singlecell data/sclc/preprocessed.h5ad \
        --tf-binding data/tf_binding/ \
        --omnipath data/omnipath/ \
        --ppi data/ppi/ \
        --output results/sclc_v7/ \
        --epochs 100

Author: BRIDGE Team
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler
from tqdm import tqdm
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# IMPORT EVERYTHING FROM train_v6 - KEEP EXACT SAME CODE
# ============================================================================
from scripts.train_v6 import (
    # Data loading
    load_singlecell_data,
    load_tf_binding,
    compute_tf_activity_linger,
    compute_baseline_metrics,
    
    # Model - THE CORE ARCHITECTURE
    IntegratedGRNModel,
    
    # Dataset
    MultiTFDataset,
    
    # Loss function
    DeltaFocusedLoss,
    
    # Trainer - USE EXACT SAME TRAINER
    IntegratedGRNTrainer,
)

# Import sophisticated model components
from models.monster_grn import MonsterGRNModel, create_monster_grn
from models.tf_encoder import TFRegulatoryEncoder, TFCascadeGCN
from models.cell_encoder import CellStateEncoder, SingleCellDataEncoder
from modules.perturbation import LINGERStyleGRN, PerturbationPredictor


# =============================================================================
# OMNIPATH LOADING
# =============================================================================

def load_omnipath(omnipath_dir: str, n_genes: int, n_tfs: int, 
                   gene_names: List[str] = None, tf_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load OmniPath signaling matrices and align to expression gene order.
    
    The OmniPath matrices may be built from a subset of genes (mappings.pkl).
    This function expands them to match the full expression gene list.
    """
    opath = Path(omnipath_dir)
    
    # Try to load OmniPath gene order (from preprocessing)
    omnipath_genes = None
    mapping_path = opath / 'uniprot_to_gene_mapping.csv'
    if mapping_path.exists() and gene_names is not None:
        # Load the gene list used during OmniPath preprocessing
        # Try to find the original gene list from mappings.pkl
        mappings_pkl = opath.parent / 'tf_binding' / 'mappings.pkl'
        if mappings_pkl.exists():
            import pickle
            with open(mappings_pkl, 'rb') as f:
                mappings = pickle.load(f)
            omnipath_genes = mappings.get('gene_names', None)
    
    # Gene→Gene signaling
    gene_gene_path = opath / 'gene_gene_signaling.npy'
    if gene_gene_path.exists():
        gene_gene_raw = np.load(gene_gene_path)
        print(f"  Gene→Gene: {gene_gene_raw.shape}, {np.count_nonzero(gene_gene_raw):,} edges")
        
        # Expand to full gene list if needed
        if gene_gene_raw.shape[0] != n_genes and omnipath_genes is not None and gene_names is not None:
            gene_gene = _expand_gene_matrix(gene_gene_raw, omnipath_genes, gene_names)
            print(f"    Expanded to {gene_gene.shape}")
        elif gene_gene_raw.shape[0] != n_genes:
            # Pad with zeros
            gene_gene = np.zeros((n_genes, n_genes), dtype=np.float32)
            min_g = min(gene_gene_raw.shape[0], n_genes)
            gene_gene[:min_g, :min_g] = gene_gene_raw[:min_g, :min_g]
            print(f"    Padded to {gene_gene.shape}")
        else:
            gene_gene = gene_gene_raw
    else:
        gene_gene = np.zeros((n_genes, n_genes), dtype=np.float32)
        print(f"  Gene→Gene: not found, using zeros")
    
    # Gene→TF signaling
    gene_tf_path = opath / 'gene_tf_signaling.npy'
    if gene_tf_path.exists():
        gene_tf_raw = np.load(gene_tf_path)
        print(f"  Gene→TF: {gene_tf_raw.shape}, {np.count_nonzero(gene_tf_raw):,} edges")
        
        # Expand to full gene list if needed
        if gene_tf_raw.shape[0] != n_genes and omnipath_genes is not None and gene_names is not None:
            gene_tf = _expand_gene_tf_matrix(gene_tf_raw, omnipath_genes, gene_names, n_tfs)
            print(f"    Expanded to {gene_tf.shape}")
        elif gene_tf_raw.shape[0] != n_genes:
            # Pad with zeros
            gene_tf = np.zeros((n_genes, n_tfs), dtype=np.float32)
            min_g = min(gene_tf_raw.shape[0], n_genes)
            min_t = min(gene_tf_raw.shape[1], n_tfs)
            gene_tf[:min_g, :min_t] = gene_tf_raw[:min_g, :min_t]
            print(f"    Padded to {gene_tf.shape}")
        else:
            gene_tf = gene_tf_raw
    else:
        gene_tf = np.zeros((n_genes, n_tfs), dtype=np.float32)
        print(f"  Gene→TF: not found, using zeros")
    
    # TF→TF from OmniPath (to blend with ChIP-Atlas)
    tf_tf_path = opath / 'tf_tf_signaling_omnipath.npy'
    if tf_tf_path.exists():
        tf_tf_omnipath = np.load(tf_tf_path)
        print(f"  TF→TF (OmniPath): {tf_tf_omnipath.shape}, {np.count_nonzero(tf_tf_omnipath):,} edges")
    else:
        tf_tf_omnipath = None
    
    return gene_gene, gene_tf, tf_tf_omnipath


def _expand_gene_matrix(small_matrix: np.ndarray, small_genes: List[str], 
                        full_genes: List[str]) -> np.ndarray:
    """Expand a gene×gene matrix from small gene set to full gene set."""
    n_full = len(full_genes)
    expanded = np.zeros((n_full, n_full), dtype=np.float32)
    
    # Build mapping: small index → full index
    full_genes_upper = [g.upper() for g in full_genes]
    full_gene_to_idx = {g: i for i, g in enumerate(full_genes_upper)}
    
    for small_i, gene_i in enumerate(small_genes):
        if gene_i.upper() in full_gene_to_idx:
            full_i = full_gene_to_idx[gene_i.upper()]
            for small_j, gene_j in enumerate(small_genes):
                if gene_j.upper() in full_gene_to_idx:
                    full_j = full_gene_to_idx[gene_j.upper()]
                    expanded[full_i, full_j] = small_matrix[small_i, small_j]
    
    return expanded


def _expand_gene_tf_matrix(small_matrix: np.ndarray, small_genes: List[str],
                           full_genes: List[str], n_tfs: int) -> np.ndarray:
    """Expand a gene×TF matrix from small gene set to full gene set."""
    n_full = len(full_genes)
    expanded = np.zeros((n_full, n_tfs), dtype=np.float32)
    
    # Build mapping: small index → full index
    full_genes_upper = [g.upper() for g in full_genes]
    full_gene_to_idx = {g: i for i, g in enumerate(full_genes_upper)}
    
    for small_i, gene_i in enumerate(small_genes):
        if gene_i.upper() in full_gene_to_idx:
            full_i = full_gene_to_idx[gene_i.upper()]
            expanded[full_i, :] = small_matrix[small_i, :min(small_matrix.shape[1], n_tfs)]
    
    return expanded


# =============================================================================
# EXTENDED MODEL - SAME AS train_v6 IntegratedGRNModel + OmniPath
# =============================================================================

class IntegratedGRNModelV7(IntegratedGRNModel):
    """
    Extended IntegratedGRNModel with OmniPath Gene→Gene and Gene→TF networks.
    
    Inherits ALL functionality from IntegratedGRNModel (train_v6), adds:
    - Gene→Gene propagation (signaling cascades)
    - Gene→TF modulation (upstream regulation)
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        n_batches: int = 1,
        hidden_dim: int = 512,
        num_gcn_layers: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_context_gating: bool = True,
        use_gene_propagation: bool = True,  # NEW
        use_gene_tf_modulation: bool = True  # NEW
    ):
        # Initialize parent class (IntegratedGRNModel from train_v6)
        super().__init__(
            n_genes=n_genes,
            n_tfs=n_tfs,
            n_batches=n_batches,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_context_gating=use_context_gating
        )
        
        self.use_gene_propagation = use_gene_propagation
        self.use_gene_tf_modulation = use_gene_tf_modulation
        
        # ============================================================
        # NEW Component: Gene→Gene Propagation Network
        # ============================================================
        if use_gene_propagation:
            # Learnable gene propagation weights (initialized from OmniPath)
            self.gene_propagation = nn.Parameter(torch.zeros(n_genes, n_genes))
            
            # Prior from OmniPath (fixed, used for regularization)
            self.register_buffer('gene_gene_prior', torch.zeros(n_genes, n_genes))
            
            # Propagation steps and weight
            self.gene_prop_steps = 2
            self.gene_prop_weight = nn.Parameter(torch.tensor(0.3))
        
        # ============================================================
        # NEW Component: Gene→TF Modulation Network
        # ============================================================
        if use_gene_tf_modulation:
            # Learnable gene→TF effects
            self.gene_to_tf = nn.Parameter(torch.zeros(n_tfs, n_genes))
            
            # Prior from OmniPath
            self.register_buffer('gene_tf_prior', torch.zeros(n_genes, n_tfs))
            
            # Modulation strength
            self.gene_tf_modulation_strength = nn.Parameter(torch.tensor(0.2))
        
        # Update config
        self.config['use_gene_propagation'] = use_gene_propagation
        self.config['use_gene_tf_modulation'] = use_gene_tf_modulation
        self.config['model_version'] = 'v7'
    
    def set_omnipath_priors(
        self,
        gene_gene: Optional[torch.Tensor] = None,
        gene_tf: Optional[torch.Tensor] = None
    ):
        """
        Set OmniPath priors for gene networks.
        
        Args:
            gene_gene: (n_genes, n_genes) Gene→Gene signaling adjacency
            gene_tf: (n_genes, n_tfs) Gene→TF regulation adjacency
        """
        device = self.direct_effects.device
        
        if gene_gene is not None and self.use_gene_propagation:
            self.gene_gene_prior.copy_(gene_gene.float().to(device))
            
            # Initialize learnable weights from prior (scaled down)
            with torch.no_grad():
                self.gene_propagation.data = gene_gene.float().to(device) * 0.1
            
            n_edges = (gene_gene != 0).sum().item()
            print(f"  [Gene→Gene] {n_edges:,} edges from OmniPath")
        
        if gene_tf is not None and self.use_gene_tf_modulation:
            self.gene_tf_prior.copy_(gene_tf.float().to(device))
            
            # Initialize learnable weights from prior (transposed for proper dims)
            with torch.no_grad():
                self.gene_to_tf.data = gene_tf.T.float().to(device) * 0.1
            
            n_edges = (gene_tf != 0).sum().item()
            print(f"  [Gene→TF] {n_edges:,} edges from OmniPath")
    
    def compute_gene_propagation(self, gene_deltas: torch.Tensor) -> torch.Tensor:
        """
        Propagate gene expression changes through the gene network.
        
        Models signaling cascades: Gene A → Gene B → Gene C
        
        Args:
            gene_deltas: (batch, n_genes) initial gene changes (from TF effects)
            
        Returns:
            propagated_deltas: (batch, n_genes) after propagation
        """
        if not self.use_gene_propagation:
            return gene_deltas
        
        # Combine learned and prior gene-gene effects
        gene_network = self.gene_propagation + 0.5 * self.gene_gene_prior
        
        # Normalize rows (each gene's outgoing effects sum to reasonable scale)
        row_norm = gene_network.abs().sum(dim=1, keepdim=True).clamp(min=1.0)
        gene_network_norm = gene_network / row_norm
        
        # Get propagation weight (bounded)
        prop_weight = torch.sigmoid(self.gene_prop_weight) * 0.5  # max 0.5
        
        # Iterative propagation with residual
        current = gene_deltas
        for _ in range(self.gene_prop_steps):
            propagated = torch.mm(current, gene_network_norm.T)
            current = current + prop_weight * propagated
        
        return current
    
    def compute_gene_tf_modulation(self, gene_states: torch.Tensor) -> torch.Tensor:
        """
        Compute how gene states modulate TF activity.
        
        Models upstream regulation: Gene → TF
        
        Args:
            gene_states: (batch, n_genes) current gene expression
            
        Returns:
            tf_modulation: (batch, n_tfs) multiplicative modulation of TF effects
        """
        if not self.use_gene_tf_modulation:
            return torch.ones(gene_states.size(0), self.n_tfs, device=gene_states.device)
        
        # Combine learned and prior gene→TF effects
        gene_tf = self.gene_to_tf + 0.5 * self.gene_tf_prior.T
        
        # Compute raw modulation
        raw_modulation = torch.mm(gene_states, gene_tf.T)  # (batch, n_tfs)
        
        # Get modulation strength (bounded)
        mod_strength = torch.sigmoid(self.gene_tf_modulation_strength) * 0.4  # max 0.4
        
        # Convert to multiplicative gate (centered at 1.0)
        tf_modulation = 1.0 + mod_strength * torch.tanh(raw_modulation / 10.0)
        
        return tf_modulation
    
    def predict_delta_multi_tf(
        self,
        tf_activity_change: torch.Tensor,
        source_expression: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expression CHANGE with full signaling network.
        
        EXTENDS parent method with:
        1. Gene→TF modulation BEFORE parent call (upstream regulation)
        2. Gene→Gene propagation AFTER parent call (signaling cascades)
        
        Uses EXACT same core computation as IntegratedGRNModel.predict_delta_multi_tf
        
        Args:
            tf_activity_change: (batch, n_tfs) - change in activity for ALL TFs
            source_expression: (batch, n_genes) - source cell expression
            
        Returns:
            predicted_delta: (batch, n_genes)
            grn_matrix: (n_genes, n_tfs)
        """
        # ============================================================
        # Step 1: Gene→TF Modulation (NEW in v7 - BEFORE parent call)
        # ============================================================
        if self.use_gene_tf_modulation and source_expression is not None:
            tf_modulation = self.compute_gene_tf_modulation(source_expression)
            tf_activity_modulated = tf_activity_change * tf_modulation
        else:
            tf_activity_modulated = tf_activity_change
        
        # ============================================================
        # Step 1b: Cofactor gating per-TF (BEFORE GRN multiplication)
        # This modulates TF activity based on cofactor expression
        # ============================================================
        if hasattr(self, 'cof_to_tf_adj') and self.cof_to_tf_adj is not None and source_expression is not None:
            cofactor_expr = source_expression[:, self.cofactor_gene_indices.long()]  # (B, n_cof)
            cofactor_influence = torch.mm(cofactor_expr, self.cof_to_tf_adj)         # (B, n_tfs)
            cofactor_gate = 1.0 + 0.2 * torch.tanh(cofactor_influence / 5.0)         # (B, n_tfs)
            tf_activity_modulated = tf_activity_modulated * cofactor_gate
        
        # ============================================================
        # Step 2: Call PARENT's predict_delta_multi_tf - EXACT v6 code
        # ============================================================
        # This calls IntegratedGRNModel.predict_delta_multi_tf with:
        # - Same GRN computation
        # - Same context gating (FiLM)
        # - Same delta_basic + delta_neural blending
        #
        # IMPORTANT: Temporarily disable parent's cofactor gate to prevent
        # double-gating (we already applied it in Step 1b above)
        saved_cof_to_tf = getattr(self, 'cof_to_tf_adj', None)
        self.cof_to_tf_adj = None  # Disable for parent call
        
        delta, grn_matrix = super().predict_delta_multi_tf(
            tf_activity_modulated, 
            source_expression
        )
        
        # Restore cofactor buffer
        self.cof_to_tf_adj = saved_cof_to_tf
        
        # ============================================================
        # Step 3: Gene→Gene Propagation (NEW in v7 - AFTER parent call)
        # ============================================================
        if self.use_gene_propagation:
            delta = self.compute_gene_propagation(delta)
        
        return delta, grn_matrix
    
    def get_signaling_networks(self) -> Dict[str, torch.Tensor]:
        """Get all learned signaling networks for analysis."""
        networks = {
            'tf_to_gene': self.get_grn_matrix().detach(),
        }
        
        if self.use_gene_propagation:
            networks['gene_to_gene'] = (self.gene_propagation + 0.5 * self.gene_gene_prior).detach()
        
        if self.use_gene_tf_modulation:
            networks['gene_to_tf'] = (self.gene_to_tf + 0.5 * self.gene_tf_prior.T).detach()
        
        networks['tf_to_tf'] = self.tf_tf_adjacency.detach()
        
        return networks


# =============================================================================
# TRAINER - INHERITS EXACTLY FROM train_v6 IntegratedGRNTrainer
# =============================================================================

class SignalingGRNTrainer(IntegratedGRNTrainer):
    """
    Trainer for v7 model.
    
    INHERITS ALL functionality from IntegratedGRNTrainer (train_v6):
    - train_epoch_multi_tf
    - validate_multi_tf
    - _batch_correlation
    - _compute_pcc
    - _compute_direction_accuracy
    
    ADDS: L2 prior penalty for OmniPath parameters to prevent drift from priors.
    """
    
    def __init__(
        self,
        model: IntegratedGRNModelV7,
        device: torch.device,
        learning_rate: float = 1e-3,
        use_amp: bool = True,
        use_chip_reg: bool = True,
        prior_penalty_weight: float = 0.01  # L2 penalty weight for OmniPath priors
    ):
        # Call parent constructor EXACTLY
        super().__init__(
            model=model,
            device=device,
            learning_rate=learning_rate,
            use_amp=use_amp,
            use_chip_reg=use_chip_reg
        )
        self.prior_penalty_weight = prior_penalty_weight
    
    def compute_prior_penalty(self) -> torch.Tensor:
        """
        Compute L2 prior penalty for OmniPath parameters.
        
        Prevents learned weights from drifting too far from priors:
        - gene_propagation should stay close to 0.5 * gene_gene_prior
        - gene_to_tf should stay close to 0.5 * gene_tf_prior.T
        
        Returns:
            penalty: Scalar loss to add to total loss
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        model = self.model
        
        # Gene→Gene propagation penalty
        if hasattr(model, 'gene_propagation') and hasattr(model, 'gene_gene_prior'):
            target = 0.5 * model.gene_gene_prior
            diff = model.gene_propagation - target
            # Use mean to normalize by matrix size
            penalty = penalty + (diff ** 2).mean()
        
        # Gene→TF modulation penalty
        if hasattr(model, 'gene_to_tf') and hasattr(model, 'gene_tf_prior'):
            target = 0.5 * model.gene_tf_prior.T
            diff = model.gene_to_tf - target
            penalty = penalty + (diff ** 2).mean()
        
        return self.prior_penalty_weight * penalty
    
    def train_epoch_multi_tf(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Training epoch for v7 model.
        
        Overrides parent to add L2 prior penalty for OmniPath parameters.
        """
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_prior = 0.0
        all_pred_deltas = []
        all_actual_deltas = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}")
        
        for batch in pbar:
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
                
                # Add L2 prior penalty (NEW in v7)
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
            
            if len(all_pred_deltas) < 50:
                all_pred_deltas.append(pred_delta.detach().cpu())
                all_actual_deltas.append(actual_delta.detach().cpu())
            
            pbar.set_postfix({
                'loss': f'{losses["total"].item():.4f}',
                'prior': f'{prior_penalty.item():.4f}'
            })
        
        # Compute metrics
        n_batches = len(dataloader)
        all_pred = torch.cat(all_pred_deltas, dim=0)
        all_actual = torch.cat(all_actual_deltas, dim=0)
        
        delta_pcc = self._batch_correlation(all_pred, all_actual)
        
        return {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches,
            'prior_penalty': total_prior / n_batches,
            'delta_pcc': delta_pcc
        }


# =============================================================================
# MAIN - Same flow as train_v6 with OmniPath added
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BRIDGE 2.0: Full Signaling Network (extends train_v6)')
    parser.add_argument('--singlecell', type=str, required=True, help='Path to h5ad file')
    parser.add_argument('--tf-binding', type=str, required=True, help='TF binding directory')
    parser.add_argument('--omnipath', type=str, default=None, help='OmniPath directory (optional)')
    parser.add_argument('--ppi', type=str, default=None, help='PPI directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--no-gene-propagation', action='store_true', help='Disable gene→gene propagation')
    parser.add_argument('--no-gene-tf-modulation', action='store_true', help='Disable gene→TF modulation')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BRIDGE 2.0: FULL MOLECULAR SIGNALING NETWORK")
    print("(Extends train_v6 architecture with OmniPath integration)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load data (SAME AS train_v6)
    # =========================================================================
    print("\n[1/6] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(args.singlecell, args.celltype_key)
    
    print("\n[2/6] Loading TF binding data...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(args.tf_binding, gene_names)
    
    print("\n[3/6] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    print(f"  TF activity: {tf_activity.shape}")
    
    print("\n[4/6] Computing baselines...")
    baselines = compute_baseline_metrics(expression)
    print(f"  Source-Target PCC: {baselines['source_target_pcc']:.4f}")
    print(f"  Mean-Pred PCC: {baselines['mean_pred_pcc']:.4f}")
    
    # =========================================================================
    # Load OmniPath (NEW in v7)
    # =========================================================================
    print("\n[5/6] Loading OmniPath signaling network...")
    if args.omnipath:
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
        print("  No OmniPath data provided, using v6 architecture")
    
    # =========================================================================
    # Create cell-type pairs (SAME AS train_v6)
    # =========================================================================
    n_celltypes = len(celltypes)
    n_holdout = max(2, n_celltypes // 6)
    np.random.seed(42)
    holdout = set(np.random.choice(n_celltypes, n_holdout, replace=False))
    train_idx = [i for i in range(n_celltypes) if i not in holdout]
    
    train_pairs = [(i, j) for i in train_idx for j in train_idx if i != j]
    val_pairs = [(i, j) for i in holdout for j in holdout if i != j]
    
    print(f"\n  Train cell-type pairs: {len(train_pairs)}")
    print(f"  Val cell-type pairs: {len(val_pairs)}")
    
    # =========================================================================
    # Create datasets (SAME AS train_v6)
    # =========================================================================
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
    
    # =========================================================================
    # Create model (EXTENDED from train_v6)
    # =========================================================================
    print("\n[6/6] Building model...")
    
    model = IntegratedGRNModelV7(
        n_genes=len(gene_names),
        n_tfs=len(tf_names),
        hidden_dim=args.hidden_dim,
        use_gene_propagation=has_omnipath and not args.no_gene_propagation,
        use_gene_tf_modulation=has_omnipath and not args.no_gene_tf_modulation
    )
    
    # Load PPI cofactor data if available
    cof_to_tf = None
    cof_indices = None
    if args.ppi:
        ppi_dir = Path(args.ppi)
        if (ppi_dir / 'cof_to_tf_adj.npy').exists():
            cof_to_tf = np.load(ppi_dir / 'cof_to_tf_adj.npy')
            cof_indices = np.load(ppi_dir / 'cofactor_gene_indices.npy')
    
    # Set all priors in ONE call (from train_v6)
    model.set_priors(
        chip_binding=torch.from_numpy(binding_matrix),
        tf_tf_adjacency=torch.from_numpy(tf_tf_combined),
        cofactor_tf_adjacency=torch.from_numpy(cof_to_tf) if cof_to_tf is not None else None,
        cofactor_gene_indices=torch.from_numpy(cof_indices) if cof_indices is not None else None
    )
    
    # Set OmniPath priors (NEW in v7)
    if has_omnipath:
        model.set_omnipath_priors(
            gene_gene=torch.from_numpy(gene_gene),
            gene_tf=torch.from_numpy(gene_tf)
        )
    
    model = model.to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Gene propagation: {'ON' if model.use_gene_propagation else 'OFF'}")
    print(f"  Gene→TF modulation: {'ON' if model.use_gene_tf_modulation else 'OFF'}")
    
    # Save metadata
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False, header=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False, header=False)
    pd.DataFrame({'celltype': celltypes}).to_csv(output_dir / 'celltype_names.csv', index=False, header=False)
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    
    with open(output_dir / 'baselines.json', 'w') as f:
        json.dump(baselines, f, indent=2)
    
    # =========================================================================
    # Train (SAME LOGIC AS train_v6)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"  Baselines to beat: Source-Target PCC={baselines['source_target_pcc']:.4f}")
    
    trainer = SignalingGRNTrainer(model, device, learning_rate=args.lr)
    
    best_delta_pcc = -1
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Use train_epoch_multi_tf from parent class (IntegratedGRNTrainer)
        train_metrics = trainer.train_epoch_multi_tf(train_loader, epoch)
        
        print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} "
              f"delta={train_metrics.get('delta_pcc', 0):.3f} dir={train_metrics.get('dir_acc', 0):.3f}")
        
        if epoch % args.val_every == 0 or epoch == 1:
            # Use validate_multi_tf from parent class (IntegratedGRNTrainer)
            val_metrics = trainer.validate_multi_tf(val_loader)
            print(f"  [VAL] Delta_PCC={val_metrics['delta_pcc']:.4f} Dir_Acc={val_metrics['dir_acc']:.4f}")
            
            if val_metrics['delta_pcc'] > best_delta_pcc:
                best_delta_pcc = val_metrics['delta_pcc']
                best_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': model.config,
                    'delta_pcc': best_delta_pcc,
                }, output_dir / 'best_model.pt')
                
                # Save GRN matrix
                grn = model.get_grn_matrix().detach().cpu().numpy()
                np.save(output_dir / 'grn_matrix.npy', grn)
                np.save(output_dir / 'grn_matrix_CORRECT.npy', grn)
                
                # Save all signaling networks
                networks = model.get_signaling_networks()
                for name, tensor in networks.items():
                    np.save(output_dir / f'{name}.npy', tensor.cpu().numpy())
                
                print(f"  >> New best! Delta_PCC={best_delta_pcc:.4f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best model: Epoch {best_epoch}, Delta_PCC={best_delta_pcc:.4f}")
    print(f"Saved to: {output_dir}")
    
    # Save final config
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model.config, f, indent=2)


if __name__ == '__main__':
    main()

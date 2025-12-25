#!/usr/bin/env python3
"""
================================================================================
Train GRN Model v5 - FIXED Virtual CRISPRi/CRISPRa
================================================================================

CRITICAL FIXES FROM v4:
========================
1. Cell-type PAIR holdout (not sample-level split)
2. Focus on DELTA prediction, not absolute expression
3. Compute GRN matrix ONCE per epoch (not every batch)
4. Stricter ChIP regularization
5. Add baseline comparison
6. Learned residual weight
7. Proper warmup for stable training

ARCHITECTURE REASONING:
=======================
- Expression Encoder: Compress 16K genes → 256 latent for efficiency
- TF Cascade GCN: Model TF→TF regulatory cascades (if TF1→TF2→Gene, KO TF1 affects Gene)
- Direct Effects: Sparse TF→Gene matrix initialized from ChIP-seq
- Neural Effects: Learn complex nonlinear TF→Gene relationships beyond ChIP
- Residual Decoder: Refine predictions with context

Usage:
    python scripts/train_v5.py \\
        --singlecell data/singlecell/preprocessed.h5ad \\
        --tf-binding data/tf_binding/ \\
        --celltype-key celltype_subset \\
        --output results/grn_model_v5/ \\
        --epochs 100
"""

import os
import sys
import json
import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

# Try DGL for TF cascade GCN
try:
    import dgl
    from dgl.nn import GraphConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_baseline_metrics(expression_means: np.ndarray) -> Dict[str, float]:
    """
    Compute baseline metrics to understand what the model needs to beat.
    
    If model PCC ≈ baseline PCC, it's not learning anything!
    """
    n_types = expression_means.shape[0]
    
    # Source-Target correlation baseline
    source_target_pccs = []
    for i in range(n_types):
        for j in range(n_types):
            if i != j:
                r = np.corrcoef(expression_means[i], expression_means[j])[0, 1]
                source_target_pccs.append(r)
                
    # Mean prediction baseline (just predict average)
    mean_expr = expression_means.mean(axis=0)
    mean_pred_pccs = []
    for i in range(n_types):
        r = np.corrcoef(mean_expr, expression_means[i])[0, 1]
        mean_pred_pccs.append(r)
        
    return {
        'source_target_pcc': float(np.mean(source_target_pccs)),
        'mean_pred_pcc': float(np.mean(mean_pred_pccs)),
        'min_pair_pcc': float(np.min(source_target_pccs)),
        'max_pair_pcc': float(np.max(source_target_pccs))
    }


# =============================================================================
# TF ACTIVITY COMPUTATION (LINGER APPROACH)
# =============================================================================

def compute_tf_activity_linger(
    expression: np.ndarray,
    binding_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute TF activity using LINGER's ChIP-weighted approach.
    
    From LINGER (Nature Biotech 2024):
    "TF activity is reflected in the expression of its target genes"
    
    TF_activity[celltype, tf] = Σ(binding[g,tf] × expr[celltype,g]) / Σ(binding[g,tf])
    """
    weighted_sum = expression @ binding_matrix
    n_targets = binding_matrix.sum(axis=0, keepdims=True).clip(min=1)
    tf_activity = weighted_sum / n_targets
    return tf_activity.astype(np.float32)


def classify_tf_states(
    tf_activity: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Classify TF states as ACTIVE (+1), INACTIVE (-1), or VARIABLE (0)."""
    mean = tf_activity.mean(axis=0, keepdims=True)
    std = tf_activity.std(axis=0, keepdims=True).clip(min=1e-6)
    z_scores = (tf_activity - mean) / std
    
    states = np.zeros_like(tf_activity, dtype=np.float32)
    states[z_scores > threshold] = 1.0
    states[z_scores < -threshold] = -1.0
    return states


# =============================================================================
# DATASET WITH CELL-TYPE PAIR HOLDOUT
# =============================================================================

class VirtualPerturbationDatasetV5(Dataset):
    """
    Virtual perturbation dataset with PROPER cell-type pair holdout.
    
    Key fix: Validation pairs are NEVER seen during training.
    
    Improvements:
    - TF-balanced sampling weights (so frequent TFs don't dominate)
    - Hard negatives (low |Δ| samples for specificity calibration)
    """
    
    def __init__(
        self,
        expression_means: np.ndarray,
        tf_activity: np.ndarray,
        celltype_names: List[str],
        allowed_pairs: List[Tuple[int, int]],  # Only these pairs are included
        min_activity_diff: float = 0.1,
        add_hard_negatives: bool = True,
        hard_negative_fraction: float = 0.1
    ):
        self.expression = torch.FloatTensor(expression_means)
        self.tf_activity = torch.FloatTensor(tf_activity)
        self.celltype_names = celltype_names
        self.allowed_pairs = set(allowed_pairs)
        
        self.n_celltypes = len(celltype_names)
        self.n_genes = expression_means.shape[1]
        self.n_tfs = tf_activity.shape[1]
        
        self.samples = self._generate_samples(min_activity_diff, add_hard_negatives, hard_negative_fraction)
        self.sample_weights = self._compute_balanced_weights()
        
    def _generate_samples(self, min_diff: float, add_hard_negatives: bool, 
                          hard_neg_frac: float) -> List[Dict]:
        samples = []
        hard_negatives = []
        
        for src_idx, tgt_idx in self.allowed_pairs:
            if src_idx == tgt_idx:
                continue
                
            activity_diff = self.tf_activity[src_idx] - self.tf_activity[tgt_idx]
            
            for tf_idx in range(self.n_tfs):
                diff = activity_diff[tf_idx].item()
                
                # Hard negatives: low activity difference (near-zero expected effect)
                if add_hard_negatives and 0.01 < abs(diff) < min_diff:
                    hard_negatives.append({
                        'source_idx': src_idx,
                        'target_idx': tgt_idx,
                        'tf_idx': tf_idx,
                        'perturbation_type': 1 if diff < 0 else -1,
                        'perturbation_strength': abs(diff),
                        'is_hard_negative': True,
                    })
                    continue
                    
                if abs(diff) < min_diff:
                    continue
                    
                if diff > 0:
                    pert_type = -1  # Knockout
                else:
                    pert_type = +1  # Overexpression
                    
                samples.append({
                    'source_idx': src_idx,
                    'target_idx': tgt_idx,
                    'tf_idx': tf_idx,
                    'perturbation_type': pert_type,
                    'perturbation_strength': abs(diff),
                    'is_hard_negative': False,
                })
        
        # Add subset of hard negatives
        if add_hard_negatives and hard_negatives:
            n_hard = int(len(samples) * hard_neg_frac)
            np.random.shuffle(hard_negatives)
            samples.extend(hard_negatives[:n_hard])
            
        return samples
    
    def _compute_balanced_weights(self) -> np.ndarray:
        """
        Compute per-sample weights so each TF is equally represented.
        This prevents frequent TFs from dominating training.
        """
        tf_counts = np.zeros(self.n_tfs)
        for s in self.samples:
            tf_counts[s['tf_idx']] += 1
        
        # Weight = 1 / count (inverse frequency)
        tf_weights = np.zeros(self.n_tfs)
        for i in range(self.n_tfs):
            if tf_counts[i] > 0:
                tf_weights[i] = 1.0 / tf_counts[i]
        
        # Normalize
        tf_weights = tf_weights / tf_weights.sum() * len(tf_weights)
        
        # Assign to samples
        sample_weights = np.array([tf_weights[s['tf_idx']] for s in self.samples])
        return sample_weights
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        src_expr = self.expression[sample['source_idx']]
        tgt_expr = self.expression[sample['target_idx']]
        
        return {
            'source_expression': src_expr,
            'target_expression': tgt_expr,
            'delta_expression': tgt_expr - src_expr,  # CRITICAL: explicit delta
            'tf_idx': torch.tensor(sample['tf_idx'], dtype=torch.long),
            'perturbation_type': torch.tensor(sample['perturbation_type'], dtype=torch.float),
            'perturbation_strength': torch.tensor(sample['perturbation_strength'], dtype=torch.float),
        }


# =============================================================================
# MULTI-TF DATASET (CORRECT APPROACH)
# =============================================================================

class MultiTFDataset(Dataset):
    """
    Dataset that uses ALL TF activity changes together.
    
    This is the CORRECT approach: when going from Cell A to Cell B,
    ALL TFs change simultaneously. The total delta is the combined
    effect of all TF changes, not attributable to just one TF.
    
    Each sample:
    - Input: ALL TF activity changes (n_tfs,)
    - Input: Source expression for context (n_genes,)
    - Target: Total expression delta (n_genes,)
    
    Data augmentation:
    - For each cell-type pair, create multiple samples with interpolation
    - This increases sample count while staying biologically meaningful
    """
    
    def __init__(
        self,
        expression_means: np.ndarray,  # (n_celltypes, n_genes)
        tf_activity: np.ndarray,       # (n_celltypes, n_tfs)
        celltype_names: List[str],
        allowed_pairs: List[Tuple[int, int]],
        augment_factor: int = 10,  # Create this many samples per pair
        noise_std: float = 0.05    # Add small noise for regularization
    ):
        self.expression = torch.FloatTensor(expression_means)
        self.tf_activity = torch.FloatTensor(tf_activity)
        self.celltype_names = celltype_names
        self.base_pairs = list(allowed_pairs)
        self.augment_factor = augment_factor
        self.noise_std = noise_std
        
        self.n_celltypes = len(celltype_names)
        self.n_genes = expression_means.shape[1]
        self.n_tfs = tf_activity.shape[1]
        
        # Create augmented sample list
        # Each base pair gets augment_factor samples with different interpolation alphas
        self.samples = []
        for src_idx, tgt_idx in self.base_pairs:
            # Original full transition (alpha=1.0)
            self.samples.append((src_idx, tgt_idx, 1.0))
            
            # Interpolated transitions (partial changes)
            for i in range(1, augment_factor):
                alpha = i / augment_factor  # 0.1, 0.2, ..., 0.9
                self.samples.append((src_idx, tgt_idx, alpha))
        
        print(f"  Multi-TF augmented: {len(self.base_pairs)} pairs × {augment_factor} = {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src_idx, tgt_idx, alpha = self.samples[idx]
        
        # Source expression (fixed)
        source_expr = self.expression[src_idx]
        
        # Target expression (interpolated based on alpha)
        full_target = self.expression[tgt_idx]
        target_expr = source_expr + alpha * (full_target - source_expr)
        
        # TF activity change (scaled by alpha)
        full_tf_change = self.tf_activity[tgt_idx] - self.tf_activity[src_idx]
        tf_activity_change = alpha * full_tf_change
        
        # Delta (scaled by alpha)
        delta_expr = target_expr - source_expr
        
        # Add small noise for regularization (only during training)
        if self.noise_std > 0:
            noise = torch.randn_like(tf_activity_change) * self.noise_std
            tf_activity_change = tf_activity_change + noise
        
        return {
            'source_expression': source_expr,
            'target_expression': target_expr,
            'tf_activity_change': tf_activity_change,
            'delta_expression': delta_expr,
            'src_idx': src_idx,
            'tgt_idx': tgt_idx
        }


# =============================================================================
# TF CASCADE GCN
# =============================================================================

class TFCascadeGCN(nn.Module):
    """
    GCN for TF-TF cascade effects (master regulator modeling).
    
    FIXES:
    - DGL mode now uses edge weights (binding strength)
    - Non-DGL mode now properly uses the adjacency matrix for message passing
    - Edge dropout to prevent over-amplification in dense TF-TF graphs
    """
    
    def __init__(self, hidden_dim: int, n_layers: int = 2, dropout: float = 0.3,
                 edge_dropout: float = 0.2):
        super().__init__()
        
        self.n_layers = n_layers
        self.use_dgl = DGL_AVAILABLE
        self.edge_dropout = edge_dropout
        
        if self.use_dgl:
            self.layers = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(n_layers):
                # Use norm='none' to manually handle edge weights
                self.layers.append(GraphConv(hidden_dim, hidden_dim, 
                                             allow_zero_in_degree=True, norm='none'))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            # Non-DGL: Linear layers (will multiply by adjacency in forward)
            self.layers = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
        self.dropout = nn.Dropout(dropout)
        self._cached_graph = None
        self._edge_weights = None
        self._adjacency_norm = None  # Normalized adjacency for non-DGL mode
        
    def set_graph(self, adjacency: torch.Tensor):
        """
        Set the TF-TF adjacency graph.
        
        FIXED: Now properly stores edge weights for DGL mode
        and normalized adjacency for non-DGL mode.
        """
        if adjacency is None:
            return
            
        n_tfs = adjacency.shape[0]
        
        if self.use_dgl:
            # Build adjacency with self-loops for normalization
            adj_with_self = adjacency.float() + torch.eye(n_tfs, device=adjacency.device)
            
            # Row-normalize adjacency FIRST (like non-DGL path)
            row_sum = adj_with_self.sum(dim=1, keepdim=True).clamp(min=1e-6)
            adj_normalized = adj_with_self / row_sum
            
            # Extract edges from normalized adjacency
            src, dst = torch.where(adj_normalized > 0)
            edge_weights = adj_normalized[src, dst].float()
            
            # Build DGL graph (self-loops already included in adj_with_self)
            graph = dgl.graph((src.cpu(), dst.cpu()), num_nodes=n_tfs)
            
            # Store normalized edge weights
            graph.edata['w'] = edge_weights
            
            self._cached_graph = graph
            self._edge_weights = edge_weights
            print(f"  [TFCascadeGCN] DGL graph: {n_tfs} nodes, {len(edge_weights)} edges (row-normalized)")
        else:
            # Non-DGL: Create normalized adjacency (add self-loops, row-normalize)
            adj = adjacency.float() + torch.eye(n_tfs, device=adjacency.device)
            row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
            self._adjacency_norm = adj / row_sum
            
    def forward(self, tf_features: torch.Tensor) -> torch.Tensor:
        h = tf_features
        
        if self.use_dgl and self._cached_graph is not None:
            graph = self._cached_graph
            if graph.device != h.device:
                graph = graph.to(h.device)
                self._cached_graph = graph
            
            # Get edge weights
            edge_weights = graph.edata.get('w', None)
            if edge_weights is not None and edge_weights.device != h.device:
                edge_weights = edge_weights.to(h.device)
                graph.edata['w'] = edge_weights
            
            for conv, bn in zip(self.layers, self.bns):
                # Edge dropout during training
                if self.training and self.edge_dropout > 0:
                    mask = torch.rand(edge_weights.shape, device=h.device) > self.edge_dropout
                    ew = edge_weights * mask.float() / (1 - self.edge_dropout)
                else:
                    ew = edge_weights
                
                # Pass edge weights to GraphConv
                h_new = conv(graph, h, edge_weight=ew)
                h_new = bn(h_new)
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)
                h = h + 0.5 * h_new  # Scaled residual
        else:
            # Non-DGL mode: Use adjacency matrix for message passing
            # FIXED: Previously ignored adjacency entirely!
            adj = self._adjacency_norm
            if adj is not None and adj.device != h.device:
                adj = adj.to(h.device)
                self._adjacency_norm = adj
            
            for linear, bn in zip(self.layers, self.bns):
                # Message passing: aggregate neighbor features
                if adj is not None:
                    # Edge dropout
                    if self.training and self.edge_dropout > 0:
                        mask = torch.rand_like(adj) > self.edge_dropout
                        adj_drop = adj * mask.float() / (1 - self.edge_dropout)
                    else:
                        adj_drop = adj
                    h_agg = torch.mm(adj_drop, h)  # Aggregate neighbors
                else:
                    h_agg = h
                
                h_new = linear(h_agg)
                h_new = bn(h_new)
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)
                h = h + 0.5 * h_new  # Scaled residual
                
        return h


# =============================================================================
# IMPROVED MODEL
# =============================================================================

class DeltaGRNModel(nn.Module):
    """
    GRN model that focuses on DELTA prediction.
    
    Key insight: We want to learn TF→Gene effects, not source→target mapping.
    So we predict Δexpression directly and compare to actual Δ.
    
    IMPROVED: 
    - Context gating so TF effects depend on cell state
    - TF specificity weights from JASPAR motif analysis
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 256,
        n_cascade_layers: int = 2,
        dropout: float = 0.2,
        use_context_gating: bool = True
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        self.use_context_gating = use_context_gating
        
        # TF embeddings
        self.tf_embeddings = nn.Embedding(n_tfs, hidden_dim)
        
        # TF cascade (TF→TF effects)
        self.tf_cascade = TFCascadeGCN(hidden_dim, n_cascade_layers, dropout)
        
        # TF→Gene effect decoder
        # This is the CORE of GRN learning: TF embedding → effect on all genes
        self.effect_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_genes)
        )
        
        # Direct TF→Gene effects (sparse, initialized from ChIP)
        self.direct_effects = nn.Parameter(torch.zeros(n_genes, n_tfs))
        
        # LEARNABLE ChIP mask leakage (starts at 0.1, can learn to be stricter)
        # This replaces the hard 0.01 constant
        self._leakage_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        # CONTEXT GATING: Makes TF effects depend on cell state
        # FiLM-style: learn γ (scale) and β (shift) from source expression
        if use_context_gating:
            # Encode source expression to a compact context
            context_dim = 128
            self.context_encoder = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, context_dim)
            )
            # Context → per-gene gate (scale factor γ)
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, n_genes),
                nn.Sigmoid()  # Output in [0, 1] to modulate effects
            )
            # Context → per-gene shift (β) 
            self.context_shift = nn.Sequential(
                nn.Linear(context_dim, n_genes),
                nn.Tanh()  # Small shifts
            )
        
        # Buffers for ChIP data and TF specificity
        self.register_buffer('chip_mask', torch.zeros(n_genes, n_tfs))
        # TF specificity weights from JASPAR (range ~0.93-1.49, mean ~1.24)
        # Higher = more specific motif (amplifies effect), lower = promiscuous (dampens)
        self.register_buffer('tf_specificity_weights', torch.ones(n_tfs))
        
        # Config for saving
        self.config = {
            'n_genes': n_genes,
            'n_tfs': n_tfs,
            'hidden_dim': hidden_dim,
            'n_cascade_layers': n_cascade_layers,
            'use_context_gating': use_context_gating,
            'uses_tf_specificity': True  # Will be set in set_priors
        }
        
        # Cache for GRN matrix (computed once per epoch)
        self._grn_cache = None
        self._grn_cache_epoch = -1
        
    def set_priors(
        self, 
        chip_binding: torch.Tensor, 
        tf_tf_adjacency: Optional[torch.Tensor] = None,
        tf_specificity_weights: Optional[torch.Tensor] = None,
        cofactor_tf_adjacency: Optional[torch.Tensor] = None,
        cofactor_gene_indices: Optional[torch.Tensor] = None
    ):
        """
        Initialize from biological priors.
        
        Args:
            chip_binding: TF-gene binding matrix (n_genes x n_tfs)
            tf_tf_adjacency: TF-TF regulatory graph (n_tfs x n_tfs)
            tf_specificity_weights: Per-TF specificity weights from JASPAR+PPI analysis
                Range ~[0.7, 1.3], mean ~1.0
                Higher = specific TF (unique motif, few targets) -> amplify
                Lower = promiscuous TF (broad motif, many targets) -> dampen
            cofactor_tf_adjacency: Cofactor->TF interaction matrix (n_cofactors x n_tfs)
                From Rosetta PPI predictions, row-normalized
            cofactor_gene_indices: Gene indices for cofactors in expression matrix (n_cofactors,)
        """
        self.chip_mask = chip_binding.to(self.direct_effects.device)
        
        # Initialize direct effects: small values where ChIP binding exists
        with torch.no_grad():
            self.direct_effects.data = chip_binding.float() * 0.01
            
        if tf_tf_adjacency is not None:
            self.tf_cascade.set_graph(tf_tf_adjacency)
            
        # Set TF specificity weights
        if tf_specificity_weights is not None:
            self.tf_specificity_weights = tf_specificity_weights.to(self.direct_effects.device)
            self.config['uses_tf_specificity'] = True
        else:
            self.config['uses_tf_specificity'] = False
            
        # Set cofactor gate priors (from Rosetta PPI)
        if cofactor_tf_adjacency is not None and cofactor_gene_indices is not None:
            self.register_buffer('cof_to_tf_adj', cofactor_tf_adjacency.to(self.direct_effects.device))
            self.register_buffer('cofactor_gene_indices', cofactor_gene_indices.to(self.direct_effects.device))
            self.config['uses_cofactor_gate'] = True
            print(f"  [Cofactor Gate] {len(cofactor_gene_indices)} cofactors -> {self.n_tfs} TFs")
        else:
            self.cof_to_tf_adj = None
            self.cofactor_gene_indices = None
            self.config['uses_cofactor_gate'] = False
    
    @property
    def chip_leakage(self) -> torch.Tensor:
        """
        Learnable ChIP mask leakage in range [0.005, 0.2].
        
        This controls how much signal can "leak" through non-ChIP edges.
        - Early training: model can explore (higher leakage)
        - Late training: model should be stricter (lower leakage)
        """
        # Sigmoid maps (-inf, +inf) to (0, 1), then scale to [0.005, 0.2]
        return 0.005 + 0.195 * torch.sigmoid(self._leakage_raw)
            
    def get_grn_matrix(self, epoch: int = -1) -> torch.Tensor:
        """
        Get full TF→Gene effect matrix.
        
        OPTIMIZATION: Cache per epoch to avoid recomputation every batch.
        
        Returns: (n_genes, n_tfs) matrix of TF→Gene regulatory effects,
                 weighted by TF specificity.
        """
        if epoch >= 0 and epoch == self._grn_cache_epoch and self._grn_cache is not None:
            return self._grn_cache
            
        # All TF embeddings
        tf_indices = torch.arange(self.n_tfs, device=self.tf_embeddings.weight.device)
        tf_emb = self.tf_embeddings(tf_indices)  # (n_tfs, hidden)
        
        # Apply cascade
        tf_emb = self.tf_cascade(tf_emb)
        
        # Decode to gene effects
        neural_effects = self.effect_decoder(tf_emb)  # (n_tfs, n_genes)
        neural_effects = neural_effects.T  # (n_genes, n_tfs)
        
        # Combine with direct effects
        grn_matrix = neural_effects + self.direct_effects
        
        # Apply TF specificity weights (column-wise, one weight per TF)
        # This amplifies specific TFs and dampens promiscuous ones
        grn_matrix = grn_matrix * self.tf_specificity_weights.unsqueeze(0)  # (1, n_tfs)
        
        # Apply ChIP mask with LEARNABLE leakage
        grn_matrix = grn_matrix * (self.chip_mask + self.chip_leakage)
        
        # Cache
        if epoch >= 0:
            self._grn_cache = grn_matrix.detach()
            self._grn_cache_epoch = epoch
            
        return grn_matrix
        
    def predict_delta(
        self,
        tf_idx: torch.Tensor,
        perturbation_type: torch.Tensor,
        perturbation_strength: torch.Tensor,
        source_expression: torch.Tensor = None,
        return_effects: bool = False
    ) -> torch.Tensor:
        """
        Predict expression CHANGE from TF perturbation.
        
        This is the KEY difference from v4: we predict Δ, not absolute expression.
        
        IMPROVED: 
        - Context gating makes TF effects cell-state dependent
        - TF specificity weights modulate effect magnitude based on motif uniqueness
        
        If return_effects=True, also returns unmasked gene_effects for ChIP regularization.
        """
        batch_size = tf_idx.size(0)
        
        # Get TF embeddings with cascade effects
        all_tf_emb = self.tf_embeddings.weight
        all_tf_cascade = self.tf_cascade(all_tf_emb)
        tf_emb = all_tf_cascade[tf_idx]  # (batch, hidden)
        
        # Decode to gene effects
        gene_effects = self.effect_decoder(tf_emb)  # (batch, n_genes)
        
        # Add direct effects
        direct = self.direct_effects[:, tf_idx].T  # (batch, n_genes)
        gene_effects_unmasked = gene_effects + direct  # KEEP THIS FOR CHIP REG
        
        # CONTEXT GATING: Modulate effects based on cell state
        if self.use_context_gating and source_expression is not None:
            context = self.context_encoder(source_expression)  # (batch, context_dim)
            gamma = self.context_gate(context)  # (batch, n_genes) in [0,1]
            beta = self.context_shift(context) * 0.1  # Small shifts
            # FiLM-style modulation: effects depend on cell state
            gene_effects_unmasked = gamma * gene_effects_unmasked + beta
        
        # COFACTOR GATE: TF effects modulated by cofactor availability (from PPI)
        # When cofactors are expressed, their partner TFs have stronger effects
        if self.cof_to_tf_adj is not None and source_expression is not None:
            # Extract cofactor expression from source
            cofactor_expr = source_expression[:, self.cofactor_gene_indices]  # (batch, n_cofactors)
            # Compute per-TF cofactor influence: sum of (cofactor_expr * PPI_weight)
            cof_influence = torch.mm(cofactor_expr, self.cof_to_tf_adj)  # (batch, n_tfs)
            # Convert to mild gate: 1.0 + alpha * tanh(normalized influence)
            # alpha=0.2 keeps modulation bounded to [0.8, 1.2]
            cof_norm = cof_influence / (cof_influence.abs().mean(dim=1, keepdim=True) + 1e-6)
            cof_gate = 1.0 + 0.2 * torch.tanh(cof_norm)  # (batch, n_tfs)
            # Apply to the specific TFs in this batch
            cof_gate_batch = cof_gate.gather(1, tf_idx.unsqueeze(1))  # (batch, 1)
            gene_effects_unmasked = gene_effects_unmasked * cof_gate_batch
        
        # TF SPECIFICITY WEIGHTING (JASPAR+PPI): 
        # More specific TFs (unique motifs, few targets) get amplified effects
        # Promiscuous TFs get dampened effects
        # Weights range ~[0.7, 1.3], mean ~1.0 - moderate, not too aggressive
        tf_spec_weights = self.tf_specificity_weights[tf_idx].unsqueeze(1)  # (batch, 1)
        gene_effects_unmasked = gene_effects_unmasked * tf_spec_weights
        
        # Apply ChIP mask with LEARNABLE leakage
        leakage = self.chip_leakage  # Learnable, in [0.005, 0.2]
        mask = self.chip_mask[:, tf_idx].T + leakage
        gene_effects_masked = gene_effects_unmasked * mask
        
        # Scale by perturbation direction and strength
        scaling = perturbation_type.unsqueeze(1) * perturbation_strength.unsqueeze(1)
        delta = gene_effects_masked * scaling
        
        if return_effects:
            # Return chip_mask for the batch TFs for proper regularization
            batch_chip_mask = self.chip_mask[:, tf_idx].T  # (batch, n_genes)
            return delta, gene_effects_unmasked, batch_chip_mask
        return delta
        
    def predict_delta_multi_tf(
        self,
        tf_activity_change: torch.Tensor,
        source_expression: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expression CHANGE from ALL TF activity changes together.
        
        This is the CORRECT approach: cell-type transitions involve ALL TFs
        changing together, not just one TF at a time.
        
        Args:
            tf_activity_change: (batch, n_tfs) - change in activity for ALL TFs
            source_expression: (batch, n_genes) - source cell expression
            
        Returns:
            delta: (batch, n_genes) - predicted expression change
            grn_matrix: (n_genes, n_tfs) - for regularization
        """
        # Get full GRN matrix: (n_genes, n_tfs)
        grn_matrix = self.get_grn_matrix()
        
        # CONTEXT GATING: Modulate GRN based on cell state
        if self.use_context_gating and source_expression is not None:
            context = self.context_encoder(source_expression)  # (batch, context_dim)
            gamma = self.context_gate(context)  # (batch, n_genes) in [0,1]
            beta = self.context_shift(context) * 0.1  # Small shifts
            
            # Predicted delta = (TF_activity_change @ GRN^T) * gamma + beta
            # (batch, n_tfs) @ (n_tfs, n_genes) -> (batch, n_genes)
            raw_delta = torch.mm(tf_activity_change, grn_matrix.T)
            delta = gamma * raw_delta + beta
        else:
            # Simple linear: delta = TF_activity_change @ GRN^T
            delta = torch.mm(tf_activity_change, grn_matrix.T)
        
        return delta, grn_matrix
    
    def forward(
        self,
        source_expression: torch.Tensor,
        tf_idx: torch.Tensor,
        perturbation_type: torch.Tensor,
        perturbation_strength: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: predict both delta and final expression."""
        
        # Predict delta with gene effects for ChIP regularization
        # Pass source_expression for context gating (cell-state dependent effects)
        delta, gene_effects, batch_chip_mask = self.predict_delta(
            tf_idx, perturbation_type, perturbation_strength,
            source_expression=source_expression,  # For context gating!
            return_effects=True
        )
        
        # Predicted expression = source + scaled delta
        predicted = source_expression + delta
        
        return {
            'predicted_expression': predicted,
            'delta_expression': delta,
            'gene_effects': gene_effects,  # For ChIP reg (has gradients!)
            'batch_chip_mask': batch_chip_mask,  # Mask for this batch's TFs
        }
    
    def forward_multi_tf(
        self,
        source_expression: torch.Tensor,
        tf_activity_change: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using ALL TF activity changes.
        
        This is the biologically correct approach for training.
        """
        delta, grn_matrix = self.predict_delta_multi_tf(
            tf_activity_change, source_expression
        )
        
        predicted = source_expression + delta
        
        return {
            'predicted_expression': predicted,
            'delta_expression': delta,
            'grn_matrix': grn_matrix
        }


# =============================================================================
# IMPROVED LOSS FUNCTION
# =============================================================================

class DeltaFocusedLoss(nn.Module):
    """
    Loss that focuses on DELTA prediction accuracy.
    
    Key insight: We care about predicting the CHANGE correctly,
    not just getting high correlation with target (which is easy due to shared biology).
    """
    
    def __init__(
        self,
        delta_weight: float = 1.0,
        ranking_weight: float = 0.3,
        direction_weight: float = 0.5,  # Increased!
        chip_reg_weight: float = 0.2    # Increased!
    ):
        super().__init__()
        self.delta_weight = delta_weight
        self.ranking_weight = ranking_weight
        self.direction_weight = direction_weight
        self.chip_reg_weight = chip_reg_weight
        
    def forward(
        self,
        pred_delta: torch.Tensor,
        actual_delta: torch.Tensor,
        gene_effects: Optional[torch.Tensor] = None,
        batch_chip_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss focused on delta prediction.
        
        NOTE: We do NOT use predicted vs target expression!
        We only use predicted_delta vs actual_delta.
        
        FIXED: chip_loss now computed on batch gene_effects (has gradients!)
        """
        # 1. Delta MSE (main loss)
        delta_loss = F.mse_loss(pred_delta, actual_delta)
        
        # 2. Delta ranking loss (preserve which genes change more)
        ranking_loss = self._ranking_loss(pred_delta, actual_delta)
        
        # 3. Direction loss (up/down accuracy) - CRITICAL
        direction_loss = self._direction_loss(pred_delta, actual_delta)
        
        # 4. ChIP regularization - NOW WITH GRADIENTS!
        # Penalize gene_effects where there's no ChIP binding evidence
        chip_loss = torch.tensor(0.0, device=pred_delta.device)
        if gene_effects is not None and batch_chip_mask is not None:
            no_binding = (batch_chip_mask < 0.5).float()  # (batch, n_genes)
            # L1 penalty on effects without binding support
            chip_loss = (gene_effects.abs() * no_binding).mean()
            
        total = (
            self.delta_weight * delta_loss +
            self.ranking_weight * ranking_loss +
            self.direction_weight * direction_loss +
            self.chip_reg_weight * chip_loss
        )
        
        return {
            'total': total,
            'delta_mse': delta_loss,
            'ranking': ranking_loss,
            'direction': direction_loss,
            'chip_reg': chip_loss
        }
        
    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor, n_pairs: int = 500) -> torch.Tensor:
        batch_size, n_genes = pred.shape
        if n_genes < 2:
            return torch.tensor(0.0, device=pred.device)
            
        losses = []
        for b in range(min(batch_size, 8)):  # Limit for speed
            idx1 = torch.randint(0, n_genes, (n_pairs,), device=pred.device)
            idx2 = torch.randint(0, n_genes, (n_pairs,), device=pred.device)
            
            pred_diff = pred[b, idx1] - pred[b, idx2]
            target_diff = target[b, idx1] - target[b, idx2]
            
            # Margin ranking loss
            target_sign = torch.sign(target_diff)
            loss = F.relu(-target_sign * pred_diff + 0.1).mean()
            losses.append(loss)
            
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)
        
    def _direction_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy on direction prediction (AMP-safe)."""
        # Focus on significant changes
        significant_mask = actual.abs() > actual.abs().mean()
        
        if significant_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Use logits directly (binary_cross_entropy_with_logits is AMP-safe)
        pred_logits = pred[significant_mask] * 3  # Scale for sharpness
        actual_binary = (actual[significant_mask] > 0).float()
        
        return F.binary_cross_entropy_with_logits(pred_logits, actual_binary)


# =============================================================================
# IMPROVED TRAINER
# =============================================================================

class DeltaGRNTrainer:
    """Trainer with proper metrics and caching."""
    
    def __init__(self, model: DeltaGRNModel, device: str = 'cuda', lr: float = 1e-3, 
                 use_amp: bool = True, chip_reg_weight: float = 0.2):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.use_amp = use_amp and device == 'cuda'
        
        # chip_reg_weight=0.0 disables ChIP regularization (ablation)
        self.loss_fn = DeltaFocusedLoss(chip_reg_weight=chip_reg_weight)
        
        # Parameter-specific weight decay:
        # - Lower WD on embeddings/priors (to preserve biological initialization)
        # - Higher WD on dense layers (for regularization)
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if 'embedding' in name or 'direct_effects' in name or '_leakage' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 2e-4},
            {'params': no_decay_params, 'weight_decay': 1e-6}
        ], lr=lr)
        
        self.scheduler = None
        self.history = []
        self.baseline_metrics = None
        
        # AMP (Automatic Mixed Precision) for speed
        # Using new torch.amp API (avoids FutureWarning)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
    def set_baseline(self, baseline_metrics: Dict[str, float]):
        self.baseline_metrics = baseline_metrics
        
    def setup_scheduler(self, n_epochs: int, warmup_epochs: int = 5):
        """Warmup + cosine decay."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
                
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
    def compute_metrics(
        self,
        pred_delta: torch.Tensor,
        actual_delta: torch.Tensor,
        pred_expr: torch.Tensor,
        target_expr: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pred_delta = pred_delta.detach().cpu().numpy()
        actual_delta = actual_delta.detach().cpu().numpy()
        pred_expr = pred_expr.detach().cpu().numpy()
        target_expr = target_expr.detach().cpu().numpy()
        
        # Delta PCC (THE KEY METRIC!)
        delta_pccs = []
        for i in range(pred_delta.shape[0]):
            r = np.corrcoef(pred_delta[i], actual_delta[i])[0, 1]
            if not np.isnan(r):
                delta_pccs.append(r)
        delta_pcc = np.mean(delta_pccs) if delta_pccs else 0.0
        
        # Expression PCC (for comparison, but NOT the main metric)
        expr_pccs = []
        for i in range(pred_expr.shape[0]):
            r = np.corrcoef(pred_expr[i], target_expr[i])[0, 1]
            if not np.isnan(r):
                expr_pccs.append(r)
        expr_pcc = np.mean(expr_pccs) if expr_pccs else 0.0
        
        # Direction accuracy
        pred_direction = np.sign(pred_delta)
        actual_direction = np.sign(actual_delta)
        significant = np.abs(actual_delta) > np.abs(actual_delta).mean()
        if significant.sum() > 0:
            direction_acc = (pred_direction[significant] == actual_direction[significant]).mean()
        else:
            direction_acc = 0.5
            
        return {
            'delta_pcc': float(delta_pcc),
            'expr_pcc': float(expr_pcc),
            'direction_acc': float(direction_acc)
        }
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        # NOTE: ChIP regularization now computed per-batch with gradients
        # No longer caching epoch-level GRN matrix for reg (was broken - no gradients)
        
        total_loss = 0.0
        # Only store first N batches for metrics (memory efficient!)
        max_metric_batches = 50
        all_pred_deltas = []
        all_actual_deltas = []
        all_pred_exprs = []
        all_target_exprs = []
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}', leave=True)
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            source = batch['source_expression'].to(self.device)
            target = batch['target_expression'].to(self.device)
            actual_delta = batch['delta_expression'].to(self.device)
            tf_idx = batch['tf_idx'].to(self.device)
            pert_type = batch['perturbation_type'].to(self.device)
            pert_strength = batch['perturbation_strength'].to(self.device)
            
            # Forward with AMP (Automatic Mixed Precision)
            # Using new torch.amp API (avoids FutureWarning)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(source, tf_idx, pert_type, pert_strength)
                
                # Loss (focused on DELTA, not expression)
                # NOTE: Using batch-level gene_effects for ChIP reg (HAS GRADIENTS!)
                losses = self.loss_fn(
                    pred_delta=outputs['delta_expression'],
                    actual_delta=actual_delta,
                    gene_effects=outputs['gene_effects'],
                    batch_chip_mask=outputs['batch_chip_mask']
                )
            
            # Backward with gradient scaling for AMP
            if self.scaler is not None:
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += losses['total'].item()
            n_batches += 1
            
            # Only store first N batches for metrics (saves ~12GB GPU memory!)
            if len(all_pred_deltas) < max_metric_batches:
                all_pred_deltas.append(outputs['delta_expression'].detach().cpu())
                all_actual_deltas.append(actual_delta.detach().cpu())
                all_pred_exprs.append(outputs['predicted_expression'].detach().cpu())
                all_target_exprs.append(target.detach().cpu())
            
            pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}'})
            
        if self.scheduler:
            self.scheduler.step()
            
        # Metrics (computed on CPU to avoid GPU OOM at end of epoch)
        all_pred_deltas = torch.cat(all_pred_deltas)  # Keep on CPU
        all_actual_deltas = torch.cat(all_actual_deltas)
        all_pred_exprs = torch.cat(all_pred_exprs)
        all_target_exprs = torch.cat(all_target_exprs)
        
        metrics = self.compute_metrics(all_pred_deltas, all_actual_deltas, all_pred_exprs, all_target_exprs)
        metrics['loss'] = total_loss / n_batches
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def train_epoch_multi_tf(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train using ALL TF activity changes together (CORRECT approach).
        
        This is biologically correct: cell transitions involve ALL TFs changing.
        """
        self.model.train()
        
        total_loss = 0.0
        all_pred_deltas = []
        all_actual_deltas = []
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}', leave=True)
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            source = batch['source_expression'].to(self.device)
            target = batch['target_expression'].to(self.device)
            tf_change = batch['tf_activity_change'].to(self.device)
            actual_delta = batch['delta_expression'].to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Use multi-TF forward
                outputs = self.model.forward_multi_tf(source, tf_change)
                
                # Loss focused on DELTA prediction
                pred_delta = outputs['delta_expression']
                
                # MSE loss
                mse_loss = F.mse_loss(pred_delta, actual_delta)
                
                # Correlation loss (maximize correlation)
                pred_c = pred_delta - pred_delta.mean(dim=1, keepdim=True)
                actual_c = actual_delta - actual_delta.mean(dim=1, keepdim=True)
                pred_std = torch.sqrt((pred_c ** 2).sum(dim=1, keepdim=True) + 1e-8)
                actual_std = torch.sqrt((actual_c ** 2).sum(dim=1, keepdim=True) + 1e-8)
                corr = (pred_c * actual_c).sum(dim=1) / (pred_std.squeeze() * actual_std.squeeze())
                corr_loss = (1 - corr.mean())
                
                # Direction loss
                significant = actual_delta.abs() > 0.1
                if significant.sum() > 0:
                    pred_sign = torch.sign(pred_delta[significant])
                    actual_sign = torch.sign(actual_delta[significant])
                    dir_loss = 1 - (pred_sign == actual_sign).float().mean()
                else:
                    dir_loss = torch.tensor(0.0, device=self.device)
                
                # GRN sparsity regularization (encourage sparse effects)
                grn_matrix = outputs['grn_matrix']
                sparsity_loss = grn_matrix.abs().mean() * 0.01
                
                total = mse_loss + 0.5 * corr_loss + 0.3 * dir_loss + sparsity_loss
            
            if self.scaler is not None:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += total.item()
            n_batches += 1
            
            if len(all_pred_deltas) < 50:
                all_pred_deltas.append(pred_delta.detach().cpu())
                all_actual_deltas.append(actual_delta.detach().cpu())
            
            pbar.set_postfix({
                'loss': f'{total_loss/n_batches:.4f}',
                'corr': f'{corr.mean().item():.3f}'
            })
        
        if self.scheduler:
            self.scheduler.step()
        
        # Compute epoch metrics
        all_pred_deltas = torch.cat(all_pred_deltas)
        all_actual_deltas = torch.cat(all_actual_deltas)
        
        # Delta PCC
        delta_pcc = np.corrcoef(
            all_pred_deltas.flatten().numpy(),
            all_actual_deltas.flatten().numpy()
        )[0, 1]
        
        # Direction accuracy
        significant = all_actual_deltas.abs() > 0.1
        dir_acc = (torch.sign(all_pred_deltas[significant]) == 
                   torch.sign(all_actual_deltas[significant])).float().mean().item()
        
        return {
            'loss': total_loss / n_batches,
            'delta_pcc': delta_pcc,
            'direction_accuracy': dir_acc,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        max_metric_batches = 50
        all_pred_deltas = []
        all_actual_deltas = []
        all_pred_exprs = []
        all_target_exprs = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(dataloader, desc="  [VAL]", leave=False, ncols=100)
            for batch in val_pbar:
                source = batch['source_expression'].to(self.device)
                target = batch['target_expression'].to(self.device)
                actual_delta = batch['delta_expression'].to(self.device)
                tf_idx = batch['tf_idx'].to(self.device)
                pert_type = batch['perturbation_type'].to(self.device)
                pert_strength = batch['perturbation_strength'].to(self.device)
                
                outputs = self.model(source, tf_idx, pert_type, pert_strength)
                
                losses = self.loss_fn(
                    pred_delta=outputs['delta_expression'],
                    actual_delta=actual_delta,
                    gene_effects=outputs['gene_effects'],
                    batch_chip_mask=outputs['batch_chip_mask']
                )
                
                total_loss += losses['total'].item()
                n_batches += 1
                
                # Update progress bar
                val_pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
                
                # Only store subset for metrics
                if len(all_pred_deltas) < max_metric_batches:
                    all_pred_deltas.append(outputs['delta_expression'].cpu())
                    all_actual_deltas.append(actual_delta.cpu())
                    all_pred_exprs.append(outputs['predicted_expression'].cpu())
                    all_target_exprs.append(target.cpu())
            
        # Keep on CPU to avoid GPU OOM
        all_pred_deltas = torch.cat(all_pred_deltas)
        all_actual_deltas = torch.cat(all_actual_deltas)
        all_pred_exprs = torch.cat(all_pred_exprs)
        all_target_exprs = torch.cat(all_target_exprs)
        
        metrics = self.compute_metrics(all_pred_deltas, all_actual_deltas, all_pred_exprs, all_target_exprs)
        metrics['loss'] = total_loss / n_batches
        
        return metrics
    
    @torch.no_grad()
    def validate_multi_tf(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate using multi-TF approach."""
        self.model.eval()
        
        all_pred_deltas = []
        all_actual_deltas = []
        total_loss = 0.0
        n_batches = 0
        
        val_pbar = tqdm(dataloader, desc="  [VAL]", leave=False, ncols=100)
        for batch in val_pbar:
            source = batch['source_expression'].to(self.device)
            tf_change = batch['tf_activity_change'].to(self.device)
            actual_delta = batch['delta_expression'].to(self.device)
            
            outputs = self.model.forward_multi_tf(source, tf_change)
            pred_delta = outputs['delta_expression']
            
            loss = F.mse_loss(pred_delta, actual_delta)
            total_loss += loss.item()
            n_batches += 1
            
            val_pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
            
            if len(all_pred_deltas) < 50:
                all_pred_deltas.append(pred_delta.cpu())
                all_actual_deltas.append(actual_delta.cpu())
        
        all_pred_deltas = torch.cat(all_pred_deltas)
        all_actual_deltas = torch.cat(all_actual_deltas)
        
        # Delta PCC
        delta_pcc = np.corrcoef(
            all_pred_deltas.flatten().numpy(),
            all_actual_deltas.flatten().numpy()
        )[0, 1]
        
        # Direction accuracy
        significant = all_actual_deltas.abs() > 0.1
        dir_acc = (torch.sign(all_pred_deltas[significant]) == 
                   torch.sign(all_actual_deltas[significant])).float().mean().item()
        
        return {
            'loss': total_loss / n_batches,
            'delta_pcc': delta_pcc,
            'direction_accuracy': dir_acc
        }
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        checkpoint_dir: str,
        val_every: int = 10,
        multi_tf: bool = False
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_scheduler(n_epochs)
        self.multi_tf = multi_tf
        
        print(f"\n{'='*80}")
        if multi_tf:
            print("  DELTA-FOCUSED GRN TRAINING v5 (MULTI-TF MODE)")
            print("  Using ALL TF activity changes together (correct approach!)")
        else:
            print("  DELTA-FOCUSED GRN TRAINING v5")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Learning rate: {self.lr}")
        
        if self.baseline_metrics:
            print(f"\n  BASELINES (model must beat these!):")
            print(f"    Source-Target PCC: {self.baseline_metrics['source_target_pcc']:.4f}")
            print(f"    Mean-Pred PCC:     {self.baseline_metrics['mean_pred_pcc']:.4f}")
            
        print(f"\n  PRIMARY METRIC: Delta_PCC (correlation of predicted vs actual CHANGE)")
        print(f"  This directly measures if we learned TF effects!")
        print(f"{'='*80}\n")
        
        best_delta_pcc = -1.0
        patience = 0
        max_patience = 25
        
        for epoch in range(1, n_epochs + 1):
            if self.multi_tf:
                train_metrics = self.train_epoch_multi_tf(train_loader, epoch)
            else:
                train_metrics = self.train_epoch(train_loader, epoch)
            
            # Print training progress every epoch
            leakage = self.model.chip_leakage.item()
            if self.multi_tf:
                print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} "
                      f"delta={train_metrics['delta_pcc']:.3f} "
                      f"dir_acc={train_metrics['direction_accuracy']:.3f}")
            else:
                print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} "
                      f"delta={train_metrics['delta_pcc']:.3f} "
                      f"expr={train_metrics['expr_pcc']:.3f} "
                      f"leak={leakage:.4f}")
            
            # Validation every val_every epochs (skip epoch 1 if val_every > 1)
            if epoch % val_every == 0:
                if self.multi_tf:
                    val_metrics = self.validate_multi_tf(val_loader)
                else:
                    val_metrics = self.validate(val_loader)
                
                self.history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                })
                
                delta_status = "LEARNING!" if val_metrics['delta_pcc'] > 0.1 else "low"
                if self.multi_tf:
                    print(f"  [VAL] loss={val_metrics['loss']:.4f} | "
                          f"Delta={val_metrics['delta_pcc']:.3f} ({delta_status}) "
                          f"Dir={val_metrics['direction_accuracy']:.3f}")
                else:
                    print(f"  [VAL] loss={val_metrics['loss']:.4f} | "
                          f"Delta={val_metrics['delta_pcc']:.3f} ({delta_status}) "
                          f"Dir={val_metrics['direction_acc']:.3f} "
                          f"Expr={val_metrics['expr_pcc']:.3f}")
                
                if val_metrics['delta_pcc'] > best_delta_pcc:
                    best_delta_pcc = val_metrics['delta_pcc']
                    patience = 0
                    self.save_checkpoint(checkpoint_dir / 'best_model.pt', val_metrics)
                    print(f"  >> New best! Delta_PCC={best_delta_pcc:.4f}")
                else:
                    patience += 1
                    
                if patience >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
                
        self.save_checkpoint(checkpoint_dir / 'final_model.pt', val_metrics)
        
        # Save GRN matrix
        grn_matrix = self.model.get_grn_matrix().detach().cpu().numpy()
        np.save(checkpoint_dir / 'grn_matrix.npy', grn_matrix)
        
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"\n{'='*80}")
        print(f"  Training complete!")
        print(f"  Best Delta_PCC: {best_delta_pcc:.4f}")
        if best_delta_pcc > 0.2:
            print(f"  Model IS learning TF effects!")
        else:
            print(f"  WARNING: Low Delta_PCC - model may not be learning real TF effects")
        print(f"{'='*80}\n")
        
    def save_checkpoint(self, path: Path, metrics: Dict):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'metrics': metrics
        }, path)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_singlecell_data(data_path: str, celltype_key: str):
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy required")
        
    print(f"Loading {data_path}...")
    adata = sc.read_h5ad(data_path)
    
    from scipy import sparse
    if sparse.issparse(adata.X):
        expression = adata.X.toarray().astype(np.float32)
    else:
        expression = np.array(adata.X, dtype=np.float32)
        
    gene_names = list(adata.var_names)
    
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"'{celltype_key}' not found")
        
    cell_types = adata.obs[celltype_key].astype(str).values
    unique_types = sorted(set(cell_types))
    
    # Check if data is pre-aggregated pseudo-bulk (1 row per cell type)
    if len(unique_types) == len(cell_types):
        # Already pseudo-bulk - each row is a cell type mean
        print(f"  Detected pre-aggregated pseudo-bulk data")
        expression_means = expression
        valid_types = list(cell_types)  # Use original order
    else:
        # Per-cell data - aggregate to pseudo-bulk
        means = []
        valid_types = []
        
        for ct in unique_types:
            mask = cell_types == ct
            if mask.sum() >= 10:
                means.append(expression[mask].mean(axis=0))
                valid_types.append(ct)
                
        expression_means = np.array(means, dtype=np.float32)
    
    print(f"  {len(valid_types)} cell types")
    print(f"  Expression: {expression_means.shape}")
    
    return expression_means, gene_names, valid_types


def load_tf_binding(binding_dir: str, gene_names: List[str]):
    binding_dir = Path(binding_dir)
    
    binding_matrix = np.load(binding_dir / 'binding_matrix.npy')
    tf_tf_adj = np.load(binding_dir / 'tf_tf_adjacency.npy')
    
    with open(binding_dir / 'mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        
    binding_genes = [str(g) for g in mappings['gene_names']]
    tf_names = mappings['tf_names']
    
    binding_gene_map = {g.upper(): i for i, g in enumerate(binding_genes)}
    
    aligned = np.zeros((len(gene_names), len(tf_names)), dtype=np.float32)
    matched = 0
    
    for i, gene in enumerate(gene_names):
        if str(gene).upper() in binding_gene_map:
            aligned[i] = binding_matrix[binding_gene_map[str(gene).upper()]]
            matched += 1
            
    print(f"  Aligned {matched}/{len(gene_names)} genes")
    
    return aligned, tf_names, tf_tf_adj


def load_tf_specificity_weights(binding_dir: str, n_tfs: int) -> Optional[np.ndarray]:
    """
    Load TF specificity weights based on JASPAR motif analysis.
    
    These weights are computed from:
    1. JASPAR motif Information Content (higher IC = more specific motif)
    2. TF-Gene binding sparsity (fewer targets = more specific)
    
    Weights are in range [0.5, 1.5], mean ~1.24
    - High weight (~1.4-1.5): TFs with unique, specific motifs (MLXIP, OLIG2)
    - Low weight (~0.93): Promiscuous TFs binding many genes (CRTC2, RAG2)
    
    These weights modulate TF effect magnitude - NOT too aggressive.
    """
    specificity_dir = Path(binding_dir).parent / 'tf_specificity'
    
    # Prefer PPI-combined weights (includes both JASPAR and cofactor PPI data)
    ppi_weights_path = specificity_dir / 'tf_specificity_weights_ppi.npy'
    jaspar_weights_path = specificity_dir / 'tf_specificity_weights.npy'
    
    if ppi_weights_path.exists():
        weights_path = ppi_weights_path
        print(f"  Using combined JASPAR+PPI weights")
    elif jaspar_weights_path.exists():
        weights_path = jaspar_weights_path
        print(f"  Using JASPAR-only weights")
    else:
        print(f"  [INFO] TF specificity weights not found")
        print(f"         Run 'python scripts/analyze_jaspar_match.py' and 'python scripts/analyze_ppi_cofactors.py'")
        print(f"         Using uniform weights (1.0) for all TFs")
        return None
    
    weights = np.load(weights_path)
    
    if len(weights) != n_tfs:
        print(f"  [WARN] TF specificity weights size mismatch: {len(weights)} vs {n_tfs}")
        return None
        
    print(f"  Loaded TF specificity weights: range=[{weights.min():.3f}, {weights.max():.3f}], mean={weights.mean():.3f}")
    
    return weights


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train GRN v5 - Delta-Focused')
    parser.add_argument('--singlecell', type=str, required=True)
    parser.add_argument('--tf-binding', type=str, required=True)
    parser.add_argument('--celltype-key', type=str, default='celltype_subset')
    parser.add_argument('--output', type=str, default='results/grn_model_v5')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--min-activity-diff', type=float, default=0.1)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--samples-per-epoch', type=int, default=10000000000000000,
                        help='Max samples per epoch (random subset)')
    parser.add_argument('--val-every', type=int, default=10,
                        help='Run validation every N epochs')
    parser.add_argument('--ppi', type=str, default=None,
                        help='Path to PPI data dir (from preprocess_ppi_rosetta.py)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Ablation flags for controlled experiments
    parser.add_argument('--no-jaspar', action='store_true',
                        help='Disable JASPAR specificity weights (ablation)')
    parser.add_argument('--no-cascade', action='store_true',
                        help='Disable TF-TF cascade (ablation)')
    parser.add_argument('--no-chip-reg', action='store_true',
                        help='Disable ChIP L1 regularization (ablation)')
    parser.add_argument('--multi-tf', action='store_true',
                        help='Use multi-TF training (correct approach: all TFs together)')
    parser.add_argument('--no-context-gate', action='store_true',
                        help='Disable context gating (ablation)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    expression_means, gene_names, celltype_names = load_singlecell_data(
        args.singlecell, args.celltype_key
    )
    n_celltypes = len(celltype_names)
    
    # Compute baseline (CRITICAL: what does model need to beat?)
    print("\nComputing baselines...")
    baseline_metrics = compute_baseline_metrics(expression_means)
    print(f"  Source-Target PCC: {baseline_metrics['source_target_pcc']:.4f} (range: {baseline_metrics['min_pair_pcc']:.4f}-{baseline_metrics['max_pair_pcc']:.4f})")
    print(f"  Mean-Pred PCC:     {baseline_metrics['mean_pred_pcc']:.4f}")
    print(f"  >>> Model Expr_PCC must significantly exceed {baseline_metrics['source_target_pcc']:.4f} to be learning!")
    
    # Load TF binding
    print(f"\nLoading TF binding from {args.tf_binding}...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(args.tf_binding, gene_names)
    n_tfs = len(tf_names)
    
    # TF activity
    print("\nComputing TF activity...")
    tf_activity = compute_tf_activity_linger(expression_means, binding_matrix)
    tf_states = classify_tf_states(tf_activity)
    
    active_per_type = (tf_states > 0).sum(axis=1)
    print(f"  TFs active per cell type: {active_per_type.mean():.0f} avg")
    
    # Generate cell-type pairs with HOLDOUT
    all_pairs = [(i, j) for i in range(n_celltypes) for j in range(n_celltypes) if i != j]
    random.shuffle(all_pairs)
    
    n_val_pairs = max(int(len(all_pairs) * args.val_fraction), 2)
    train_pairs = all_pairs[n_val_pairs:]
    val_pairs = all_pairs[:n_val_pairs]
    
    print(f"\nCell-type pair split (HOLDOUT validation):")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)} (NEVER seen during training!)")
    
    # Create datasets
    use_multi_tf = getattr(args, 'multi_tf', False)
    
    if use_multi_tf:
        print("\n>>> USING MULTI-TF TRAINING (all TFs together - CORRECT approach)")
        train_dataset = MultiTFDataset(
            expression_means, tf_activity, celltype_names,
            allowed_pairs=train_pairs,
            augment_factor=10,  # 10x augmentation for training
            noise_std=0.05     # Small noise for regularization
        )
        val_dataset = MultiTFDataset(
            expression_means, tf_activity, celltype_names,
            allowed_pairs=val_pairs,
            augment_factor=1,   # NO augmentation for validation
            noise_std=0.0       # NO noise for validation
        )
    else:
        train_dataset = VirtualPerturbationDatasetV5(
            expression_means, tf_activity, celltype_names,
            allowed_pairs=train_pairs,
            min_activity_diff=args.min_activity_diff
        )
        val_dataset = VirtualPerturbationDatasetV5(
            expression_means, tf_activity, celltype_names,
            allowed_pairs=val_pairs,
            min_activity_diff=args.min_activity_diff
        )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Use random sampler to limit samples per epoch (faster training)
    from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
    
    train_samples_per_epoch = min(args.samples_per_epoch, len(train_dataset))
    val_samples_per_epoch = min(args.samples_per_epoch // 4, len(val_dataset))
    
    print(f"Samples per epoch: {train_samples_per_epoch} train, {val_samples_per_epoch} val")
    
    if use_multi_tf:
        # Multi-TF mode: simple random sampling (no TF balancing needed)
        train_sampler = RandomSampler(train_dataset, replacement=True,
                                      num_samples=train_samples_per_epoch)
    else:
        # Single-TF mode: TF-balanced weighted sampling (so frequent TFs don't dominate)
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=train_samples_per_epoch,
            replacement=True
        )
    
    # Validation sampler (same for both modes)
    val_sampler = RandomSampler(val_dataset, replacement=True, 
                                 num_samples=min(val_samples_per_epoch, len(val_dataset)))
    
    # num_workers: 4 for Linux/WSL, 0 for Windows (multiprocessing issues)
    import platform
    num_workers = 4 if platform.system() != 'Windows' else 0
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            sampler=val_sampler, num_workers=num_workers,
                            pin_memory=True)
    
    # Load TF specificity weights (JASPAR motif analysis)
    if args.no_jaspar:
        print("  [ABLATION] JASPAR specificity weights DISABLED")
        tf_specificity_weights = None
    else:
        tf_specificity_weights = load_tf_specificity_weights(args.tf_binding, n_tfs)
    
    # Model (with ablation support)
    model = DeltaGRNModel(
        n_genes=len(gene_names),
        n_tfs=n_tfs,
        hidden_dim=args.hidden_dim,
        n_cascade_layers=0 if args.no_cascade else 2,  # Ablation: disable cascade
        use_context_gating=not args.no_context_gate     # Ablation: disable context gate
    )
    
    # Load PPI cofactor gate data if provided
    cof_to_tf_adj = None
    cofactor_gene_indices = None
    
    if args.ppi:
        ppi_dir = Path(args.ppi)
        cof_to_tf_path = ppi_dir / 'cof_to_tf_adj.npy'
        cof_idx_path = ppi_dir / 'cofactor_gene_indices.npy'
        tf_tf_ppi_path = ppi_dir / 'tf_tf_ppi.npy'
        
        if cof_to_tf_path.exists() and cof_idx_path.exists():
            cof_to_tf_adj = np.load(cof_to_tf_path)
            cofactor_gene_indices = np.load(cof_idx_path)
            print(f"  Loaded PPI cofactor gate: {cof_to_tf_adj.shape[0]} cofactors -> {cof_to_tf_adj.shape[1]} TFs")
        else:
            print(f"  [WARN] PPI cofactor files not found in {ppi_dir}, skipping cofactor gate")
        
        # OPTION B: Blend TF-TF PPI into cascade adjacency
        if tf_tf_ppi_path.exists():
            tf_tf_ppi = np.load(tf_tf_ppi_path)
            # Combine with regulatory TF-TF: w_reg * reg + w_ppi * ppi
            w_reg, w_ppi = 0.8, 0.2  # PPI is supplementary, not dominant
            tf_tf_adj_combined = w_reg * tf_tf_adj + w_ppi * tf_tf_ppi
            # Re-normalize rows
            row_sum = tf_tf_adj_combined.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            tf_tf_adj = tf_tf_adj_combined / row_sum
            print(f"  Blended TF-TF PPI into cascade (w_ppi={w_ppi})")
    
    model.set_priors(
        chip_binding=torch.FloatTensor(binding_matrix),
        tf_tf_adjacency=torch.FloatTensor(tf_tf_adj),
        tf_specificity_weights=torch.FloatTensor(tf_specificity_weights) if tf_specificity_weights is not None else None,
        cofactor_tf_adjacency=torch.FloatTensor(cof_to_tf_adj) if cof_to_tf_adj is not None else None,
        cofactor_gene_indices=torch.LongTensor(cofactor_gene_indices) if cofactor_gene_indices is not None else None
    )
    
    # ==========================================================================
    # SAVE METADATA FIRST (so analysis works even if training is interrupted!)
    # ==========================================================================
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving metadata files...")
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False)
    pd.DataFrame({'celltype': celltype_names}).to_csv(output_dir / 'celltype_names.csv', index=False)
    
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'tf_states.npy', tf_states)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    np.save(output_dir / 'expression_means.npy', expression_means)
    
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
        
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model.config, f, indent=2)
    
    print(f"  Saved: gene_names.csv, tf_names.csv, celltype_names.csv")
    print(f"  Saved: tf_activity.npy, tf_states.npy, binding_matrix.npy")
    print(f"  Saved: model_config.json, baseline_metrics.json")
    
    # ==========================================================================
    # TRAIN
    # ==========================================================================
    chip_reg = 0.0 if args.no_chip_reg else 0.2
    if args.no_chip_reg:
        print("  [ABLATION] ChIP regularization DISABLED")
    
    trainer = DeltaGRNTrainer(model, device, args.lr, chip_reg_weight=chip_reg)
    trainer.set_baseline(baseline_metrics)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        checkpoint_dir=args.output,
        val_every=args.val_every,
        multi_tf=use_multi_tf
    )
    
    print(f"\nTraining complete! All outputs in {output_dir}")


if __name__ == '__main__':
    main()


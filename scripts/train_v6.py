#!/usr/bin/env python3
"""
================================================================================
Train GRN Model v6 - FULLY INTEGRATED WITH SOPHISTICATED MODELS
================================================================================

This version PROPERLY USES the sophisticated model components from models/:
- MonsterGRNModel with multi-network GCN
- TFRegulatoryEncoder with cascade GCN  
- CellStateEncoder with transformer
- TransformerFusionBlock for multi-modal fusion
- PerturbationSimulator for in-silico perturbations

Key features retained from v5:
- Cell-type PAIR holdout validation
- DELTA prediction focus
- JASPAR/PPI integration
- Multi-TF training mode

Usage:
    python scripts/train_v6.py \
        --singlecell data/sclc/preprocessed.h5ad \
        --tf-binding data/tf_binding/ \
        --celltype-key sclc_subcluster \
        --output results/sclc_grn_v6/ \
        --ppi data/ppi/ \
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from sophisticated models - THIS IS THE KEY CHANGE!
from models.monster_grn import MonsterGRNModel, create_monster_grn
from models.tf_encoder import TFRegulatoryEncoder, TFCascadeGCN, ChIPAtlasLoader
from models.cell_encoder import CellStateEncoder, SingleCellDataEncoder
from modules.perturbation import LINGERStyleGRN, PerturbationPredictor

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import dgl
    from dgl.nn import GraphConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


# =============================================================================
# BASELINE COMPUTATION (from v5)
# =============================================================================

def compute_baseline_metrics(expression_means: np.ndarray) -> Dict[str, float]:
    """Compute baseline metrics to understand what the model needs to beat."""
    n_types = expression_means.shape[0]
    
    source_target_pccs = []
    for i in range(n_types):
        for j in range(n_types):
            if i != j:
                r = np.corrcoef(expression_means[i], expression_means[j])[0, 1]
                source_target_pccs.append(r)
                
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
    """Compute TF activity using LINGER's ChIP-weighted approach."""
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
# DATASETS (from v5)
# =============================================================================

class VirtualPerturbationDatasetV5(Dataset):
    """Virtual perturbation dataset with PROPER cell-type pair holdout."""
    
    def __init__(
        self,
        expression_means: np.ndarray,
        tf_activity: np.ndarray,
        celltype_names: List[str],
        allowed_pairs: List[Tuple[int, int]],
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
        
        if add_hard_negatives and hard_negatives:
            n_hard = int(len(samples) * hard_neg_frac)
            np.random.shuffle(hard_negatives)
            samples.extend(hard_negatives[:n_hard])
            
        return samples
    
    def _compute_balanced_weights(self) -> np.ndarray:
        """Compute per-sample weights so each TF is equally represented."""
        tf_counts = np.zeros(self.n_tfs)
        for s in self.samples:
            tf_counts[s['tf_idx']] += 1
        
        tf_weights = np.zeros(self.n_tfs)
        for i in range(self.n_tfs):
            if tf_counts[i] > 0:
                tf_weights[i] = 1.0 / tf_counts[i]
        
        tf_weights = tf_weights / tf_weights.sum() * len(tf_weights)
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
            'delta_expression': tgt_expr - src_expr,
            'tf_idx': torch.tensor(sample['tf_idx'], dtype=torch.long),
            'perturbation_type': torch.tensor(sample['perturbation_type'], dtype=torch.float),
            'perturbation_strength': torch.tensor(sample['perturbation_strength'], dtype=torch.float),
        }


class MultiTFDataset(Dataset):
    """Dataset that uses ALL TF activity changes together - CORRECT approach."""
    
    def __init__(
        self,
        expression_means: np.ndarray,
        tf_activity: np.ndarray,
        celltype_names: List[str],
        allowed_pairs: List[Tuple[int, int]],
        augment_factor: int = 10,
        noise_std: float = 0.05
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
        
        self.samples = []
        for src_idx, tgt_idx in self.base_pairs:
            self.samples.append((src_idx, tgt_idx, 1.0))
            for i in range(1, augment_factor):
                alpha = i / augment_factor
                self.samples.append((src_idx, tgt_idx, alpha))
        
        print(f"  Multi-TF augmented: {len(self.base_pairs)} pairs × {augment_factor} = {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src_idx, tgt_idx, alpha = self.samples[idx]
        
        source_expr = self.expression[src_idx]
        full_target = self.expression[tgt_idx]
        target_expr = source_expr + alpha * (full_target - source_expr)
        
        full_tf_change = self.tf_activity[tgt_idx] - self.tf_activity[src_idx]
        tf_activity_change = alpha * full_tf_change
        
        delta_expr = target_expr - source_expr
        
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
# INTEGRATED GRN MODEL - Uses MonsterGRNModel Components!
# =============================================================================

class IntegratedGRNModel(nn.Module):
    """
    Properly integrated GRN model that uses ALL sophisticated components.
    
    Uses:
    - TFRegulatoryEncoder from models/tf_encoder.py
    - CellStateEncoder from models/cell_encoder.py
    - TransformerFusionBlock from models/monster_grn.py
    - PerturbationSimulator from models/monster_grn.py
    
    Trained with delta prediction (from v5 approach).
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
        use_context_gating: bool = True
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        self.use_context_gating = use_context_gating
        
        # Store config
        self.config = {
            'n_genes': n_genes,
            'n_tfs': n_tfs,
            'n_batches': n_batches,
            'hidden_dim': hidden_dim,
            'num_gcn_layers': num_gcn_layers,
            'num_transformer_layers': num_transformer_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'use_context_gating': use_context_gating
        }
        
        # ============================================================
        # Component 1: Cell State Encoder (from models/cell_encoder.py)
        # ============================================================
        self.cell_encoder = SingleCellDataEncoder(
            n_genes=n_genes,
            n_tfs=n_tfs,
            n_batches=n_batches,
            hidden_dim=hidden_dim,
            n_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # ============================================================
        # Component 2: TF Regulatory Encoder (from models/tf_encoder.py)
        # ============================================================
        self.tf_encoder = TFRegulatoryEncoder(
            n_tfs=n_tfs,
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            num_attention_heads=num_heads,
            dropout=dropout
        )
        
        # ============================================================
        # Component 3: Delta Expression Decoder
        # ============================================================
        # Multi-TF mode: takes all TF activity changes -> predicts gene delta
        self.delta_decoder = nn.Sequential(
            nn.Linear(n_tfs + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_genes)
        )
        
        # ============================================================
        # Component 4: Direct TF->Gene Effects (sparse, ChIP-initialized)
        # ============================================================
        self.direct_effects = nn.Parameter(torch.zeros(n_genes, n_tfs))
        
        # Learnable ChIP mask leakage
        self._leakage_raw = nn.Parameter(torch.tensor(0.0))
        
        # ============================================================
        # Component 5: Context Gating (FiLM-style modulation)
        # ============================================================
        if use_context_gating:
            context_dim = 128
            self.context_encoder = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, context_dim)
            )
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, n_genes),
                nn.Sigmoid()
            )
            self.context_shift = nn.Sequential(
                nn.Linear(context_dim, n_genes),
                nn.Tanh()
            )
        
        # Buffers - these get saved in state_dict
        self.register_buffer('chip_mask', torch.zeros(n_genes, n_tfs))
        self.register_buffer('tf_specificity_weights', torch.ones(n_tfs))
        # Initialize tf_tf_adjacency with identity matrix (will be set properly in set_priors)
        # Cannot use None because PyTorch won't save None buffers to state_dict!
        self.register_buffer('tf_tf_adjacency', torch.eye(n_tfs))
        
        # Cache
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
        """Initialize from biological priors."""
        device = self.direct_effects.device
        self.chip_mask = chip_binding.to(device)
        
        with torch.no_grad():
            self.direct_effects.data = chip_binding.float() * 0.01
            
        # Store TF-TF adjacency as buffer for use in forward passes
        if tf_tf_adjacency is not None:
            # Row-normalize the adjacency matrix
            adj = tf_tf_adjacency.clone().to(device)
            # Add self-loops
            adj = adj + torch.eye(adj.size(0), device=device)
            # Row normalize
            row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
            adj = adj / row_sum
            # Use copy_ to update the registered buffer in-place (preserves buffer status!)
            self.tf_tf_adjacency.copy_(adj)
            self.config['uses_tf_cascade'] = True
            print(f"  [TF Cascade] {adj.size(0)} TFs, {(tf_tf_adjacency > 0).sum().item()} edges (row-normalized)")
        # If tf_tf_adjacency is None, keep the default identity matrix
            
        if tf_specificity_weights is not None:
            self.tf_specificity_weights = tf_specificity_weights.to(device)
            self.config['uses_tf_specificity'] = True
            
        if cofactor_tf_adjacency is not None and cofactor_gene_indices is not None:
            self.register_buffer('cof_to_tf_adj', cofactor_tf_adjacency.to(device))
            self.register_buffer('cofactor_gene_indices', cofactor_gene_indices.to(device))
            self.config['uses_cofactor_gate'] = True
            print(f"  [Cofactor Gate] {len(cofactor_gene_indices)} cofactors -> {self.n_tfs} TFs")
        else:
            self.cof_to_tf_adj = None
            self.cofactor_gene_indices = None
    
    @property
    def chip_leakage(self) -> torch.Tensor:
        """Learnable ChIP mask leakage in range [0.005, 0.2]."""
        return 0.005 + 0.195 * torch.sigmoid(self._leakage_raw)
    
    def get_grn_matrix(self, epoch: int = -1, for_prediction: bool = False) -> torch.Tensor:
        """
        Get full TF→Gene effect matrix.
        
        Args:
            epoch: Epoch number for caching (-1 = no cache)
            for_prediction: If True, apply TF importance modulation (for runtime predictions)
                           If False, return raw GRN weights (for saving/analysis)
        
        The GRN matrix represents learned TF→Gene regulatory effects.
        Formula: direct_effects * tf_specificity_weights * (chip_mask + leakage)
        
        NOTE: tf_importance from the cascade GCN is ONLY applied during forward pass
        predictions, NOT to the saved GRN matrix. This is because tf_importance
        is softmaxed (sums to 1.0), which would destroy the magnitude of effects
        if applied to the GRN directly (1480 TFs → each gets ~0.0007 weight).
        """
        if epoch >= 0 and epoch == self._grn_cache_epoch and self._grn_cache is not None:
            return self._grn_cache
        
        # GRN matrix = direct_effects * tf_specificity * (chip_mask + leakage)
        grn_matrix = self.direct_effects * self.tf_specificity_weights.unsqueeze(0)
        grn_matrix = grn_matrix * (self.chip_mask + self.chip_leakage)
        
        # IMPORTANT: Do NOT multiply by softmaxed tf_importance here!
        # That destroys magnitudes. tf_importance is used in predict_delta_multi_tf
        # to modulate the prediction, not to scale the GRN itself.
        
        if epoch >= 0:
            self._grn_cache = grn_matrix.detach()
            self._grn_cache_epoch = epoch
            
        return grn_matrix
    
    def forward(
        self,
        expression: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for expression prediction/reconstruction.
        
        Args:
            expression: Gene expression (batch, n_genes)
            batch_ids: Optional batch IDs
            return_embeddings: Return intermediate embeddings
        """
        # Cell state encoding
        cell_outputs = self.cell_encoder(expression, batch_ids)
        cell_emb = cell_outputs['cell_embedding']  # (batch, hidden)
        tf_activity = cell_outputs['tf_activity']   # (batch, n_tfs)
        
        # TF regulatory encoding (with cascade GCN!)
        tf_outputs = self.tf_encoder(self.chip_mask, self.tf_tf_adjacency)
        tf_features = tf_outputs['tf_features']  # (n_tfs, hidden)
        tf_importance = tf_outputs.get('tf_importance')  # (n_tfs,)
        
        # Reconstruct expression
        # Combine cell embedding with TF activity
        combined = torch.cat([cell_emb, tf_activity], dim=-1)
        # This is simplified - full model would use TransformerFusion here
        reconstructed = self.delta_decoder(combined)
        
        outputs = {
            'reconstructed': reconstructed,
            'predicted_expression': reconstructed,
            'tf_activity': tf_activity,
            'cell_embedding': cell_emb
        }
        
        if tf_importance is not None:
            outputs['tf_importance'] = tf_importance
            
        if return_embeddings:
            outputs['tf_features'] = tf_features
            
        return outputs
    
    def predict_delta_multi_tf(
        self,
        tf_activity_change: torch.Tensor,
        source_expression: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expression CHANGE from ALL TF activity changes.
        
        Args:
            tf_activity_change: (batch, n_tfs) - change in activity for ALL TFs
            source_expression: (batch, n_genes) - source cell expression
            
        Returns:
            predicted_delta: (batch, n_genes)
            grn_matrix: (n_genes, n_tfs)
        """
        batch_size = tf_activity_change.size(0)
        device = tf_activity_change.device
        
        # Get GRN matrix
        grn_matrix = self.get_grn_matrix()  # (n_genes, n_tfs)
        
        # Basic delta = TF_change @ GRN
        delta_basic = torch.mm(tf_activity_change, grn_matrix.T)  # (batch, n_genes)
        
        # Context gating
        if self.use_context_gating and source_expression is not None:
            # Encode cell state
            cell_outputs = self.cell_encoder(source_expression)
            cell_emb = cell_outputs['cell_embedding']
            
            # Combine TF change with cell embedding for context-aware delta
            combined = torch.cat([tf_activity_change, cell_emb], dim=-1)
            delta_neural = self.delta_decoder(combined)
            
            # FiLM modulation
            context = self.context_encoder(source_expression)
            gamma = self.context_gate(context)
            beta = self.context_shift(context) * 0.1
            
            # Combine: basic GRN effect + neural refinement, modulated by context
            delta = gamma * (delta_basic + 0.1 * delta_neural) + beta
        else:
            delta = delta_basic
            
        # Cofactor gate
        if hasattr(self, 'cof_to_tf_adj') and self.cof_to_tf_adj is not None and source_expression is not None:
            cofactor_expr = source_expression[:, self.cofactor_gene_indices]
            cof_influence = torch.mm(cofactor_expr, self.cof_to_tf_adj)
            cof_norm = cof_influence / (cof_influence.abs().mean(dim=1, keepdim=True) + 1e-6)
            cof_gate = 1.0 + 0.2 * torch.tanh(cof_norm)
            
            # Weight TF effects by cofactor availability
            tf_weighted = tf_activity_change * cof_gate
            delta_cof = torch.mm(tf_weighted, grn_matrix.T)
            delta = 0.5 * delta + 0.5 * delta_cof
            
        return delta, grn_matrix
    
    def predict_delta(
        self,
        tf_idx: torch.Tensor,
        perturbation_type: torch.Tensor,
        perturbation_strength: torch.Tensor,
        source_expression: torch.Tensor = None,
        return_effects: bool = False
    ) -> torch.Tensor:
        """Single-TF perturbation prediction (for backward compatibility)."""
        batch_size = tf_idx.size(0)
        
        # Get GRN matrix
        grn_matrix = self.get_grn_matrix()
        
        # Get effects for specified TFs
        gene_effects = grn_matrix[:, tf_idx].T  # (batch, n_genes)
        
        # Context gating
        if self.use_context_gating and source_expression is not None:
            context = self.context_encoder(source_expression)
            gamma = self.context_gate(context)
            beta = self.context_shift(context) * 0.1
            gene_effects = gamma * gene_effects + beta
        
        # Scale by perturbation
        scaling = perturbation_type.unsqueeze(1) * perturbation_strength.unsqueeze(1)
        delta = gene_effects * scaling
        
        if return_effects:
            batch_chip_mask = self.chip_mask[:, tf_idx].T
            return delta, gene_effects, batch_chip_mask
        return delta
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# LOSS FUNCTION (from v5)
# =============================================================================

class DeltaFocusedLoss(nn.Module):
    """Loss function focused on DELTA prediction."""
    
    def __init__(
        self,
        ranking_weight: float = 0.3,
        direction_weight: float = 0.2,
        chip_reg_weight: float = 0.1,
        use_chip_reg: bool = True
    ):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.direction_weight = direction_weight
        self.chip_reg_weight = chip_reg_weight
        self.use_chip_reg = use_chip_reg
        
    def forward(
        self,
        pred_delta: torch.Tensor,
        actual_delta: torch.Tensor,
        gene_effects: Optional[torch.Tensor] = None,
        chip_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute loss components."""
        # MSE on delta
        mse_loss = F.mse_loss(pred_delta, actual_delta)
        
        # Ranking loss (preserve relative order)
        ranking_loss = self._ranking_loss(pred_delta, actual_delta)
        
        # Direction loss (sign agreement)
        direction_loss = self._direction_loss(pred_delta, actual_delta)
        
        # ChIP regularization
        chip_loss = torch.tensor(0.0, device=pred_delta.device)
        if self.use_chip_reg and gene_effects is not None and chip_mask is not None:
            chip_loss = self._chip_regularization(gene_effects, chip_mask)
        
        total = mse_loss + self.ranking_weight * ranking_loss + \
                self.direction_weight * direction_loss + \
                self.chip_reg_weight * chip_loss
        
        return {
            'total': total,
            'mse': mse_loss,
            'ranking': ranking_loss,
            'direction': direction_loss,
            'chip': chip_loss
        }
    
    def _ranking_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Margin ranking loss for preserving relative order."""
        n = pred.size(1)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
            
        # Sample random pairs
        n_pairs = min(1000, n * (n - 1) // 2)
        idx1 = torch.randint(0, n, (n_pairs,), device=pred.device)
        idx2 = torch.randint(0, n, (n_pairs,), device=pred.device)
        
        pred_diff = pred[:, idx1] - pred[:, idx2]
        actual_diff = actual[:, idx1] - actual[:, idx2]
        
        target = torch.sign(actual_diff)
        return F.margin_ranking_loss(
            pred_diff.flatten(), torch.zeros_like(pred_diff.flatten()),
            target.flatten(), margin=0.1
        )
    
    def _direction_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy for sign agreement."""
        significant_mask = actual.abs() > 0.1
        if not significant_mask.any():
            return torch.tensor(0.0, device=pred.device)
            
        pred_probs = torch.sigmoid(pred[significant_mask] * 3)
        actual_binary = (actual[significant_mask] > 0).float()
        return F.binary_cross_entropy_with_logits(pred[significant_mask] * 3, actual_binary)
    
    def _chip_regularization(self, effects: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Penalize effects where ChIP says no binding."""
        no_bind_mask = (mask < 0.5).float()
        no_bind_effects = effects.abs() * no_bind_mask
        return no_bind_effects.mean()


# =============================================================================
# TRAINER
# =============================================================================

class IntegratedGRNTrainer:
    """Trainer for IntegratedGRNModel."""
    
    def __init__(
        self,
        model: IntegratedGRNModel,
        device: torch.device,
        learning_rate: float = 1e-3,
        use_amp: bool = True,
        use_chip_reg: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.loss_fn = DeltaFocusedLoss(use_chip_reg=use_chip_reg)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
    def train_epoch_multi_tf(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Training epoch for multi-TF mode."""
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        all_pred_deltas = []
        all_actual_deltas = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}")
        
        for batch in pbar:
            # Move to device
            tf_change = batch['tf_activity_change'].to(self.device)
            source_expr = batch['source_expression'].to(self.device)
            actual_delta = batch['delta_expression'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred_delta, grn_matrix = self.model.predict_delta_multi_tf(
                    tf_change, source_expr
                )
                
                losses = self.loss_fn(pred_delta, actual_delta)
                
            if self.use_amp:
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
            total_mse += losses['mse'].item()
            
            # Collect for metrics (on CPU)
            if len(all_pred_deltas) < 50:
                all_pred_deltas.append(pred_delta.detach().cpu())
                all_actual_deltas.append(actual_delta.detach().cpu())
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'corr': f"{self._batch_correlation(pred_delta, actual_delta):.3f}"
            })
        
        # Compute metrics
        n_batches = len(dataloader)
        metrics = {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches
        }
        
        if all_pred_deltas:
            pred_all = torch.cat(all_pred_deltas)
            actual_all = torch.cat(all_actual_deltas)
            metrics['delta_pcc'] = self._compute_pcc(pred_all, actual_all)
            metrics['dir_acc'] = self._compute_direction_accuracy(pred_all, actual_all)
        
        return metrics
    
    def validate_multi_tf(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validation for multi-TF mode."""
        self.model.eval()
        
        all_pred_deltas = []
        all_actual_deltas = []
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="  [VAL]")
        
        with torch.no_grad():
            for batch in pbar:
                tf_change = batch['tf_activity_change'].to(self.device)
                source_expr = batch['source_expression'].to(self.device)
                actual_delta = batch['delta_expression'].to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred_delta, _ = self.model.predict_delta_multi_tf(
                        tf_change, source_expr
                    )
                    losses = self.loss_fn(pred_delta, actual_delta)
                
                total_loss += losses['total'].item()
                all_pred_deltas.append(pred_delta.cpu())
                all_actual_deltas.append(actual_delta.cpu())
                
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
        pred_all = torch.cat(all_pred_deltas)
        actual_all = torch.cat(all_actual_deltas)
        
        delta_pcc = self._compute_pcc(pred_all, actual_all)
        dir_acc = self._compute_direction_accuracy(pred_all, actual_all)
        
        return {
            'loss': total_loss / len(dataloader),
            'delta_pcc': delta_pcc,
            'dir_acc': dir_acc
        }
    
    def _batch_correlation(self, pred: torch.Tensor, actual: torch.Tensor) -> float:
        """Quick correlation for progress bar."""
        pred_flat = pred.flatten()
        actual_flat = actual.flatten()
        
        pred_mean = pred_flat.mean()
        actual_mean = actual_flat.mean()
        
        cov = ((pred_flat - pred_mean) * (actual_flat - actual_mean)).mean()
        pred_std = pred_flat.std()
        actual_std = actual_flat.std()
        
        if pred_std > 0 and actual_std > 0:
            return (cov / (pred_std * actual_std)).item()
        return 0.0
    
    def _compute_pcc(self, pred: torch.Tensor, actual: torch.Tensor) -> float:
        """Compute Pearson correlation."""
        pred_np = pred.numpy().flatten()
        actual_np = actual.numpy().flatten()
        
        if len(pred_np) == 0:
            return 0.0
            
        r = np.corrcoef(pred_np, actual_np)[0, 1]
        return float(r) if not np.isnan(r) else 0.0
    
    def _compute_direction_accuracy(self, pred: torch.Tensor, actual: torch.Tensor) -> float:
        """Compute direction accuracy."""
        significant = actual.abs() > 0.1
        if not significant.any():
            return 0.5
            
        pred_sign = torch.sign(pred[significant])
        actual_sign = torch.sign(actual[significant])
        
        return (pred_sign == actual_sign).float().mean().item()


# =============================================================================
# DATA LOADING (from v5)
# =============================================================================

def load_singlecell_data(
    path: str,
    celltype_key: str = 'celltype'
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load single-cell data and compute pseudo-bulk."""
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy required")
    
    print(f"Loading {path}...")
    adata = sc.read_h5ad(path)
    
    # Check if pre-aggregated
    if adata.n_obs < 1000:
        print("  Detected pre-aggregated pseudo-bulk data")
        expression_means = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        celltype_names = list(adata.obs.index)
    else:
        print("  Computing pseudo-bulk from single cells...")
        celltype_names = list(adata.obs[celltype_key].unique())
        expression_means = np.zeros((len(celltype_names), adata.n_vars))
        
        for i, ct in enumerate(celltype_names):
            mask = adata.obs[celltype_key] == ct
            expression_means[i] = adata[mask].X.mean(axis=0).A1 if hasattr(adata[mask].X, 'A1') else adata[mask].X.mean(axis=0)
    
    print(f"  {len(celltype_names)} cell types")
    print(f"  Expression: {expression_means.shape}")
    
    gene_names = list(adata.var_names)
    
    return expression_means.astype(np.float32), gene_names, celltype_names


def load_tf_binding(
    path: str,
    gene_names: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load TF binding data."""
    binding_dir = Path(path)
    
    # Load from separate files (same structure as train_v5.py)
    binding_matrix_path = binding_dir / 'binding_matrix.npy'
    tf_tf_adj_path = binding_dir / 'tf_tf_adjacency.npy'
    mappings_path = binding_dir / 'mappings.pkl'
    
    if not binding_matrix_path.exists():
        raise FileNotFoundError(f"Binding matrix not found: {binding_matrix_path}")
    
    binding_matrix = np.load(binding_matrix_path)
    tf_tf_adj = np.load(tf_tf_adj_path) if tf_tf_adj_path.exists() else None
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    binding_genes = [str(g) for g in mappings['gene_names']]
    tf_names = mappings['tf_names']
    
    # Create gene name to index map
    binding_gene_map = {g.upper(): i for i, g in enumerate(binding_genes)}
    
    # Align binding matrix to our gene order
    n_genes = len(gene_names)
    n_tfs = len(tf_names)
    aligned = np.zeros((n_genes, n_tfs), dtype=np.float32)
    matched = 0
    
    for i, gene in enumerate(gene_names):
        if str(gene).upper() in binding_gene_map:
            aligned[i] = binding_matrix[binding_gene_map[str(gene).upper()]]
            matched += 1
    
    print(f"  Aligned {matched}/{n_genes} genes")
    print(f"  TFs: {n_tfs}")
    print(f"  Binding: {aligned.shape}")
    
    return aligned, tf_names, tf_tf_adj


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train GRN Model v6 (Integrated)')
    parser.add_argument('--singlecell', type=str, required=True)
    parser.add_argument('--tf-binding', type=str, required=True)
    parser.add_argument('--celltype-key', type=str, default='celltype')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--ppi', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--multi-tf', action='store_true', default=True)
    parser.add_argument('--no-jaspar', action='store_true')
    parser.add_argument('--no-chip-reg', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    expression_means, gene_names, celltype_names = load_singlecell_data(
        args.singlecell, args.celltype_key
    )
    
    # Compute baselines
    print("\nComputing baselines...")
    baselines = compute_baseline_metrics(expression_means)
    print(f"  Source-Target PCC: {baselines['source_target_pcc']:.4f} (range: {baselines['min_pair_pcc']:.4f}-{baselines['max_pair_pcc']:.4f})")
    print(f"  Mean-Pred PCC:     {baselines['mean_pred_pcc']:.4f}")
    print(f"  >>> Model Expr_PCC must significantly exceed {baselines['source_target_pcc']:.4f} to be learning!")
    
    # Load TF binding
    print(f"\nLoading TF binding from {args.tf_binding}...")
    binding_matrix, tf_names, tf_tf_adjacency = load_tf_binding(args.tf_binding, gene_names)
    
    n_genes = len(gene_names)
    n_tfs = len(tf_names)
    
    # Compute TF activity
    print("\nComputing TF activity...")
    tf_activity = compute_tf_activity_linger(expression_means, binding_matrix)
    tf_states = classify_tf_states(tf_activity)
    print(f"  TFs active per cell type: {int((tf_states != 0).sum(axis=1).mean())} avg")
    
    # Create cell-type pair splits
    n_celltypes = len(celltype_names)
    all_pairs = [(i, j) for i in range(n_celltypes) for j in range(n_celltypes) if i != j]
    random.shuffle(all_pairs)
    
    val_size = int(0.15 * len(all_pairs))
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]
    
    print(f"\nCell-type pair split (HOLDOUT validation):")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)} (NEVER seen during training!)")
    
    # Create datasets
    if args.multi_tf:
        print("\n>>> USING MULTI-TF TRAINING (all TFs together - CORRECT approach)")
        train_dataset = MultiTFDataset(expression_means, tf_activity, celltype_names, train_pairs)
        val_dataset = MultiTFDataset(expression_means, tf_activity, celltype_names, val_pairs, augment_factor=1, noise_std=0.0)
    else:
        train_dataset = VirtualPerturbationDatasetV5(expression_means, tf_activity, celltype_names, train_pairs)
        val_dataset = VirtualPerturbationDatasetV5(expression_means, tf_activity, celltype_names, val_pairs)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load optional priors
    tf_specificity_weights = None
    cofactor_tf_adjacency = None
    cofactor_gene_indices = None
    
    if not args.no_jaspar:
        jaspar_path = Path(args.tf_binding).parent / 'tf_specificity' / 'tf_specificity_weights_ppi.npy'
        if jaspar_path.exists():
            tf_specificity_weights = np.load(jaspar_path)
            print(f"  Loaded TF specificity weights: range=[{tf_specificity_weights.min():.3f}, {tf_specificity_weights.max():.3f}]")
    
    if args.ppi:
        ppi_path = Path(args.ppi)
        cof_path = ppi_path / 'cof_to_tf_adj.npy'
        cof_idx_path = ppi_path / 'cofactor_gene_indices.npy'
        
        if cof_path.exists() and cof_idx_path.exists():
            cofactor_tf_adjacency = np.load(cof_path)
            cofactor_gene_indices = np.load(cof_idx_path)
            print(f"  Loaded PPI cofactor gate: {len(cofactor_gene_indices)} cofactors -> {n_tfs} TFs")
            
        tf_tf_ppi_path = ppi_path / 'tf_tf_ppi.npy'
        if tf_tf_ppi_path.exists() and tf_tf_adjacency is not None:
            tf_tf_ppi = np.load(tf_tf_ppi_path)
            tf_tf_adjacency = 0.8 * tf_tf_adjacency + 0.2 * tf_tf_ppi
            print(f"  Blended TF-TF PPI into cascade (w_ppi=0.2)")
    
    # Create model - USING INTEGRATED MODEL!
    print("\n" + "=" * 80)
    print("  CREATING INTEGRATED GRN MODEL (v6)")
    print("  Using sophisticated components from models/")
    print("=" * 80)
    
    model = IntegratedGRNModel(
        n_genes=n_genes,
        n_tfs=n_tfs,
        hidden_dim=args.hidden_dim,
        num_gcn_layers=3,
        num_transformer_layers=2,
        num_heads=8,
        dropout=0.1,
        use_context_gating=True
    )
    
    # Set priors
    model.set_priors(
        chip_binding=torch.FloatTensor(binding_matrix),
        tf_tf_adjacency=torch.FloatTensor(tf_tf_adjacency) if tf_tf_adjacency is not None else None,
        tf_specificity_weights=torch.FloatTensor(tf_specificity_weights) if tf_specificity_weights is not None else None,
        cofactor_tf_adjacency=torch.FloatTensor(cofactor_tf_adjacency) if cofactor_tf_adjacency is not None else None,
        cofactor_gene_indices=torch.LongTensor(cofactor_gene_indices) if cofactor_gene_indices is not None else None
    )
    
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Save metadata
    print("\nSaving metadata files...")
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False)
    pd.DataFrame({'celltype': celltype_names}).to_csv(output_dir / 'celltype_names.csv', index=False)
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'tf_states.npy', tf_states)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model.config, f, indent=2)
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(baselines, f, indent=2)
    
    print("  Saved: gene_names.csv, tf_names.csv, celltype_names.csv")
    print("  Saved: tf_activity.npy, tf_states.npy, binding_matrix.npy")
    print("  Saved: model_config.json, baseline_metrics.json")
    
    # Create dataloaders
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create trainer
    trainer = IntegratedGRNTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_chip_reg=not args.no_chip_reg
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("  TRAINING INTEGRATED GRN MODEL v6")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Learning rate: {args.lr}")
    print(f"\n  BASELINES (model must beat these!):")
    print(f"    Source-Target PCC: {baselines['source_target_pcc']:.4f}")
    print(f"    Mean-Pred PCC:     {baselines['mean_pred_pcc']:.4f}")
    print("=" * 80)
    
    best_delta_pcc = -1
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.multi_tf:
            train_metrics = trainer.train_epoch_multi_tf(train_loader, epoch)
        else:
            raise NotImplementedError("Single-TF mode not implemented in v6 - use --multi-tf")
        
        print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} delta={train_metrics.get('delta_pcc', 0):.3f} dir_acc={train_metrics.get('dir_acc', 0):.3f}")
        
        # Validate
        if epoch % args.val_every == 0:
            if args.multi_tf:
                val_metrics = trainer.validate_multi_tf(val_loader)
            else:
                raise NotImplementedError("Single-TF mode not implemented in v6")
            
            delta_pcc = val_metrics['delta_pcc']
            is_learning = delta_pcc > 0.1
            
            print(f"  [VAL] loss={val_metrics['loss']:.4f} | Delta={delta_pcc:.3f} {'(LEARNING!)' if is_learning else ''} Dir={val_metrics['dir_acc']:.3f}")
            
            if delta_pcc > best_delta_pcc:
                best_delta_pcc = delta_pcc
                best_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'delta_pcc': delta_pcc,
                    'config': model.config
                }, output_dir / 'best_model.pt')
                
                # Save GRN matrix (both regular and CORRECT versions for fine-tuning)
                grn_matrix = model.get_grn_matrix(epoch).detach().cpu().numpy()
                np.save(output_dir / 'grn_matrix.npy', grn_matrix)
                np.save(output_dir / 'grn_matrix_CORRECT.npy', grn_matrix)  # For fine-tuning (no warning)
                
                print(f"  >> New best! Delta_PCC={delta_pcc:.4f}")
    
    print("\n" + "=" * 80)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best Delta_PCC: {best_delta_pcc:.4f} at epoch {best_epoch}")
    print(f"  Model saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: COMPLETE MOLECULAR SIGNALING NETWORK TRAINING
================================================================================

Extension of train_v6 to incorporate:
- Gene→Gene signaling (OmniPath)
- Gene→TF regulation (OmniPath) 
- TF→Gene expression (ChIP-Atlas)
- TF→TF cascade (ChIP-Atlas + OmniPath)
- Cofactor modulation (Rosetta PPI)

Key Innovation:
- Phase 1: Learn known network from OmniPath + ChIP-Atlas
- Phase 2: Discovery mode - explore unknown connections

Architecture:
```
                    OmniPath                     ChIP-Atlas
                   Gene→Gene                      TF→Gene
                   Gene→TF                        TF→TF
                      ↓                              ↓
              ┌──────────────────────────────────────────────┐
              │         UNIFIED SIGNALING GRAPH GNN          │
              │                                              │
              │   Gene Nodes ←→ TF Nodes (message passing)   │
              │                                              │
              └──────────────────────────────────────────────┘
                                    ↓
              ┌──────────────────────────────────────────────┐
              │         PERTURBATION PREDICTION              │
              │                                              │
              │   TF_activity_change → Gene_expression_Δ     │
              │   + Gene_state_change → TF_activity_Δ        │
              │                                              │
              └──────────────────────────────────────────────┘
```

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
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from scipy import stats as scipy_stats
import warnings

# DGL for graph neural networks
try:
    import dgl
    from dgl.nn import GraphConv, GATConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("Warning: DGL not installed. Using simplified cascade.")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_singlecell_data(h5ad_path: str, celltype_key: str = 'celltype'):
    """Load and preprocess single-cell data."""
    import scanpy as sc
    
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Check if already aggregated
    if len(adata) < 500:
        print(f"  Detected pre-aggregated pseudo-bulk data")
        expression = adata.X
        if hasattr(expression, 'toarray'):
            expression = expression.toarray()
        celltypes = adata.obs[celltype_key].values if celltype_key in adata.obs else adata.obs.index.values
        gene_names = list(adata.var_names)
    else:
        print(f"  Aggregating to pseudo-bulk by {celltype_key}...")
        celltypes = adata.obs[celltype_key].unique()
        expression = np.zeros((len(celltypes), adata.n_vars))
        for i, ct in enumerate(celltypes):
            mask = adata.obs[celltype_key] == ct
            ct_expr = adata.X[mask]
            if hasattr(ct_expr, 'toarray'):
                ct_expr = ct_expr.toarray()
            expression[i] = ct_expr.mean(axis=0)
        gene_names = list(adata.var_names)
    
    print(f"  {len(celltypes)} cell types")
    print(f"  Expression: {expression.shape}")
    
    return expression.astype(np.float32), gene_names, list(celltypes)


def load_tf_binding(tf_binding_dir: str, gene_names: List[str]):
    """Load TF binding matrices from ChIP-Atlas data."""
    tf_dir = Path(tf_binding_dir)
    
    # Load mappings
    mappings_path = tf_dir / 'mappings.pkl'
    if mappings_path.exists():
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        tf_names = mappings.get('tf_names', [])
        binding_gene_names = mappings.get('gene_names', [])
    else:
        tf_names = pd.read_csv(tf_dir.parent / 'tf_names.csv', header=None)[0].tolist()
        binding_gene_names = gene_names
    
    # Load binding matrix
    binding_path = tf_dir / 'binding_matrix.npy'
    if binding_path.exists():
        binding_matrix = np.load(binding_path)
    else:
        binding_matrix = np.zeros((len(gene_names), len(tf_names)))
    
    # Load TF-TF adjacency
    tf_tf_path = tf_dir / 'tf_tf_adjacency.npy'
    if tf_tf_path.exists():
        tf_tf_adj = np.load(tf_tf_path)
    else:
        tf_tf_adj = np.zeros((len(tf_names), len(tf_names)))
    
    # Align genes
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    binding_gene_to_idx = {g.upper(): i for i, g in enumerate(binding_gene_names)}
    
    aligned_binding = np.zeros((len(gene_names), len(tf_names)), dtype=np.float32)
    for gene in gene_names:
        gene_upper = gene.upper()
        if gene_upper in gene_to_idx and gene_upper in binding_gene_to_idx:
            aligned_binding[gene_to_idx[gene_upper]] = binding_matrix[binding_gene_to_idx[gene_upper]]
    
    print(f"  Aligned {np.sum(aligned_binding.any(axis=1))}/{len(gene_names)} genes")
    print(f"  TFs: {len(tf_names)}")
    print(f"  Binding: {aligned_binding.shape}")
    
    return aligned_binding, tf_names, tf_tf_adj


def load_omnipath(omnipath_dir: str, n_genes: int, n_tfs: int):
    """Load OmniPath signaling matrices."""
    opath = Path(omnipath_dir)
    
    # Gene→Gene signaling
    gene_gene_path = opath / 'gene_gene_signaling.npy'
    if gene_gene_path.exists():
        gene_gene = np.load(gene_gene_path)
        print(f"  Gene→Gene: {gene_gene.shape}, {np.count_nonzero(gene_gene):,} edges")
    else:
        gene_gene = np.zeros((n_genes, n_genes), dtype=np.float32)
        print(f"  Gene→Gene: not found, using zeros")
    
    # Gene→TF signaling
    gene_tf_path = opath / 'gene_tf_signaling.npy'
    if gene_tf_path.exists():
        gene_tf = np.load(gene_tf_path)
        print(f"  Gene→TF: {gene_tf.shape}, {np.count_nonzero(gene_tf):,} edges")
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


def compute_tf_activity_linger(expression: np.ndarray, binding_matrix: np.ndarray) -> np.ndarray:
    """Compute TF activity using LINGER-style method."""
    binding_norm = binding_matrix / (binding_matrix.sum(axis=0, keepdims=True) + 1e-8)
    tf_activity = expression @ binding_norm
    return tf_activity.astype(np.float32)


# =============================================================================
# UNIFIED SIGNALING GRAPH MODEL
# =============================================================================

class UnifiedSignalingGNN(nn.Module):
    """
    Graph Neural Network that operates on the unified signaling graph.
    
    Nodes: All genes (including TFs)
    Edges: 
        - Gene→Gene (OmniPath activation/inhibition)
        - Gene→TF (OmniPath TF regulation)
        - TF→Gene (ChIP-Atlas binding)
        - TF→TF (ChIP-Atlas + OmniPath cascade)
    """
    
    def __init__(self, n_genes: int, n_tfs: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.gene_embed = nn.Embedding(n_genes, hidden_dim)
        
        # Edge type embeddings (for heterogeneous edges)
        self.edge_type_embed = nn.Embedding(4, hidden_dim)  # 4 edge types
        
        # GNN layers
        if HAS_DGL:
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // 4, num_heads=4, 
                       feat_drop=0.1, attn_drop=0.1, residual=True)
                for _ in range(n_layers)
            ])
        else:
            # Fallback: simple message passing
            self.gnn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(n_layers)
            ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Graph storage
        self.graph = None
        self.edge_weights = None
        
    def build_graph(self, gene_gene: np.ndarray, gene_tf: np.ndarray,
                    tf_gene: np.ndarray, tf_tf: np.ndarray, tf_indices: np.ndarray):
        """
        Build unified graph from adjacency matrices.
        
        All matrices are in gene-space:
        - gene_gene: (n_genes, n_genes)
        - gene_tf: (n_genes, n_tfs) → map TF indices to gene indices
        - tf_gene: (n_tfs, n_genes) → map TF indices to gene indices
        - tf_tf: (n_tfs, n_tfs) → map TF indices to gene indices
        
        Args:
            tf_indices: Indices of TFs in the gene list
        """
        if not HAS_DGL:
            return
        
        edges_src = []
        edges_dst = []
        edge_weights = []
        edge_types = []  # 0: gene→gene, 1: gene→tf, 2: tf→gene, 3: tf→tf
        
        # Gene→Gene edges
        src, dst = np.nonzero(gene_gene)
        for i in range(len(src)):
            edges_src.append(src[i])
            edges_dst.append(dst[i])
            edge_weights.append(gene_gene[src[i], dst[i]])
            edge_types.append(0)
        
        # Gene→TF edges (map TF to gene index)
        src, dst = np.nonzero(gene_tf)
        for i in range(len(src)):
            gene_idx = src[i]
            tf_idx = dst[i]
            if tf_idx < len(tf_indices):
                tf_gene_idx = tf_indices[tf_idx]
                edges_src.append(gene_idx)
                edges_dst.append(tf_gene_idx)
                edge_weights.append(gene_tf[src[i], dst[i]])
                edge_types.append(1)
        
        # TF→Gene edges
        src, dst = np.nonzero(tf_gene)
        for i in range(len(src)):
            tf_idx = src[i]
            gene_idx = dst[i]
            if tf_idx < len(tf_indices):
                tf_gene_idx = tf_indices[tf_idx]
                edges_src.append(tf_gene_idx)
                edges_dst.append(gene_idx)
                edge_weights.append(tf_gene[src[i], dst[i]])
                edge_types.append(2)
        
        # TF→TF edges
        src, dst = np.nonzero(tf_tf)
        for i in range(len(src)):
            if src[i] < len(tf_indices) and dst[i] < len(tf_indices):
                edges_src.append(tf_indices[src[i]])
                edges_dst.append(tf_indices[dst[i]])
                edge_weights.append(tf_tf[src[i], dst[i]])
                edge_types.append(3)
        
        # Create DGL graph
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.n_genes)
        self.edge_weights = torch.FloatTensor(edge_weights)
        self.edge_types = torch.LongTensor(edge_types)
        
        # Add self-loops
        self.graph = dgl.add_self_loop(self.graph)
        self.edge_weights = torch.cat([self.edge_weights, torch.ones(self.n_genes)])
        self.edge_types = torch.cat([self.edge_types, torch.zeros(self.n_genes, dtype=torch.long)])
        
        print(f"  [UnifiedGNN] Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def forward(self, gene_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the signaling graph.
        
        Args:
            gene_states: Optional (batch, n_genes) expression states
            
        Returns:
            node_embeddings: (n_genes, hidden_dim) updated node representations
        """
        # Initial embeddings
        x = self.gene_embed.weight  # (n_genes, hidden_dim)
        
        if self.graph is None or not HAS_DGL:
            # Fallback: simple transformation
            for layer, norm in zip(self.gnn_layers, self.layer_norms):
                x = x + F.gelu(layer(x))
                x = norm(x)
            return x
        
        # Move graph to device
        device = x.device
        graph = self.graph.to(device)
        edge_weights = self.edge_weights.to(device)
        
        # GNN message passing
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            # Apply GAT
            h = layer(graph, x).flatten(1)  # GAT output
            x = x + h
            x = norm(x)
        
        return x


class FullSignalingGRNModel(nn.Module):
    """
    Complete GRN model with full signaling network.
    
    Combines:
    - TF→Gene effects (learned from ChIP-Atlas prior)
    - Gene→Gene propagation (OmniPath signaling)
    - Gene→TF regulation (OmniPath upstream)
    - TF cascade (TF→TF)
    """
    
    def __init__(self, n_genes: int, n_tfs: int, hidden_dim: int = 256,
                 use_gnn: bool = True, n_gnn_layers: int = 3):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        self.use_gnn = use_gnn and HAS_DGL
        
        # Learnable TF→Gene effects (main regulatory effects)
        self.tf_to_gene = nn.Parameter(torch.zeros(n_genes, n_tfs))
        nn.init.xavier_normal_(self.tf_to_gene)
        
        # Learnable Gene→TF effects (upstream regulation)
        self.gene_to_tf = nn.Parameter(torch.zeros(n_tfs, n_genes))
        nn.init.xavier_normal_(self.gene_to_tf)
        
        # Gene→Gene propagation weights
        self.gene_propagation = nn.Parameter(torch.zeros(n_genes, n_genes))
        
        # TF importance (attention-like weighting)
        self.tf_importance = nn.Parameter(torch.zeros(n_tfs))
        
        # ChIP mask and leakage
        self.register_buffer('chip_mask', torch.ones(n_genes, n_tfs))
        self._leakage_raw = nn.Parameter(torch.tensor(0.0))
        
        # Signaling network priors
        self.register_buffer('gene_gene_prior', torch.zeros(n_genes, n_genes))
        self.register_buffer('gene_tf_prior', torch.zeros(n_genes, n_tfs))
        self.register_buffer('tf_tf_prior', torch.zeros(n_tfs, n_tfs))
        
        # Optional GNN for message passing
        if self.use_gnn:
            self.signaling_gnn = UnifiedSignalingGNN(n_genes, n_tfs, hidden_dim, n_gnn_layers)
        
        # Context encoder (cell state)
        self.context_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_tfs)
        )
        
        # Discovery mode parameters
        self.discovery_threshold = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('discovered_edges', torch.zeros(n_genes, n_genes))
        
    @property
    def leakage(self):
        return 0.005 + 0.195 * torch.sigmoid(self._leakage_raw)
    
    def set_priors(self, chip_mask: np.ndarray, gene_gene: np.ndarray,
                   gene_tf: np.ndarray, tf_tf: np.ndarray,
                   tf_specificity: Optional[np.ndarray] = None,
                   tf_indices: Optional[np.ndarray] = None):
        """Set network priors from preprocessed data."""
        
        self.chip_mask.copy_(torch.from_numpy(chip_mask).float())
        self.gene_gene_prior.copy_(torch.from_numpy(gene_gene).float())
        self.gene_tf_prior.copy_(torch.from_numpy(gene_tf).float())
        self.tf_tf_prior.copy_(torch.from_numpy(tf_tf).float())
        
        # Initialize learnable parameters from priors
        with torch.no_grad():
            # Gene propagation starts from OmniPath prior
            self.gene_propagation.copy_(torch.from_numpy(gene_gene).float() * 0.1)
            
            # Gene→TF starts from OmniPath prior
            self.gene_to_tf.copy_(torch.from_numpy(gene_tf.T).float() * 0.1)
        
        # Build GNN graph
        if self.use_gnn and tf_indices is not None:
            tf_gene = self.chip_mask.cpu().numpy().T  # (n_tfs, n_genes)
            self.signaling_gnn.build_graph(
                gene_gene, gene_tf, tf_gene, tf_tf, tf_indices
            )
    
    def get_effective_tf_effects(self) -> torch.Tensor:
        """Get TF→Gene effects with ChIP mask and leakage."""
        leakage = self.leakage
        masked_effects = self.tf_to_gene * (self.chip_mask + leakage)
        return masked_effects
    
    def compute_gene_propagation(self, gene_deltas: torch.Tensor, n_steps: int = 2) -> torch.Tensor:
        """
        Propagate gene expression changes through the gene network.
        
        This models signaling cascades: Gene A → Gene B → Gene C
        
        Args:
            gene_deltas: (batch, n_genes) initial gene changes (from TF effects)
            n_steps: Number of propagation steps
            
        Returns:
            propagated_deltas: (batch, n_genes) after propagation
        """
        # Combine learned and prior gene-gene effects
        gene_network = self.gene_propagation + 0.5 * self.gene_gene_prior
        
        # Normalize rows (each gene's outgoing effects sum to reasonable scale)
        row_norm = gene_network.abs().sum(dim=1, keepdim=True).clamp(min=1.0)
        gene_network_norm = gene_network / row_norm
        
        # Iterative propagation
        current = gene_deltas
        for _ in range(n_steps):
            # Message passing: each gene receives signals from connected genes
            propagated = current @ gene_network_norm.T
            current = current + 0.3 * propagated  # Residual connection
        
        return current
    
    def compute_tf_modulation(self, gene_states: torch.Tensor) -> torch.Tensor:
        """
        Compute how gene states modulate TF activity.
        
        This models upstream regulation: Gene → TF
        
        Args:
            gene_states: (batch, n_genes) current gene expression
            
        Returns:
            tf_modulation: (batch, n_tfs) multiplicative modulation of TF effects
        """
        # Combine learned and prior gene→TF effects
        gene_tf = self.gene_to_tf + 0.5 * self.gene_tf_prior.T
        
        # Compute modulation
        raw_modulation = gene_states @ gene_tf.T  # (batch, n_tfs)
        
        # Convert to multiplicative gate (centered at 1.0)
        tf_modulation = 1.0 + 0.2 * torch.tanh(raw_modulation / 10.0)
        
        return tf_modulation
    
    def forward(self, tf_activity_change: torch.Tensor, 
                source_expression: torch.Tensor) -> torch.Tensor:
        """
        Predict gene expression changes from TF activity changes.
        
        Full pipeline:
        1. TF cascade effects (TF→TF)
        2. Context modulation (cell state)
        3. Gene→TF modulation (upstream regulation)
        4. TF→Gene effects
        5. Gene→Gene propagation
        
        Args:
            tf_activity_change: (batch, n_tfs) change in TF activity
            source_expression: (batch, n_genes) source cell expression
            
        Returns:
            predicted_delta: (batch, n_genes) predicted expression changes
        """
        batch_size = tf_activity_change.shape[0]
        device = tf_activity_change.device
        
        # 1. TF cascade (TF→TF propagation)
        tf_cascade = tf_activity_change @ (self.tf_tf_prior + 0.1 * torch.eye(self.n_tfs, device=device))
        tf_effective = tf_activity_change + 0.3 * tf_cascade
        
        # 2. Context modulation (cell-type specific)
        context = self.context_encoder(source_expression)  # (batch, n_tfs)
        context_gate = torch.sigmoid(context)
        tf_effective = tf_effective * context_gate
        
        # 3. Gene→TF modulation (upstream regulation)
        tf_modulation = self.compute_tf_modulation(source_expression)
        tf_effective = tf_effective * tf_modulation
        
        # 4. TF→Gene effects
        tf_effects = self.get_effective_tf_effects()  # (n_genes, n_tfs)
        
        # Apply TF importance weighting
        tf_importance = F.softmax(self.tf_importance, dim=0)
        weighted_tf = tf_effective * tf_importance
        
        # Compute initial gene deltas
        gene_deltas = weighted_tf @ tf_effects.T  # (batch, n_genes)
        
        # 5. Gene→Gene propagation
        propagated_deltas = self.compute_gene_propagation(gene_deltas)
        
        return propagated_deltas
    
    def get_grn_matrix(self) -> torch.Tensor:
        """Get the learned GRN matrix (TF→Gene effects)."""
        return self.get_effective_tf_effects()
    
    def get_signaling_network(self) -> Dict[str, torch.Tensor]:
        """Get all learned signaling networks."""
        return {
            'tf_to_gene': self.get_effective_tf_effects().detach(),
            'gene_to_gene': (self.gene_propagation + 0.5 * self.gene_gene_prior).detach(),
            'gene_to_tf': (self.gene_to_tf + 0.5 * self.gene_tf_prior.T).detach(),
            'tf_to_tf': self.tf_tf_prior.detach(),
        }


# =============================================================================
# DATASET
# =============================================================================

class SignalingPerturbationDataset(Dataset):
    """
    Dataset for full signaling network training.
    
    Each sample: (source_celltype, target_celltype) pair
    - TF activity change between cell types
    - Gene expression change (ground truth)
    - Gene states for upstream modulation
    """
    
    def __init__(self, expression: np.ndarray, tf_activity: np.ndarray,
                 celltype_pairs: List[Tuple[int, int]]):
        self.expression = torch.FloatTensor(expression)
        self.tf_activity = torch.FloatTensor(tf_activity)
        self.pairs = celltype_pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]
        
        return {
            'tf_activity_change': self.tf_activity[tgt_idx] - self.tf_activity[src_idx],
            'source_expression': self.expression[src_idx],
            'target_expression': self.expression[tgt_idx],
            'gene_delta': self.expression[tgt_idx] - self.expression[src_idx],
        }


# =============================================================================
# TRAINING
# =============================================================================

class SignalingGRNTrainer:
    """Trainer for full signaling GRN model."""
    
    def __init__(self, model: FullSignalingGRNModel, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 0.01):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        epoch_loss = 0
        epoch_delta_corr = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            tf_change = batch['tf_activity_change'].to(self.device)
            source_expr = batch['source_expression'].to(self.device)
            gene_delta = batch['gene_delta'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                pred_delta = self.model(tf_change, source_expr)
                
                # MSE loss on gene deltas
                mse_loss = F.mse_loss(pred_delta, gene_delta)
                
                # Correlation loss (maximize correlation)
                pred_flat = pred_delta.flatten()
                actual_flat = gene_delta.flatten()
                if pred_flat.std() > 1e-6 and actual_flat.std() > 1e-6:
                    corr = torch.corrcoef(torch.stack([pred_flat, actual_flat]))[0, 1]
                    corr_loss = 1 - corr
                else:
                    corr_loss = torch.tensor(0.0, device=self.device)
                
                # L1 regularization on network sparsity
                l1_reg = 0.001 * (
                    self.model.gene_propagation.abs().mean() +
                    self.model.gene_to_tf.abs().mean()
                )
                
                loss = mse_loss + 0.5 * corr_loss + l1_reg
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track delta correlation
            with torch.no_grad():
                if pred_flat.std() > 1e-6 and actual_flat.std() > 1e-6:
                    epoch_delta_corr += torch.corrcoef(
                        torch.stack([pred_flat, actual_flat])
                    )[0, 1].item()
            
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        self.scheduler.step()
        
        return {
            'loss': epoch_loss / n_batches,
            'delta_corr': epoch_delta_corr / n_batches,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_pred = []
        all_actual = []
        
        for batch in val_loader:
            tf_change = batch['tf_activity_change'].to(self.device)
            source_expr = batch['source_expression'].to(self.device)
            gene_delta = batch['gene_delta'].to(self.device)
            
            pred_delta = self.model(tf_change, source_expr)
            
            all_pred.append(pred_delta.cpu())
            all_actual.append(gene_delta.cpu())
        
        all_pred = torch.cat(all_pred).numpy()
        all_actual = torch.cat(all_actual).numpy()
        
        # Overall correlation
        pred_flat = all_pred.flatten()
        actual_flat = all_actual.flatten()
        
        delta_pcc, _ = scipy_stats.pearsonr(pred_flat, actual_flat)
        
        # Direction accuracy
        pred_sign = np.sign(pred_flat)
        actual_sign = np.sign(actual_flat)
        significant = np.abs(actual_flat) > np.std(actual_flat) * 0.1
        dir_acc = np.mean(pred_sign[significant] == actual_sign[significant])
        
        return {
            'delta_pcc': delta_pcc,
            'dir_acc': dir_acc,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BRIDGE 2.0: Full Signaling Network Training')
    parser.add_argument('--singlecell', type=str, required=True, help='Path to h5ad file')
    parser.add_argument('--tf-binding', type=str, required=True, help='TF binding directory')
    parser.add_argument('--omnipath', type=str, default=None, help='OmniPath directory')
    parser.add_argument('--ppi', type=str, default=None, help='PPI directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--use-gnn', action='store_true', help='Use GNN for message passing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BRIDGE 2.0: FULL MOLECULAR SIGNALING NETWORK")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(args.singlecell, args.celltype_key)
    
    print("\n[2/5] Loading TF binding data...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(args.tf_binding, gene_names)
    
    print("\n[3/5] Loading OmniPath signaling network...")
    if args.omnipath:
        gene_gene, gene_tf, tf_tf_omnipath = load_omnipath(
            args.omnipath, len(gene_names), len(tf_names)
        )
        # Blend TF-TF from ChIP and OmniPath
        tf_tf_combined = tf_tf_adj + 0.5 * (tf_tf_omnipath if tf_tf_omnipath is not None else 0)
    else:
        gene_gene = np.zeros((len(gene_names), len(gene_names)), dtype=np.float32)
        gene_tf = np.zeros((len(gene_names), len(tf_names)), dtype=np.float32)
        tf_tf_combined = tf_tf_adj
        print("  No OmniPath data, using zeros")
    
    print("\n[4/5] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    print(f"  TF activity: {tf_activity.shape}")
    
    # Create cell-type pairs
    n_celltypes = len(celltypes)
    n_holdout = max(2, n_celltypes // 6)
    np.random.seed(42)
    holdout = set(np.random.choice(n_celltypes, n_holdout, replace=False))
    train_idx = [i for i in range(n_celltypes) if i not in holdout]
    
    train_pairs = [(i, j) for i in train_idx for j in train_idx if i != j]
    val_pairs = [(i, j) for i in holdout for j in holdout if i != j]
    
    print(f"\n  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = SignalingPerturbationDataset(expression, tf_activity, train_pairs)
    val_dataset = SignalingPerturbationDataset(expression, tf_activity, val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Find TF indices in gene list
    gene_upper = {g.upper(): i for i, g in enumerate(gene_names)}
    tf_indices = np.array([gene_upper.get(t.upper(), -1) for t in tf_names])
    valid_tf_mask = tf_indices >= 0
    tf_indices = tf_indices[valid_tf_mask]
    
    print("\n[5/5] Building model...")
    model = FullSignalingGRNModel(
        n_genes=len(gene_names),
        n_tfs=len(tf_names),
        hidden_dim=args.hidden_dim,
        use_gnn=args.use_gnn
    )
    
    # Set network priors
    model.set_priors(
        chip_mask=binding_matrix,
        gene_gene=gene_gene,
        gene_tf=gene_tf,
        tf_tf=tf_tf_combined,
        tf_indices=tf_indices
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save metadata
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False, header=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False, header=False)
    pd.DataFrame({'celltype': celltypes}).to_csv(output_dir / 'celltype_names.csv', index=False, header=False)
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    trainer = SignalingGRNTrainer(model, device, lr=args.lr)
    
    best_delta_pcc = -1
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        
        print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} delta_corr={train_metrics['delta_corr']:.4f}")
        
        if epoch % args.val_every == 0:
            val_metrics = trainer.validate(val_loader)
            print(f"  [VAL] Delta_PCC={val_metrics['delta_pcc']:.4f} Dir_Acc={val_metrics['dir_acc']:.4f}")
            
            if val_metrics['delta_pcc'] > best_delta_pcc:
                best_delta_pcc = val_metrics['delta_pcc']
                best_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'delta_pcc': best_delta_pcc,
                }, output_dir / 'best_model.pt')
                
                # Save signaling networks
                networks = model.get_signaling_network()
                for name, tensor in networks.items():
                    np.save(output_dir / f'{name}.npy', tensor.cpu().numpy())
                
                print(f"  >> New best! Delta_PCC={best_delta_pcc:.4f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best model: Epoch {best_epoch}, Delta_PCC={best_delta_pcc:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()


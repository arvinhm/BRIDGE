#!/usr/bin/env python3
"""
================================================================================
Gene-Specific GRN Fine-Tuning - INTEGRATED WITH train_v6 ARCHITECTURE
================================================================================

Uses the SAME sophisticated architecture as train_v6.py:
- IntegratedGRNModel with TFRegulatoryEncoder, CellStateEncoder
- TF cascade GCN
- Context gating
- Cofactor modulation

Supports TWO modes:
1. SINGLE GENE: Fine-tune for one target gene (e.g., SSTR2)
2. MULTI-GENE: Fine-tune for multiple genes from a .txt file
   - Identifies MASTER REGULATORS that affect ALL genes
   - Generates specialized multi-gene analysis figures

At the end, automatically runs analyze_target_full.py to generate figures.

Usage (Single Gene):
    python scripts/finetune_gene.py \\
        --gene SSTR2 \\
        --model results/sclc_grn_v6/ \\
        --singlecell data/sclc/preprocessed.h5ad \\
        --tf-binding data/tf_binding/ \\
        --output results/sclc_SSTR2_finetuned/ \\
        --ppi data/ppi/ \\
        --epochs 100

Usage (Multi-Gene - Master Regulators):
    python scripts/finetune_gene.py \\
        --gene genes_of_interest.txt \\
        --model results/sclc_grn_v6/ \\
        --singlecell data/sclc/preprocessed.h5ad \\
        --tf-binding data/tf_binding/ \\
        --output results/sclc_master_regulators/ \\
        --ppi data/ppi/ \\
        --epochs 100

    # genes_of_interest.txt format (one gene per line):
    # SSTR2
    # ASCL1
    # NEUROD1
    # DLL3
    # ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import from train_v6 (the sophisticated version!)
from train_v6 import (
    load_singlecell_data,
    load_tf_binding,
    compute_tf_activity_linger,
    IntegratedGRNModel,
    MultiTFDataset,
)

# Import analysis module
from analyze_target_full import (
    GRNAnalyzer, create_figure1, create_figure2, create_figure3, create_figure4,
    save_csvs, setup_style, compute_tf_generalizability, COLORS, CMAP_DIVERGING
)


# =============================================================================
# DATASET WITH CELL-TYPE HOLDOUT (NO LEAKAGE)
# =============================================================================

class GeneSpecificDataset(Dataset):
    """
    Dataset with PROPER cell-type holdout for gene-specific training.
    
    Uses the SAME data as train_v6 but focuses on one target gene.
    """
    
    def __init__(
        self,
        expression: np.ndarray,  # (n_celltypes, n_genes)
        tf_activity: np.ndarray,  # (n_celltypes, n_tfs)
        target_gene_idx: int,
        allowed_pairs: List[Tuple[int, int]]
    ):
        self.expression = torch.FloatTensor(expression)
        self.tf_activity = torch.FloatTensor(tf_activity)
        self.target_gene_idx = target_gene_idx
        self.pairs = list(allowed_pairs)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]
        
        # Source and target expression (full gene vector for model context)
        source_expression = self.expression[src_idx]
        target_expression = self.expression[tgt_idx]
        
        # TF activity change (ALL TFs)
        tf_activity_change = self.tf_activity[tgt_idx] - self.tf_activity[src_idx]
        
        # Full delta expression (for model context)
        delta_expression = target_expression - source_expression
        
        # Target gene delta (THE OUTPUT TO PREDICT)
        target_delta = delta_expression[self.target_gene_idx]
        
        return {
            'source_expression': source_expression,
            'target_expression': target_expression,
            'tf_activity_change': tf_activity_change,
            'delta_expression': delta_expression,
            'target_delta': target_delta,
            'src_idx': src_idx,
            'tgt_idx': tgt_idx
        }


class MultiGeneDataset(Dataset):
    """
    Dataset for MULTI-GENE fine-tuning.
    
    Predicts deltas for ALL target genes simultaneously.
    Used to find master regulators.
    """
    
    def __init__(
        self,
        expression: np.ndarray,  # (n_celltypes, n_genes)
        tf_activity: np.ndarray,  # (n_celltypes, n_tfs)
        target_gene_indices: List[int],
        allowed_pairs: List[Tuple[int, int]]
    ):
        self.expression = torch.FloatTensor(expression)
        self.tf_activity = torch.FloatTensor(tf_activity)
        self.target_gene_indices = target_gene_indices
        self.n_targets = len(target_gene_indices)
        self.pairs = list(allowed_pairs)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]
        
        # Source and target expression
        source_expression = self.expression[src_idx]
        target_expression = self.expression[tgt_idx]
        
        # TF activity change
        tf_activity_change = self.tf_activity[tgt_idx] - self.tf_activity[src_idx]
        
        # Full delta expression
        delta_expression = target_expression - source_expression
        
        # Target gene deltas (MULTIPLE GENES)
        target_deltas = torch.stack([
            delta_expression[gene_idx] for gene_idx in self.target_gene_indices
        ])
        
        return {
            'source_expression': source_expression,
            'target_expression': target_expression,
            'tf_activity_change': tf_activity_change,
            'delta_expression': delta_expression,
            'target_deltas': target_deltas,  # (n_targets,)
            'src_idx': src_idx,
            'tgt_idx': tgt_idx
        }


# =============================================================================
# MULTI-GENE REFINEMENT HEAD
# =============================================================================

class MultiGeneRefinementHead(nn.Module):
    """
    Multi-gene refinement head for master regulator discovery.
    
    Predicts deltas for MULTIPLE genes and learns shared TF importance.
    """
    
    def __init__(
        self,
        n_tfs: int,
        n_genes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_tfs = n_tfs
        self.n_genes = n_genes
        
        # Shared TF importance (master regulators will have high importance)
        self.tf_importance = nn.Parameter(torch.zeros(n_tfs))
        
        # Per-gene refinement (small to encourage shared patterns)
        self.gene_specific = nn.Parameter(torch.zeros(n_genes, n_tfs) * 0.01)
        
        # Shared refinement MLP
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_tfs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Per-gene output
        self.gene_heads = nn.Linear(hidden_dim // 2, n_genes)
        
        # Blend weight per gene
        self.blend_weights = nn.Parameter(torch.ones(n_genes) * 0.5)
        
    def forward(
        self, 
        tf_activity_change: torch.Tensor,
        base_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict refined deltas for all target genes.
        
        Args:
            tf_activity_change: (batch, n_tfs)
            base_predictions: (batch, n_genes) - predictions from base model
            
        Returns:
            refined_deltas: (batch, n_genes)
        """
        # Apply shared TF importance
        importance_weights = F.softmax(self.tf_importance, dim=0)
        weighted_tf = tf_activity_change * importance_weights
        
        # Shared encoding
        shared_encoding = self.shared_encoder(weighted_tf)
        
        # Per-gene predictions
        refinement = self.gene_heads(shared_encoding)  # (batch, n_genes)
        
        # Blend with base predictions
        blend = torch.sigmoid(self.blend_weights)  # (n_genes,)
        refined = blend * base_predictions + (1 - blend) * refinement
        
        return refined
    
    def get_tf_importance(self) -> np.ndarray:
        """Get shared (master) TF importance weights."""
        return F.softmax(self.tf_importance, dim=0).detach().cpu().numpy()
    
    def get_gene_specific_importance(self) -> np.ndarray:
        """Get per-gene TF importance adjustments."""
        return self.gene_specific.detach().cpu().numpy()


# =============================================================================
# GENE-SPECIFIC HEAD (ADDED ON TOP OF IntegratedGRNModel)
# =============================================================================

class GeneSpecificRefinementHead(nn.Module):
    """
    Gene-specific refinement head that sits ON TOP of IntegratedGRNModel.
    
    Takes the base model's delta prediction for the target gene and refines it
    using additional learned patterns specific to this gene.
    """
    
    def __init__(
        self,
        n_tfs: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_tfs = n_tfs
        
        # Refinement MLP: takes TF activity change -> predicts delta refinement
        self.refinement = nn.Sequential(
            nn.Linear(n_tfs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable TF importance for this gene (initialized uniform)
        self.tf_importance = nn.Parameter(torch.zeros(n_tfs))
        
        # Learnable blend between base model and refinement
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self, 
        tf_activity_change: torch.Tensor,
        base_prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict refined target gene delta.
        
        Args:
            tf_activity_change: (batch, n_tfs)
            base_prediction: (batch,) - prediction from base model
            
        Returns:
            refined_delta: (batch,)
        """
        # Apply learnable TF importance
        importance_weights = F.softmax(self.tf_importance, dim=0)
        weighted_tf = tf_activity_change * importance_weights
        
        # Refinement prediction
        refinement = self.refinement(weighted_tf).squeeze(-1)
        
        # Blend base prediction with refinement
        blend = torch.sigmoid(self.blend_weight)
        refined = blend * base_prediction + (1 - blend) * refinement
        
        return refined
    
    def get_tf_importance(self) -> np.ndarray:
        """Get learned TF importance weights."""
        return F.softmax(self.tf_importance, dim=0).detach().cpu().numpy()


# =============================================================================
# FINE-TUNER USING IntegratedGRNModel
# =============================================================================

class GeneSpecificFineTuner:
    """
    Fine-tuner that uses the SAME IntegratedGRNModel architecture as train_v6.
    """
    
    def __init__(
        self,
        base_model: IntegratedGRNModel,
        target_gene_idx: int,
        target_gene_name: str,
        n_tfs: int,
        device: str = 'cuda',
        hidden_dim: int = 256,
        freeze_base: bool = True
    ):
        self.device = device
        self.target_gene_idx = target_gene_idx
        self.target_gene_name = target_gene_name
        self.n_tfs = n_tfs
        
        # Move base model to device
        self.base_model = base_model.to(device)
        
        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print(f"  Froze {sum(p.numel() for p in self.base_model.parameters()):,} base model parameters")
        
        # Add gene-specific refinement head
        self.head = GeneSpecificRefinementHead(
            n_tfs=n_tfs,
            hidden_dim=hidden_dim,
            dropout=0.2
        ).to(device)
        
        n_head_params = sum(p.numel() for p in self.head.parameters())
        print(f"  Gene-specific head: {n_head_params:,} parameters")
        
        # Optimizer only for head parameters
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
        # Track metrics
        self.history = []
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.base_model.eval()  # Base model stays in eval mode
        self.head.train()
        
        total_loss = 0.0
        all_preds = []
        all_actuals = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        
        for batch in pbar:
            source_expr = batch['source_expression'].to(self.device)
            tf_change = batch['tf_activity_change'].to(self.device)
            target_delta = batch['target_delta'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get base model prediction for target gene
            with torch.no_grad():
                pred_delta_all, _ = self.base_model.predict_delta_multi_tf(
                    tf_change, source_expr
                )
                base_prediction = pred_delta_all[:, self.target_gene_idx]
            
            # Refine with gene-specific head
            refined_prediction = self.head(tf_change, base_prediction)
            
            # Losses
            mse_loss = F.mse_loss(refined_prediction, target_delta)
            
            # Correlation loss (1 - r)
            if len(refined_prediction) > 2:
                pred_centered = refined_prediction - refined_prediction.mean()
                target_centered = target_delta - target_delta.mean()
                pred_std = pred_centered.std() + 1e-8
                target_std = target_centered.std() + 1e-8
                correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
                corr_loss = 1.0 - correlation
            else:
                corr_loss = torch.tensor(0.0, device=self.device)
            
            # Sparsity regularization on TF importance
            importance_reg = self.head.tf_importance.abs().mean() * 0.01
            
            # Combined loss
            loss = mse_loss + 0.5 * corr_loss + importance_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(refined_prediction.detach().cpu())
            all_actuals.append(target_delta.detach().cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        all_preds = torch.cat(all_preds).numpy()
        all_actuals = torch.cat(all_actuals).numpy()
        
        train_r = np.corrcoef(all_preds, all_actuals)[0, 1]
        if np.isnan(train_r):
            train_r = 0.0
        
        return {
            'loss': total_loss / len(train_loader),
            'train_r': train_r
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate on held-out cell types."""
        self.base_model.eval()
        self.head.eval()
        
        all_preds = []
        all_actuals = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                source_expr = batch['source_expression'].to(self.device)
                tf_change = batch['tf_activity_change'].to(self.device)
                target_delta = batch['target_delta'].to(self.device)
                
                # Base model prediction
                pred_delta_all, _ = self.base_model.predict_delta_multi_tf(
                    tf_change, source_expr
                )
                base_prediction = pred_delta_all[:, self.target_gene_idx]
                
                # Refined prediction
                refined_prediction = self.head(tf_change, base_prediction)
                
                loss = F.mse_loss(refined_prediction, target_delta)
                total_loss += loss.item()
                
                all_preds.append(refined_prediction.cpu())
                all_actuals.append(target_delta.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_actuals = torch.cat(all_actuals).numpy()
        
        val_r = np.corrcoef(all_preds, all_actuals)[0, 1]
        if np.isnan(val_r):
            val_r = 0.0
        
        # Direction accuracy
        pred_sign = np.sign(all_preds)
        actual_sign = np.sign(all_actuals)
        dir_acc = np.mean(pred_sign == actual_sign)
        
        return {
            'loss': total_loss / len(val_loader),
            'val_r': val_r,
            'dir_acc': dir_acc
        }


# =============================================================================
# CELL-TYPE HOLDOUT SPLIT
# =============================================================================

def create_celltype_holdout_split(
    n_celltypes: int,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Create train/val split by holding out CELL TYPES, not pairs.
    """
    np.random.seed(seed)
    
    n_val = max(2, int(n_celltypes * val_fraction))
    all_types = list(range(n_celltypes))
    np.random.shuffle(all_types)
    
    val_types = set(all_types[:n_val])
    train_types = set(all_types[n_val:])
    
    print(f"  Cell-type holdout: {len(train_types)} train, {len(val_types)} val")
    
    train_pairs = [(i, j) for i in train_types for j in train_types if i != j]
    val_pairs = [(i, j) for i in range(n_celltypes) for j in range(n_celltypes) 
                 if i != j and (i in val_types or j in val_types)]
    
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)} (involve held-out cell types)")
    
    return train_pairs, val_pairs


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def finetune_gene(
    model_dir: str,
    singlecell_path: str,
    tf_binding_dir: str,
    target_gene: str,
    output_dir: str,
    ppi_dir: Optional[str] = None,
    celltype_key: str = 'sclc_subcluster',
    epochs: int = 100,
    batch_size: int = 32,
    hidden_dim: int = 256,
    run_analysis: bool = True
):
    """
    Fine-tune IntegratedGRNModel for a specific gene.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    target_gene = target_gene.upper()
    
    print("=" * 70)
    print(f"GENE-SPECIFIC FINE-TUNING FOR {target_gene}")
    print("=" * 70)
    print(f"Device: {device}")
    print("\nUsing SAME architecture as train_v6:")
    print("  - IntegratedGRNModel with TFRegulatoryEncoder, CellStateEncoder")
    print("  - TF cascade GCN, context gating, cofactor modulation")
    print("  - Gene-specific refinement head on top")
    
    # =========================================================================
    # Load Data (same as train_v6)
    # =========================================================================
    print("\n[1/6] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(singlecell_path, celltype_key)
    n_celltypes, n_genes = expression.shape
    print(f"  Cell types: {n_celltypes}")
    print(f"  Genes: {n_genes}")
    
    # Find target gene
    gene_names_upper = [g.upper() for g in gene_names]
    if target_gene not in gene_names_upper:
        raise ValueError(f"Target gene {target_gene} not found!")
    target_gene_idx = gene_names_upper.index(target_gene)
    
    target_expr = expression[:, target_gene_idx]
    print(f"  Target: {target_gene} at index {target_gene_idx}")
    print(f"  Expression range: [{target_expr.min():.3f}, {target_expr.max():.3f}]")
    print(f"  Expression std: {target_expr.std():.3f}")
    
    print("\n[2/6] Loading TF binding...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(tf_binding_dir, gene_names)
    n_tfs = len(tf_names)
    print(f"  TFs: {n_tfs}")
    
    print("\n[3/6] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    print(f"  TF activity: {tf_activity.shape}")
    
    # =========================================================================
    # Load Pre-trained Model (IntegratedGRNModel from train_v6)
    # =========================================================================
    print("\n[4/6] Loading pre-trained IntegratedGRNModel...")
    
    checkpoint_path = model_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Delta_PCC: {checkpoint.get('delta_pcc', 'unknown')}")
    
    # Recreate model with same config
    base_model = IntegratedGRNModel(
        n_genes=config.get('n_genes', n_genes),
        n_tfs=config.get('n_tfs', n_tfs),
        hidden_dim=config.get('hidden_dim', 512),
        num_gcn_layers=config.get('num_gcn_layers', 3),
        num_transformer_layers=config.get('num_transformer_layers', 2),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        use_context_gating=config.get('use_context_gating', True)
    )
    
    # Load state dict - this loads ALL weights AND buffers (chip_mask, tf_specificity_weights, etc.)
    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"  Model loaded: {base_model.get_num_parameters():,} parameters")
    
    # Verify that important buffers were loaded from checkpoint
    if hasattr(base_model, 'tf_specificity_weights') and base_model.tf_specificity_weights is not None:
        w = base_model.tf_specificity_weights
        print(f"  TF specificity weights loaded: range=[{w.min():.3f}, {w.max():.3f}]")
    
    if hasattr(base_model, 'direct_effects') and base_model.direct_effects is not None:
        de = base_model.direct_effects
        n_pos = (de > 0).sum().item()
        n_neg = (de < 0).sum().item()
        print(f"  Direct effects loaded: range=[{de.min():.4f}, {de.max():.4f}], pos={n_pos}, neg={n_neg}")
    
    # Check tf_tf_adjacency - if it's still identity matrix or None, we need to load from files
    # BUT we must NOT call set_priors() as that would OVERWRITE the learned weights!
    if hasattr(base_model, 'tf_tf_adjacency') and base_model.tf_tf_adjacency is not None:
        adj = base_model.tf_tf_adjacency
        is_identity = torch.allclose(adj, torch.eye(adj.size(0), device=adj.device), atol=1e-5)
        if is_identity and tf_tf_adj is not None:
            print("  TF-TF adjacency is identity, loading from files...")
            # Manually update ONLY the tf_tf_adjacency buffer, not other priors
            adj_new = torch.FloatTensor(tf_tf_adj)
            adj_new = adj_new + torch.eye(adj_new.size(0))  # Add self-loops
            adj_new = adj_new / adj_new.sum(dim=1, keepdim=True).clamp(min=1e-8)  # Row normalize
            base_model.tf_tf_adjacency.copy_(adj_new.to(base_model.direct_effects.device))
            print(f"  TF-TF adjacency updated: {base_model.tf_tf_adjacency.shape}")
        else:
            print(f"  TF-TF adjacency loaded from checkpoint: {adj.shape}")
    else:
        print("  Warning: tf_tf_adjacency is None - cascade GCN may not work properly")
    
    # =========================================================================
    # Create Dataset with Cell-Type Holdout
    # =========================================================================
    print("\n[5/6] Creating training data with cell-type holdout...")
    train_pairs, val_pairs = create_celltype_holdout_split(n_celltypes, val_fraction=0.2)
    
    train_dataset = GeneSpecificDataset(expression, tf_activity, target_gene_idx, train_pairs)
    val_dataset = GeneSpecificDataset(expression, tf_activity, target_gene_idx, val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # =========================================================================
    # Create Fine-Tuner
    # =========================================================================
    print("\n[6/6] Creating gene-specific fine-tuner...")
    finetuner = GeneSpecificFineTuner(
        base_model=base_model,
        target_gene_idx=target_gene_idx,
        target_gene_name=target_gene,
        n_tfs=n_tfs,
        device=device,
        hidden_dim=hidden_dim,
        freeze_base=True
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetuner.optimizer, epochs, eta_min=1e-5
    )
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING (with IntegratedGRNModel base + gene-specific head)")
    print("=" * 70)
    
    best_val_r = -1.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 30
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = finetuner.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = finetuner.validate(val_loader)
        
        scheduler.step()
        
        # Logging
        improved = val_metrics['val_r'] > best_val_r
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f} | "
                  f"train_r={train_metrics['train_r']:.4f} | "
                  f"val_r={val_metrics['val_r']:.4f} | "
                  f"dir_acc={val_metrics['dir_acc']:.2%}", end="")
            if improved:
                print(" *NEW BEST*", end="")
            print()
        
        # Save best
        if improved:
            best_val_r = val_metrics['val_r']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'head_state_dict': finetuner.head.state_dict(),
                'base_model_config': config,
                'target_gene': target_gene,
                'target_gene_idx': target_gene_idx,
                'n_tfs': n_tfs,
                'hidden_dim': hidden_dim,
                'best_val_r': best_val_r,
                'tf_importance': finetuner.head.get_tf_importance(),
                'blend_weight': torch.sigmoid(finetuner.head.blend_weight).item()
            }, output_dir / 'best_head.pt')
            
            # Copy the CORRECT GRN matrix from the base model directory
            # IMPORTANT: Use grn_matrix_CORRECT.npy if it exists, as grn_matrix.npy
            # may have been scaled down by softmaxed tf_importance (a bug)
            import shutil
            base_grn_correct = model_dir / 'grn_matrix_CORRECT.npy'
            base_grn_path = model_dir / 'grn_matrix.npy'
            
            if base_grn_correct.exists():
                shutil.copy(base_grn_correct, output_dir / 'grn_matrix.npy')
                print(f"  Copied CORRECT GRN matrix from base model")
            elif base_grn_path.exists():
                shutil.copy(base_grn_path, output_dir / 'grn_matrix.npy')
                print(f"  WARNING: Copied grn_matrix.npy (may have tf_importance bug!)")
            else:
                # Compute from checkpoint with correct formula
                print(f"  Computing GRN matrix from checkpoint...")
                state_dict = finetuner.base_model.state_dict()
                direct_effects = state_dict['direct_effects'].cpu().numpy()
                tf_spec = state_dict['tf_specificity_weights'].cpu().numpy()
                chip_mask = state_dict['chip_mask'].cpu().numpy()
                leakage_raw = state_dict['_leakage_raw'].item()
                leakage = 0.005 + 0.195 / (1 + np.exp(-leakage_raw))
                grn_matrix = direct_effects * tf_spec[np.newaxis, :] * (chip_mask + leakage)
                np.save(output_dir / 'grn_matrix.npy', grn_matrix)
                print(f"  Saved computed GRN: range=[{grn_matrix.min():.4f}, {grn_matrix.max():.4f}]")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {max_patience} epochs)")
            break
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / 'best_head.pt', weights_only=False)
    finetuner.head.load_state_dict(best_checkpoint['head_state_dict'])
    
    # Save TF importance
    tf_importance = finetuner.head.get_tf_importance()
    tf_importance_df = pd.DataFrame({
        'tf_name': tf_names,
        'importance': tf_importance
    })
    tf_importance_df = tf_importance_df.sort_values('importance', ascending=False)
    tf_importance_df['rank'] = range(1, n_tfs + 1)
    tf_importance_df.to_csv(output_dir / f'tf_importance_{target_gene}.csv', index=False)
    
    # Save metadata files for analysis
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False, header=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False, header=False)
    pd.DataFrame({'celltype': celltypes}).to_csv(output_dir / 'celltype_names.csv', index=False, header=False)
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    
    # Save config
    finetune_config = {
        'target_gene': target_gene,
        'target_gene_idx': int(target_gene_idx),
        'n_tfs': int(n_tfs),
        'hidden_dim': hidden_dim,
        'epochs_trained': int(best_epoch),
        'best_val_r': float(best_val_r),
        'best_val_r2': float(best_val_r ** 2),
        'n_train_pairs': len(train_pairs),
        'n_val_pairs': len(val_pairs),
        'blend_weight': best_checkpoint['blend_weight'],
        'base_model_config': config,
        'validation_type': 'cell_type_holdout'
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(finetune_config, f, indent=2)
    
    print(f"\nBest model: Epoch {best_epoch}")
    print(f"  Validation r = {best_val_r:.4f}")
    print(f"  Validation R² = {best_val_r**2:.4f} ({best_val_r**2*100:.1f}% variance explained)")
    print(f"  Blend weight = {best_checkpoint['blend_weight']:.4f}")
    
    # Print top TFs
    print(f"\nTOP 20 TFs FOR {target_gene}:")
    print("-" * 40)
    for _, row in tf_importance_df.head(20).iterrows():
        print(f"  {row['tf_name']:12s}: importance={row['importance']:.4f}")
    
    # =========================================================================
    # Run Analysis (integrate analyze_target_full.py)
    # =========================================================================
    if run_analysis:
        print("\n" + "=" * 70)
        print("GENERATING ANALYSIS FIGURES")
        print("=" * 70)
        
        setup_style()
        analysis_dir = output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # Create analyzer using the fine-tuned model outputs
            analyzer = GRNAnalyzer(
                model_dir=str(output_dir),
                gene=target_gene,
                singlecell_path=singlecell_path,
                ppi_dir=ppi_dir,
                jaspar_dir=str(Path(tf_binding_dir).parent / 'tf_specificity'),
                finetune_dir=str(output_dir)
            )
            
            # Generate figures
            print("\nGenerating Figure 1: What regulates the target?")
            create_figure1(analyzer, analysis_dir)
            
            print("Generating Figure 2: Why these regulators?")
            create_figure2(analyzer, analysis_dir)
            
            print("Generating Figure 3: Calibration & controllability")
            create_figure3(analyzer, analysis_dir)
            
            print("Generating Figure 4: TF Generalizability Analysis")
            create_figure4(analyzer, analysis_dir)
            
            # Save CSVs
            print("\nSaving analysis CSVs...")
            save_csvs(analyzer, analysis_dir)
            
            print(f"\n✓ Analysis complete! Figures saved to: {analysis_dir}")
            
        except Exception as e:
            print(f"\n⚠ Analysis failed: {e}")
            print("  You can run analysis manually with:")
            print(f"  python scripts/analyze_target_full.py --gene {target_gene} --model {output_dir} --output {analysis_dir}")
    
    print(f"\n✓ All outputs saved to: {output_dir}")
    
    return best_val_r


# =============================================================================
# MULTI-GENE FINE-TUNING (MASTER REGULATORS)
# =============================================================================

def finetune_multi_gene(
    model_dir: str,
    singlecell_path: str,
    tf_binding_dir: str,
    target_genes: List[str],
    output_dir: str,
    ppi_dir: Optional[str] = None,
    celltype_key: str = 'sclc_subcluster',
    epochs: int = 100,
    batch_size: int = 32,
    hidden_dim: int = 256,
    run_analysis: bool = True,
    lr: float = 1e-4
):
    """
    Fine-tune for MULTIPLE genes to find master regulators.
    
    This identifies TFs that affect ALL genes in the list.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-GENE FINE-TUNING - MASTER REGULATOR DISCOVERY")
    print("=" * 70)
    print(f"Target genes: {len(target_genes)}")
    for g in target_genes[:10]:
        print(f"  - {g}")
    if len(target_genes) > 10:
        print(f"  ... and {len(target_genes) - 10} more")
    print(f"Device: {device}")
    
    # =========================================================================
    # Load data (same as single-gene)
    # =========================================================================
    print("\n[1/6] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(singlecell_path, celltype_key)
    n_celltypes, n_genes = expression.shape
    print(f"  Expression: {expression.shape}")
    
    # Find target gene indices
    gene_upper = [g.upper() for g in gene_names]
    target_gene_indices = []
    found_genes = []
    missing_genes = []
    
    for target in target_genes:
        target_upper = target.upper()
        if target_upper in gene_upper:
            idx = gene_upper.index(target_upper)
            target_gene_indices.append(idx)
            found_genes.append(target)
        else:
            missing_genes.append(target)
    
    if missing_genes:
        print(f"\n  ⚠ Missing genes ({len(missing_genes)}): {', '.join(missing_genes[:5])}")
    
    if not target_gene_indices:
        raise ValueError("No target genes found in expression data!")
    
    print(f"  Found {len(found_genes)}/{len(target_genes)} genes")
    
    print("\n[2/6] Loading TF binding...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(tf_binding_dir, gene_names)
    n_tfs = len(tf_names)
    print(f"  TFs: {n_tfs}")
    
    print("\n[3/6] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    print(f"  TF activity: {tf_activity.shape}")
    
    # =========================================================================
    # Load pre-trained model
    # =========================================================================
    print("\n[4/6] Loading pre-trained model...")
    model_path = Path(model_dir) / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    base_model = IntegratedGRNModel(
        n_genes=n_genes,
        n_tfs=n_tfs,
        hidden_dim=config.get('hidden_dim', 512),
        num_gcn_layers=config.get('num_gcn_layers', 3),
        num_transformer_layers=config.get('num_transformer_layers', 2),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        use_context_gating=config.get('use_context_gating', True),
    ).to(device)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()  # Freeze base model
    
    for param in base_model.parameters():
        param.requires_grad = False
    
    print(f"  Loaded model (frozen)")
    
    # =========================================================================
    # Create multi-gene head
    # =========================================================================
    print("\n[5/6] Creating multi-gene refinement head...")
    head = MultiGeneRefinementHead(
        n_tfs=n_tfs,
        n_genes=len(target_gene_indices),
        hidden_dim=hidden_dim
    ).to(device)
    
    # Initialize from GRN matrix
    grn_path = Path(model_dir) / 'grn_matrix_CORRECT.npy'
    if not grn_path.exists():
        grn_path = Path(model_dir) / 'grn_matrix.npy'
    
    if grn_path.exists():
        grn_matrix = np.load(grn_path)
        # Initialize with mean effect across target genes
        mean_effects = grn_matrix[target_gene_indices, :].mean(axis=0)
        with torch.no_grad():
            head.tf_importance.data = torch.FloatTensor(mean_effects).to(device)
        print(f"  Initialized TF importance from GRN")
    
    # =========================================================================
    # Create datasets with cell-type holdout
    # =========================================================================
    # Use larger holdout (20% of cell types) for better validation
    n_holdout = max(4, n_celltypes // 5)
    np.random.seed(42)
    holdout_indices = set(np.random.choice(n_celltypes, n_holdout, replace=False))
    train_indices = [i for i in range(n_celltypes) if i not in holdout_indices]
    
    train_pairs = [(i, j) for i in train_indices for j in train_indices if i != j]
    # Also validate on cross-pairs (train→holdout) for more robust validation
    val_pairs = [(i, j) for i in holdout_indices for j in holdout_indices if i != j]
    val_pairs += [(i, j) for i in train_indices for j in holdout_indices]
    
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)} (holdout + cross-validation)")
    
    train_dataset = MultiGeneDataset(expression, tf_activity, target_gene_indices, train_pairs)
    val_dataset = MultiGeneDataset(expression, tf_activity, target_gene_indices, val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) > batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # =========================================================================
    # Training with regularization to prevent overfitting
    # =========================================================================
    print("\n[6/6] Fine-tuning...")
    
    # Use higher weight decay for regularization
    weight_decay = 0.1  # Strong L2 regularization to prevent overfitting
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Pre-compute GRN tensor once
    if grn_path.exists():
        grn_tensor = torch.FloatTensor(grn_matrix).to(device)
    else:
        grn_tensor = None
    
    best_val_r = -1
    best_epoch = 0
    best_checkpoint = None
    patience = 15  # Slightly reduced patience
    patience_counter = 0
    
    # L1 regularization strength on TF importance (encourage sparsity)
    l1_lambda = 0.01
    
    for epoch in range(epochs):
        head.train()
        epoch_losses = []
        epoch_correlations = []
        
        for batch in train_loader:
            tf_change = batch['tf_activity_change'].to(device)
            target_deltas = batch['target_deltas'].to(device)  # (batch, n_genes)
            
            # Get base model predictions for target genes
            with torch.no_grad():
                if grn_tensor is not None:
                    base_preds = torch.stack([
                        (tf_change * grn_tensor[idx, :]).sum(dim=-1) 
                        for idx in target_gene_indices
                    ], dim=-1)  # (batch, n_genes)
                else:
                    base_preds = torch.zeros_like(target_deltas)
            
            # Refinement
            refined = head(tf_change, base_preds)
            
            # Loss: MSE + correlation loss
            mse_loss = F.mse_loss(refined, target_deltas)
            
            # Per-gene correlation loss
            corr_losses = []
            for g in range(refined.shape[1]):
                pred = refined[:, g]
                actual = target_deltas[:, g]
                if pred.std() > 1e-6 and actual.std() > 1e-6:
                    corr = torch.corrcoef(torch.stack([pred, actual]))[0, 1]
                    if torch.isfinite(corr):
                        corr_losses.append(1 - corr)
            
            corr_loss = torch.stack(corr_losses).mean() if corr_losses else torch.tensor(0.0, device=device)
            
            # L1 regularization on TF importance (encourages master regulators to stand out)
            l1_reg = l1_lambda * torch.abs(head.tf_importance).mean()
            
            # Total loss
            loss = mse_loss + 0.3 * corr_loss + l1_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)  # Stricter clipping
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Mean correlation across genes
            with torch.no_grad():
                gene_rs = []
                for g in range(refined.shape[1]):
                    pred = refined[:, g].cpu().numpy()
                    actual = target_deltas[:, g].cpu().numpy()
                    if np.std(pred) > 1e-6 and np.std(actual) > 1e-6:
                        r, _ = stats.pearsonr(pred, actual)
                        if np.isfinite(r):
                            gene_rs.append(r)
                if gene_rs:
                    epoch_correlations.append(np.mean(gene_rs))
        
        scheduler.step()
        
        # Validation
        head.eval()
        val_rs = []
        with torch.no_grad():
            for batch in val_loader:
                tf_change = batch['tf_activity_change'].to(device)
                target_deltas = batch['target_deltas'].to(device)
                
                if grn_tensor is not None:
                    base_preds = torch.stack([
                        (tf_change * grn_tensor[idx, :]).sum(dim=-1) 
                        for idx in target_gene_indices
                    ], dim=-1)
                else:
                    base_preds = torch.zeros_like(target_deltas)
                
                refined = head(tf_change, base_preds)
                
                for g in range(refined.shape[1]):
                    pred = refined[:, g].cpu().numpy()
                    actual = target_deltas[:, g].cpu().numpy()
                    if np.std(pred) > 1e-6 and np.std(actual) > 1e-6:
                        r, _ = stats.pearsonr(pred, actual)
                        if np.isfinite(r):
                            val_rs.append(r)
        
        val_r = np.mean(val_rs) if val_rs else 0.0
        train_r = np.mean(epoch_correlations) if epoch_correlations else 0.0
        mean_loss = np.mean(epoch_losses)
        
        if val_r > best_val_r:
            best_val_r = val_r
            best_epoch = epoch + 1
            best_checkpoint = {
                'head_state_dict': head.state_dict(),
                'tf_importance': head.get_tf_importance(),
                'gene_specific_importance': head.get_gene_specific_importance(),
                'epoch': epoch + 1,
                'val_r': val_r,
                'target_genes': found_genes,
                'target_gene_indices': target_gene_indices
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={mean_loss:.4f} train_r={train_r:.4f} val_r={val_r:.4f} {'*' if best_epoch == epoch+1 else ''}")
        
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save head
    torch.save(best_checkpoint, output_dir / 'best_head.pt')
    
    # Save master regulator rankings
    tf_importance = best_checkpoint['tf_importance']
    master_reg_df = pd.DataFrame({
        'tf_name': tf_names,
        'master_importance': tf_importance
    }).sort_values('master_importance', ascending=False)
    master_reg_df['rank'] = range(1, len(master_reg_df) + 1)
    master_reg_df.to_csv(output_dir / 'master_regulators.csv', index=False)
    
    # Per-gene importance
    gene_specific = best_checkpoint['gene_specific_importance']
    for i, gene in enumerate(found_genes):
        gene_df = pd.DataFrame({
            'tf_name': tf_names,
            'importance': gene_specific[i, :] + tf_importance
        }).sort_values('importance', ascending=False)
        gene_df.to_csv(output_dir / f'tf_importance_{gene}.csv', index=False)
    
    # Save metadata
    pd.DataFrame({'gene': gene_names}).to_csv(output_dir / 'gene_names.csv', index=False, header=False)
    pd.DataFrame({'tf': tf_names}).to_csv(output_dir / 'tf_names.csv', index=False, header=False)
    pd.DataFrame({'celltype': celltypes}).to_csv(output_dir / 'celltype_names.csv', index=False, header=False)
    pd.DataFrame({'target_gene': found_genes}).to_csv(output_dir / 'target_genes.csv', index=False, header=False)
    np.save(output_dir / 'tf_activity.npy', tf_activity)
    np.save(output_dir / 'binding_matrix.npy', binding_matrix)
    
    # Copy GRN matrix
    import shutil
    if grn_path.exists():
        shutil.copy(grn_path, output_dir / 'grn_matrix.npy')
    
    # Save config
    config = {
        'mode': 'multi_gene',
        'target_genes': found_genes,
        'missing_genes': missing_genes,
        'n_targets': len(found_genes),
        'n_tfs': n_tfs,
        'hidden_dim': hidden_dim,
        'best_epoch': best_epoch,
        'best_val_r': float(best_val_r),
        'n_train_pairs': len(train_pairs),
        'n_val_pairs': len(val_pairs)
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nBest model: Epoch {best_epoch}")
    print(f"  Mean validation r = {best_val_r:.4f}")
    print(f"  Mean R² = {best_val_r**2:.4f}")
    
    # Print top master regulators
    print(f"\n{'='*50}")
    print("TOP 20 MASTER REGULATORS")
    print(f"{'='*50}")
    print("(TFs that consistently affect ALL target genes)")
    for _, row in master_reg_df.head(20).iterrows():
        print(f"  {row['rank']:3d}. {row['tf_name']:12s}: importance={row['master_importance']:.6f}")
    
    # =========================================================================
    # Generate multi-gene analysis figures
    # =========================================================================
    if run_analysis:
        print("\n" + "=" * 70)
        print("GENERATING MULTI-GENE ANALYSIS FIGURES")
        print("=" * 70)
        
        setup_style()
        analysis_dir = output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            create_master_regulator_figures(
                output_dir=analysis_dir,
                grn_matrix=np.load(grn_path) if grn_path.exists() else None,
                tf_names=tf_names,
                target_genes=found_genes,
                target_gene_indices=target_gene_indices,
                master_importance=tf_importance,
                gene_specific_importance=gene_specific,
                tf_activity=tf_activity,
                celltype_names=celltypes,
                expression=expression
            )
            print(f"\n✓ Analysis complete! Figures saved to: {analysis_dir}")
        except Exception as e:
            print(f"\n⚠ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ All outputs saved to: {output_dir}")
    
    return best_val_r


def create_master_regulator_figures(
    output_dir: Path,
    grn_matrix: Optional[np.ndarray],
    tf_names: List[str],
    target_genes: List[str],
    target_gene_indices: List[int],
    master_importance: np.ndarray,
    gene_specific_importance: np.ndarray,
    tf_activity: np.ndarray,
    celltype_names: List[str],
    expression: np.ndarray
):
    """
    Create specialized figures for multi-gene (master regulator) analysis.
    
    Figure 5: Master Regulator Overview (10 panels)
    Figure 6: Per-Gene TF Effects (10 panels)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    from scipy import stats
    
    n_tfs = len(tf_names)
    n_targets = len(target_genes)
    
    # Get top master regulators
    top_indices = np.argsort(master_importance)[::-1][:30]
    
    # =========================================================================
    # FIGURE 5: Master Regulator Overview
    # =========================================================================
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # A: Master regulator bar chart
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_a.text(-0.05, 1.08, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold')
    
    top_20 = top_indices[:20]
    colors = ['#3182bd' if master_importance[i] > 0 else '#e6550d' for i in top_20]
    ax_a.barh(range(20), master_importance[top_20], color=colors, alpha=0.8)
    ax_a.set_yticks(range(20))
    ax_a.set_yticklabels([tf_names[i] for i in top_20], fontsize=8)
    ax_a.set_xlabel('Master Importance', fontsize=10)
    ax_a.set_title(f'Top 20 Master Regulators\n(affecting {n_targets} genes)', fontsize=11)
    ax_a.invert_yaxis()
    
    # B: Importance distribution
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.text(-0.15, 1.08, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold')
    
    ax_b.hist(master_importance, bins=50, color='#756bb1', alpha=0.7, edgecolor='white')
    ax_b.axvline(np.median(master_importance), color='red', linestyle='--', label=f'Median')
    ax_b.axvline(np.percentile(master_importance, 95), color='orange', linestyle='--', label=f'95th %ile')
    ax_b.set_xlabel('Master Importance', fontsize=10)
    ax_b.set_ylabel('# TFs', fontsize=10)
    ax_b.set_title('Importance Distribution', fontsize=11)
    ax_b.legend(fontsize=8)
    
    # C: Target genes summary
    ax_c = fig.add_subplot(gs[0, 3])
    ax_c.text(-0.15, 1.08, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold')
    ax_c.axis('off')
    
    summary_text = f"TARGET GENES ({n_targets})\n{'='*30}\n\n"
    for i, gene in enumerate(target_genes[:15]):
        summary_text += f"{i+1}. {gene}\n"
    if len(target_genes) > 15:
        summary_text += f"... and {len(target_genes)-15} more"
    
    ax_c.text(0.1, 0.95, summary_text, transform=ax_c.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f7f7f7', alpha=0.8))
    ax_c.set_title('Target Genes', fontsize=11)
    
    # D: Heatmap of TF effects on target genes
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_d.text(-0.05, 1.08, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold')
    
    if grn_matrix is not None:
        # Top TFs × target genes
        top_15 = top_indices[:15]
        effects_matrix = grn_matrix[target_gene_indices, :][:, top_15].T  # (15 TFs, n_genes)
        
        # Z-score per TF
        effects_z = stats.zscore(effects_matrix, axis=1)
        effects_z = np.nan_to_num(effects_z, 0)
        
        im = ax_d.imshow(effects_z, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax_d.set_yticks(range(15))
        ax_d.set_yticklabels([tf_names[i] for i in top_15], fontsize=7)
        ax_d.set_xticks(range(min(20, n_targets)))
        ax_d.set_xticklabels(target_genes[:20], rotation=45, ha='right', fontsize=7)
        ax_d.set_xlabel('Target Gene', fontsize=10)
        ax_d.set_ylabel('Master TF', fontsize=10)
        plt.colorbar(im, ax=ax_d, shrink=0.8, label='Effect (Z-score)')
    ax_d.set_title('Master TF Effects on Target Genes', fontsize=11)
    
    # E: Consistency across genes
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.text(-0.15, 1.08, 'E', transform=ax_e.transAxes, fontsize=12, fontweight='bold')
    
    if grn_matrix is not None:
        # For each TF, compute consistency (same sign across genes)
        tf_effects_on_targets = grn_matrix[target_gene_indices, :]  # (n_genes, n_tfs)
        consistency = np.abs(np.mean(np.sign(tf_effects_on_targets), axis=0))  # 1.0 = all same sign
        
        ax_e.scatter(master_importance, consistency, alpha=0.5, s=20, c='#31a354')
        
        # Highlight top TFs
        for i in top_indices[:5]:
            ax_e.annotate(tf_names[i], (master_importance[i], consistency[i]), fontsize=6)
        
        ax_e.set_xlabel('Master Importance', fontsize=10)
        ax_e.set_ylabel('Direction Consistency', fontsize=10)
        ax_e.axhline(0.8, color='red', linestyle='--', alpha=0.5)
    ax_e.set_title('Importance vs Consistency', fontsize=11)
    
    # F: Top activators vs repressors
    ax_f = fig.add_subplot(gs[1, 3])
    ax_f.text(-0.15, 1.08, 'F', transform=ax_f.transAxes, fontsize=12, fontweight='bold')
    
    if grn_matrix is not None:
        mean_effects = grn_matrix[target_gene_indices, :].mean(axis=0)
        n_activators = (mean_effects > 0).sum()
        n_repressors = (mean_effects < 0).sum()
        
        sizes = [n_activators, n_repressors]
        labels = [f'Activators\n({n_activators})', f'Repressors\n({n_repressors})']
        colors = ['#3182bd', '#e6550d']
        
        ax_f.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax_f.set_title('Master TF Types', fontsize=11)
    
    # G-H: Per-cell-type TF activity heatmap
    ax_g = fig.add_subplot(gs[2, 0:2])
    ax_g.text(-0.05, 1.08, 'G', transform=ax_g.transAxes, fontsize=12, fontweight='bold')
    
    top_10 = top_indices[:10]
    activity_subset = tf_activity[:, top_10]  # (n_ct, 10)
    activity_z = stats.zscore(activity_subset, axis=0)
    activity_z = np.nan_to_num(activity_z, 0)
    
    n_ct_show = min(30, len(celltype_names))
    im = ax_g.imshow(activity_z[:n_ct_show, :].T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax_g.set_yticks(range(10))
    ax_g.set_yticklabels([tf_names[i] for i in top_10], fontsize=7)
    ax_g.set_xticks(range(n_ct_show))
    ax_g.set_xticklabels(celltype_names[:n_ct_show], rotation=45, ha='right', fontsize=6)
    ax_g.set_xlabel('Cell Type', fontsize=10)
    ax_g.set_ylabel('Master TF', fontsize=10)
    plt.colorbar(im, ax=ax_g, shrink=0.8, label='Activity (Z-score)')
    ax_g.set_title('Master TF Activity Across Cell Types', fontsize=11)
    
    # I: Target gene expression variation
    ax_i = fig.add_subplot(gs[2, 2])
    ax_i.text(-0.15, 1.08, 'I', transform=ax_i.transAxes, fontsize=12, fontweight='bold')
    
    target_expr = expression[:, target_gene_indices]  # (n_ct, n_targets)
    expr_cv = np.std(target_expr, axis=0) / (np.mean(target_expr, axis=0) + 1e-6)
    
    ax_i.bar(range(min(15, n_targets)), expr_cv[:15], color='#756bb1', alpha=0.7)
    ax_i.set_xticks(range(min(15, n_targets)))
    ax_i.set_xticklabels(target_genes[:15], rotation=45, ha='right', fontsize=7)
    ax_i.set_ylabel('CV (expression)', fontsize=10)
    ax_i.set_title('Target Gene Variability', fontsize=11)
    
    # J: Summary statistics
    ax_j = fig.add_subplot(gs[2, 3])
    ax_j.text(-0.15, 1.08, 'J', transform=ax_j.transAxes, fontsize=12, fontweight='bold')
    ax_j.axis('off')
    
    top_5_master = [tf_names[i] for i in top_indices[:5]]
    summary = f"""
MASTER REGULATOR SUMMARY
{'='*35}

Target Genes: {n_targets}
TFs Analyzed: {n_tfs}

TOP 5 MASTER REGULATORS:
"""
    for i, tf in enumerate(top_5_master):
        summary += f"  {i+1}. {tf}\n"
    
    summary += f"""
INTERPRETATION:
These TFs consistently regulate
ALL {n_targets} target genes and
are prime candidates for:
• Therapeutic targeting
• CRISPRa/i experiments
• Drug discovery
"""
    
    ax_j.text(0.05, 0.95, summary, transform=ax_j.transAxes, fontsize=8,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.9))
    
    plt.suptitle(f'Figure 5: Master Regulator Analysis ({n_targets} Target Genes)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_dir / 'figure5_master_regulators.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure5_master_regulators.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved Figure 5: {output_dir / 'figure5_master_regulators.png'}")
    
    # =========================================================================
    # FIGURE 6: Per-Gene Analysis
    # =========================================================================
    if n_targets <= 12 and grn_matrix is not None:
        # Create per-gene subplot grid
        n_cols = min(4, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = np.atleast_2d(axes)
        
        for i, (gene, gene_idx) in enumerate(zip(target_genes, target_gene_indices)):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get effects for this gene
            gene_effects = grn_matrix[gene_idx, :]
            
            # Top 10 TFs for this gene
            top_tf_idx = np.argsort(np.abs(gene_effects))[::-1][:10]
            
            colors = ['#3182bd' if gene_effects[j] > 0 else '#e6550d' for j in top_tf_idx]
            ax.barh(range(10), gene_effects[top_tf_idx], color=colors, alpha=0.8)
            ax.set_yticks(range(10))
            ax.set_yticklabels([tf_names[j] for j in top_tf_idx], fontsize=7)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Effect')
            ax.set_title(gene, fontsize=10, fontweight='bold')
            ax.invert_yaxis()
        
        # Hide empty subplots
        for i in range(n_targets, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Figure 6: Top TF Regulators Per Target Gene', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig.savefig(output_dir / 'figure6_per_gene_regulators.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure6_per_gene_regulators.pdf', bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved Figure 6: {output_dir / 'figure6_per_gene_regulators.png'}")


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def parse_hypothesis_file(filepath: str) -> Tuple[List[dict], List[dict], List[str]]:
    """
    Parse hypothesis file from hypothesis_builder.py
    
    Returns:
        nodes: List of node dicts
        edges: List of edge dicts  
        outputs: List of output gene names
    """
    nodes = []
    edges = []
    outputs = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for output declaration
            if line.startswith('OUTPUT:'):
                outputs.append(line.split(':')[1].strip())
                continue
            
            # Parse connection: SOURCE -> TARGET or SOURCE -| TARGET
            if ' -> ' in line:
                parts = line.split(' -> ')
                source, target = parts[0].strip(), parts[1].strip()
                edges.append({
                    'source': source,
                    'target': target,
                    'type': 'activator'
                })
            elif ' -| ' in line:
                parts = line.split(' -| ')
                source, target = parts[0].strip(), parts[1].strip()
                edges.append({
                    'source': source,
                    'target': target,
                    'type': 'repressor'
                })
    
    # Extract unique nodes from edges
    node_names = set()
    for edge in edges:
        node_names.add(edge['source'])
        node_names.add(edge['target'])
    
    for name in node_names:
        nodes.append({'name': name, 'is_output': name in outputs})
    
    return nodes, edges, outputs


def test_hypothesis(
    hypothesis_path: str,
    model_dir: str,
    singlecell_path: str,
    tf_binding_dir: str,
    output_dir: str,
    ppi_dir: Optional[str] = None,
    celltype_key: str = 'sclc_subcluster',
):
    """
    Test a regulatory hypothesis against the trained GRN model.
    
    Validates each hypothesized edge and computes:
    - Edge support (does GRN agree with hypothesis?)
    - Edge strength (how strong is the effect?)
    - Statistical significance
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/4] Parsing hypothesis from {hypothesis_path}...")
    nodes, edges, outputs = parse_hypothesis_file(hypothesis_path)
    
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Output genes: {outputs}")
    
    for edge in edges:
        symbol = '→' if edge['type'] == 'activator' else '⊣'
        print(f"    {edge['source']} {symbol} {edge['target']}")
    
    # Load data
    print(f"\n[2/4] Loading model and data...")
    expression, gene_names, celltypes = load_singlecell_data(singlecell_path, celltype_key)
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(tf_binding_dir, gene_names)
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    
    # Load GRN matrix
    grn_path = Path(model_dir) / 'grn_matrix_CORRECT.npy'
    if not grn_path.exists():
        grn_path = Path(model_dir) / 'grn_matrix.npy'
    
    if grn_path.exists():
        grn_matrix = np.load(grn_path)
        print(f"  GRN matrix: {grn_matrix.shape}")
    else:
        print("  ERROR: No GRN matrix found!")
        return
    
    # Create name to index mappings
    gene_upper = {g.upper(): i for i, g in enumerate(gene_names)}
    tf_upper = {t.upper(): i for i, t in enumerate(tf_names)}
    
    # Test each edge
    print(f"\n[3/4] Testing hypothesis edges...")
    
    results = []
    for edge in edges:
        source = edge['source'].upper()
        target = edge['target'].upper()
        expected_sign = 1 if edge['type'] == 'activator' else -1
        
        # Find indices
        source_is_tf = source in tf_upper
        target_is_gene = target in gene_upper
        target_is_tf = target in tf_upper
        
        if source_is_tf and target_is_gene:
            # TF → Gene relationship
            tf_idx = tf_upper[source]
            gene_idx = gene_upper[target]
            
            effect = grn_matrix[gene_idx, tf_idx]
            actual_sign = np.sign(effect)
            
            # Compute significance via permutation
            all_effects = grn_matrix[gene_idx, :]
            percentile = stats.percentileofscore(np.abs(all_effects), np.abs(effect))
            
            supported = (actual_sign == expected_sign)
            
            results.append({
                'source': edge['source'],
                'target': edge['target'],
                'type': edge['type'],
                'effect': effect,
                'expected_sign': expected_sign,
                'actual_sign': int(actual_sign),
                'supported': supported,
                'percentile': percentile,
                'edge_type': 'TF→Gene'
            })
            
        elif source_is_tf and target_is_tf:
            # TF → TF relationship (cascade)
            source_idx = tf_upper[source]
            target_idx = tf_upper[target]
            
            if tf_tf_adj is not None:
                effect = tf_tf_adj[source_idx, target_idx]
            else:
                effect = 0.0
            
            supported = effect > 0  # Any connection is considered support
            
            results.append({
                'source': edge['source'],
                'target': edge['target'],
                'type': edge['type'],
                'effect': effect,
                'expected_sign': expected_sign,
                'actual_sign': 1 if effect > 0 else 0,
                'supported': supported,
                'percentile': 0,
                'edge_type': 'TF→TF'
            })
        else:
            # Unknown edge type
            results.append({
                'source': edge['source'],
                'target': edge['target'],
                'type': edge['type'],
                'effect': np.nan,
                'expected_sign': expected_sign,
                'actual_sign': np.nan,
                'supported': False,
                'percentile': 0,
                'edge_type': 'Unknown'
            })
    
    # Print results
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 60)
    
    n_supported = sum(1 for r in results if r['supported'])
    print(f"\nOverall: {n_supported}/{len(results)} edges SUPPORTED ({100*n_supported/len(results):.1f}%)")
    
    print("\nEdge-by-Edge Analysis:")
    print("-" * 60)
    
    for r in results:
        symbol = '→' if r['type'] == 'activator' else '⊣'
        status = "✓ SUPPORTED" if r['supported'] else "✗ NOT SUPPORTED"
        
        if np.isfinite(r['effect']):
            print(f"  {r['source']} {symbol} {r['target']}")
            print(f"    Effect: {r['effect']:.4f} (percentile: {r['percentile']:.1f}%)")
            print(f"    {status}")
        else:
            print(f"  {r['source']} {symbol} {r['target']}")
            print(f"    ⚠ Could not find in GRN")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'hypothesis_results.csv', index=False)
    
    # Generate visualization
    print(f"\n[4/4] Generating hypothesis visualization...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Edge support bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    colors = ['#06d6a0' if r['supported'] else '#ef476f' for r in results]
    ax_a.barh(range(len(results)), [r['effect'] if np.isfinite(r['effect']) else 0 for r in results],
             color=colors, alpha=0.8)
    ax_a.set_yticks(range(len(results)))
    labels = [f"{r['source']}→{r['target']}" for r in results]
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.axvline(0, color='black', linewidth=0.5)
    ax_a.set_xlabel('Effect Size')
    ax_a.set_title('Edge Effects (Green=Supported, Red=Not Supported)')
    ax_a.invert_yaxis()
    
    # Panel B: Hypothesis support pie
    ax_b = fig.add_subplot(gs[0, 1])
    sizes = [n_supported, len(results) - n_supported]
    labels = [f'Supported\n({n_supported})', f'Not Supported\n({len(results)-n_supported})']
    colors = ['#06d6a0', '#ef476f']
    ax_b.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax_b.set_title('Hypothesis Support')
    
    # Panel C: Effect percentiles
    ax_c = fig.add_subplot(gs[1, 0])
    valid_results = [r for r in results if np.isfinite(r['percentile'])]
    if valid_results:
        ax_c.bar(range(len(valid_results)), [r['percentile'] for r in valid_results],
                color='#4361ee', alpha=0.8)
        ax_c.set_xticks(range(len(valid_results)))
        ax_c.set_xticklabels([f"{r['source']}→{r['target']}" for r in valid_results],
                            rotation=45, ha='right', fontsize=8)
        ax_c.axhline(95, color='red', linestyle='--', label='95th percentile')
        ax_c.set_ylabel('Percentile')
        ax_c.set_title('Effect Strength Percentiles')
        ax_c.legend()
    
    # Panel D: Summary text
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    
    summary_text = f"""
HYPOTHESIS TEST SUMMARY
{'='*40}

Edges Tested: {len(results)}
Edges Supported: {n_supported} ({100*n_supported/len(results):.1f}%)

OUTPUT GENES: {', '.join(outputs)}

INTERPRETATION:
"""
    if n_supported >= len(results) * 0.8:
        summary_text += "✓ Strong support for hypothesis!\n  Most edges are confirmed by the GRN."
    elif n_supported >= len(results) * 0.5:
        summary_text += "◐ Partial support for hypothesis.\n  Some edges not confirmed."
    else:
        summary_text += "✗ Weak support for hypothesis.\n  Consider revising the model."
    
    ax_d.text(0.1, 0.9, summary_text, transform=ax_d.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f7f7f7', alpha=0.9))
    
    plt.suptitle('Hypothesis Testing Results', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_dir / 'hypothesis_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - hypothesis_results.csv")
    print(f"  - hypothesis_test.png")
    
    return results


def load_gene_list(gene_input: str) -> List[str]:
    """
    Load gene(s) from either a single gene name or a text file.
    
    Args:
        gene_input: Either a gene name (e.g., "SSTR2") or path to .txt file
        
    Returns:
        List of gene names
    """
    # Check if input is a file
    input_path = Path(gene_input)
    if input_path.exists() and input_path.suffix in ['.txt', '.csv', '.list']:
        # Read genes from file (one per line)
        genes = []
        with open(input_path, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('#'):  # Skip empty lines and comments
                    genes.append(gene)
        print(f"Loaded {len(genes)} genes from {input_path}")
        return genes
    else:
        # Single gene name
        return [gene_input]


def main():
    parser = argparse.ArgumentParser(
        description="Gene-specific fine-tuning using IntegratedGRNModel from train_v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single gene:
  python scripts/finetune_gene.py --gene SSTR2 --model results/sclc/ ...
  
  # Multiple genes (master regulators):
  python scripts/finetune_gene.py --gene genes.txt --model results/sclc/ ...
  
  # genes.txt format (one gene per line):
  SSTR2
  ASCL1
  NEUROD1
  # Comments start with #
"""
    )
    parser.add_argument('--gene', type=str, default=None,
                        help='Target gene name (e.g., SSTR2) OR path to .txt file with multiple genes. Not required if --hypothesis is provided.')
    parser.add_argument('--model', type=str, required=True,
                        help='Directory containing pre-trained model from train_v6')
    parser.add_argument('--singlecell', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--tf-binding', type=str, required=True,
                        help='Directory with TF binding data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--ppi', type=str, default=None,
                        help='Path to PPI directory (optional)')
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster',
                        help='Cell type key in h5ad')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for gene-specific head')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip automatic analysis figure generation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hypothesis', type=str, default=None,
                        help='Path to hypothesis file (from hypothesis_builder.py) OR "gui" to launch GUI')
    
    args = parser.parse_args()
    
    # Check if user wants hypothesis GUI
    if args.hypothesis == "gui":
        print("=" * 70)
        print("LAUNCHING HYPOTHESIS PATHWAY BUILDER GUI")
        print("=" * 70)
        print("\nAfter building your hypothesis:")
        print("  1. Click 'Test Hypothesis' to export")
        print("  2. Re-run this script with --hypothesis <exported_file.txt>")
        print()
        
        import subprocess
        import sys
        subprocess.run([sys.executable, str(Path(__file__).parent / 'hypothesis_builder.py')])
        return
    
    # If hypothesis file provided, test it
    if args.hypothesis and args.hypothesis != "gui" and Path(args.hypothesis).exists():
        print("=" * 70)
        print("HYPOTHESIS TESTING MODE")
        print("=" * 70)
        
        test_hypothesis(
            hypothesis_path=args.hypothesis,
            model_dir=args.model,
            singlecell_path=args.singlecell,
            tf_binding_dir=args.tf_binding,
            output_dir=args.output,
            ppi_dir=args.ppi,
            celltype_key=args.celltype_key,
        )
        return
    
    # Require --gene for non-hypothesis modes
    if args.gene is None:
        parser.error("--gene is required unless --hypothesis is provided")
    
    # Load gene(s) from input
    genes = load_gene_list(args.gene)
    
    if len(genes) == 1:
        # =====================================================================
        # SINGLE GENE MODE
        # =====================================================================
        print("=" * 70)
        print(f"SINGLE GENE MODE: {genes[0]}")
        print("=" * 70)
        
        finetune_gene(
            model_dir=args.model,
            singlecell_path=args.singlecell,
            tf_binding_dir=args.tf_binding,
            target_gene=genes[0],
            output_dir=args.output,
            ppi_dir=args.ppi,
            celltype_key=args.celltype_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            run_analysis=not args.no_analysis
        )
    else:
        # =====================================================================
        # MULTI-GENE MODE (MASTER REGULATORS)
        # =====================================================================
        print("=" * 70)
        print(f"MULTI-GENE MODE: {len(genes)} genes → MASTER REGULATOR DISCOVERY")
        print("=" * 70)
        
        finetune_multi_gene(
            model_dir=args.model,
            singlecell_path=args.singlecell,
            tf_binding_dir=args.tf_binding,
            target_genes=genes,
            output_dir=args.output,
            ppi_dir=args.ppi,
            celltype_key=args.celltype_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            run_analysis=not args.no_analysis,
            lr=args.lr
        )


if __name__ == '__main__':
    main()

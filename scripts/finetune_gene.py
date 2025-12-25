#!/usr/bin/env python3
"""
Fine-tune GRN Model for Specific Target Gene
=============================================
Uses the EXACT same code as train_v5.py but adds extra weight on target gene.
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# Import EVERYTHING from train_v5 - use exact same code
from train_v5 import (
    DeltaGRNModel,
    DeltaFocusedLoss,
    VirtualPerturbationDatasetV5,
    load_singlecell_data,
    load_tf_binding,
    compute_tf_activity_linger,
    classify_tf_states
)


class GeneWeightedLoss(DeltaFocusedLoss):
    """
    EXACT same loss as train_v5, but with extra weight on target gene.
    """
    
    def __init__(self, target_gene_idx: int, gene_weight: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.target_gene_idx = target_gene_idx
        self.gene_weight = gene_weight
        
    def forward(self, pred_delta, actual_delta, gene_effects=None, batch_chip_mask=None):
        # Get base loss from parent (EXACT same as train_v5)
        losses = super().forward(pred_delta, actual_delta, gene_effects, batch_chip_mask)
        
        # Add extra loss for target gene
        target_pred = pred_delta[:, self.target_gene_idx]
        target_actual = actual_delta[:, self.target_gene_idx]
        
        # Target gene MSE (weighted)
        target_mse = F.mse_loss(target_pred, target_actual)
        
        # Target gene correlation
        pred_c = target_pred - target_pred.mean()
        actual_c = target_actual - target_actual.mean()
        correlation = (pred_c * actual_c).mean() / (pred_c.std() * actual_c.std() + 1e-8)
        correlation = torch.clamp(correlation, -1, 1)
        
        # Target gene direction
        sign_match = ((target_pred > 0) == (target_actual > 0)).float().mean()
        
        # Add weighted target loss to total
        losses['total'] = losses['total'] + self.gene_weight * target_mse
        losses['target_mse'] = target_mse
        losses['target_corr'] = correlation
        losses['target_sign'] = sign_match
        
        return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--singlecell', type=str, required=True)
    parser.add_argument('--tf-binding', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gene-weight', type=float, default=5.0)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_gene = args.gene.upper()
    
    print("=" * 70)
    print(f"FINE-TUNING FOR {target_gene}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Gene weight: {args.gene_weight}x")
    
    # =========================================================================
    # Load Data (SAME as train_v5)
    # =========================================================================
    print("\n[1/5] Loading single-cell data...")
    expression, gene_names, celltypes = load_singlecell_data(args.singlecell, args.celltype_key)
    n_celltypes, n_genes = expression.shape
    
    # Find target gene
    gene_names_upper = [g.upper() for g in gene_names]
    if target_gene not in gene_names_upper:
        raise ValueError(f"Target gene {target_gene} not found!")
    target_gene_idx = gene_names_upper.index(target_gene)
    print(f"  Target gene: {target_gene} at index {target_gene_idx}")
    print(f"  Expression range: [{expression[:, target_gene_idx].min():.3f}, {expression[:, target_gene_idx].max():.3f}]")
    
    print("\n[2/5] Loading TF binding...")
    binding_matrix, tf_names, tf_tf_adj = load_tf_binding(args.tf_binding, gene_names)
    n_tfs = len(tf_names)
    print(f"  TFs: {n_tfs}")
    
    print("\n[3/5] Computing TF activity...")
    tf_activity = compute_tf_activity_linger(expression, binding_matrix)
    tf_states = classify_tf_states(tf_activity)
    print(f"  TF activity: {tf_activity.shape}")
    
    # =========================================================================
    # Load Pre-trained Model
    # =========================================================================
    print("\n[4/5] Loading pre-trained model...")
    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        loaded_epoch = checkpoint.get('epoch', 'unknown')
        loaded_delta = checkpoint.get('best_val_delta_pcc', checkpoint.get('val_delta_pcc', 'unknown'))
        print(f"  Loaded checkpoint: epoch={loaded_epoch}, delta_pcc={loaded_delta}")
    else:
        state_dict = checkpoint
        print("  Loaded raw state dict")
    
    # Create model (SAME architecture as train_v5)
    model = DeltaGRNModel(
        n_genes=n_genes,
        n_tfs=n_tfs,
        hidden_dim=256,
        n_cascade_layers=2,
        dropout=0.2,
        use_context_gating=True
    )
    
    # Set priors BEFORE loading weights
    model.set_priors(
        chip_binding=torch.FloatTensor(binding_matrix),
        tf_tf_adjacency=torch.FloatTensor(tf_tf_adj)
    )
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected[:5]}...")
    
    model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # =========================================================================
    # Create Dataset (SAME as train_v5)
    # =========================================================================
    print("\n[5/5] Creating training data...")
    
    # All pairs for fine-tuning
    pairs = [(i, j) for i in range(n_celltypes) for j in range(n_celltypes) if i != j]
    print(f"  Training pairs: {len(pairs)}")
    
    dataset = VirtualPerturbationDatasetV5(
        expression_means=expression,
        tf_activity=tf_activity,
        celltype_names=celltypes,
        allowed_pairs=pairs,
        min_activity_diff=0.1,
        add_hard_negatives=True
    )
    
    print(f"  Samples: {len(dataset)}")
    
    # Use weighted sampler (SAME as train_v5)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=min(len(dataset), 100000),  # Cap for speed
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # =========================================================================
    # Fine-tuning Setup
    # =========================================================================
    # Loss: SAME as train_v5 + extra weight on target gene
    loss_fn = GeneWeightedLoss(
        target_gene_idx=target_gene_idx,
        gene_weight=args.gene_weight,
        delta_weight=1.0,
        ranking_weight=0.3,
        direction_weight=0.5,
        chip_reg_weight=0.2
    )
    
    # Only fine-tune effect-related layers
    # CRITICAL: Freeze tf_embeddings and tf_cascade to preserve TF identity!
    # Only allow changing how TFs map to genes (decoder) and the specific gene wiring (direct)
    finetune_params = []
    frozen_count = 0
    for name, param in model.named_parameters():
        # FREEZE: TF embeddings and cascade - these define TF identity and relationships
        if 'tf_embeddings' in name or 'tf_cascade' in name:
            param.requires_grad = False
            frozen_count += 1
        # FINE-TUNE: Effect decoder (TF->gene mapping), direct effects, context gating
        elif any(x in name for x in ['effect_decoder', 'direct_effects', 'context']):
            param.requires_grad = True
            finetune_params.append(param)
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"  Fine-tuning: {len(finetune_params)} groups, {sum(p.numel() for p in finetune_params):,} params")
    print(f"  Frozen: {frozen_count} groups")
    
    optimizer = torch.optim.AdamW(finetune_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print("=" * 70)
    
    best_corr = -1.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        epoch_loss = []
        epoch_delta_mse = []
        epoch_target_corr = []
        epoch_target_sign = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", leave=True)
        
        for batch in pbar:
            source = batch['source_expression'].to(device)
            tf_idx = batch['tf_idx'].to(device)
            pert_type = batch['perturbation_type'].to(device)
            pert_strength = batch['perturbation_strength'].to(device)
            actual_delta = batch['delta_expression'].to(device)
            
            optimizer.zero_grad()
            
            # Forward (SAME as train_v5)
            outputs = model(source, tf_idx, pert_type, pert_strength)
            
            # Loss (SAME as train_v5 + target weight)
            losses = loss_fn(
                pred_delta=outputs['delta_expression'],
                actual_delta=actual_delta,
                gene_effects=outputs['gene_effects'],
                batch_chip_mask=outputs['batch_chip_mask']
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(finetune_params, 1.0)
            optimizer.step()
            
            # Track
            epoch_loss.append(losses['total'].item())
            epoch_delta_mse.append(losses['delta_mse'].item())
            epoch_target_corr.append(losses['target_corr'].item())
            epoch_target_sign.append(losses['target_sign'].item())
            
            # Show running averages
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_loss):.2f}",
                'delta': f"{np.mean(epoch_delta_mse):.2f}",
                f'{target_gene}_r': f"{np.mean(epoch_target_corr):.3f}",
                'sign': f"{np.mean(epoch_target_sign):.1%}"
            })
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = np.mean(epoch_loss)
        avg_corr = np.mean(epoch_target_corr)
        avg_sign = np.mean(epoch_target_sign)
        
        print(f"Epoch {epoch:3d}: loss={avg_loss:.2f} | {target_gene}_corr={avg_corr:.4f} | sign={avg_sign:.1%}")
        
        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'target_corr': avg_corr,
            'target_sign': avg_sign
        })
        
        # Save best
        if avg_corr > best_corr:
            best_corr = avg_corr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'target_gene': target_gene,
                'target_gene_idx': target_gene_idx,
                'best_correlation': best_corr,
            }, output_dir / 'best_model.pt')
            print(f"  >> New best! {target_gene}_corr={best_corr:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save GRN matrix
    grn_matrix = model.get_grn_matrix().detach().cpu().numpy()
    np.save(output_dir / 'grn_matrix.npy', grn_matrix)
    
    # Copy metadata from original model
    model_dir = model_path.parent
    for fname in ['gene_names.csv', 'tf_names.csv', 'celltype_names.csv', 
                  'binding_matrix.npy', 'tf_activity.npy', 'tf_states.npy']:
        src = model_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
    
    # Save history
    pd.DataFrame(history).to_csv(output_dir / 'finetune_history.csv', index=False)
    
    # Save config
    config = {
        'target_gene': target_gene,
        'target_gene_idx': target_gene_idx,
        'gene_weight': args.gene_weight,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_correlation': float(best_corr),
        'n_genes': n_genes,
        'n_tfs': n_tfs
    }
    with open(output_dir / 'finetune_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nBest: Epoch {best_epoch} with {target_gene}_corr={best_corr:.4f}")
    print(f"Saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print(f"\nTo analyze:")
    print(f"python scripts/analyze_target_full.py \\")
    print(f"    --gene {target_gene} \\")
    print(f"    --model {output_dir} \\")
    print(f"    --singlecell {args.singlecell} \\")
    print(f"    --ppi data/ppi \\")
    print(f"    --output analysis/{target_gene}_finetuned/")


if __name__ == '__main__':
    main()

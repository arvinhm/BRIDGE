#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: NATURE-LEVEL EVALUATION METRICS
================================================================================

Comprehensive evaluation for publication-quality reporting:

1. Per-gene ΔPCC distribution (median, IQR, fraction >0.3/0.5/0.7)
2. AUPRC/AUROC for up vs down regulation
3. Ablations with 95% bootstrap CI
4. Held-out TF test
5. Delta baselines (ridge, ChIP-only, no-change, copy-source)

Usage:
    python scripts/evaluate_nature_metrics.py \
        --model results/sclc_v7/best_model.pt \
        --singlecell data/sclc/preprocessed.h5ad \
        --tf-binding data/tf_binding/ \
        --omnipath data/omnipath/ \
        --ppi data/ppi/ \
        --output analysis/nature_metrics/

Author: BRIDGE Team
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.linear_model import Ridge
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_v7 import (
    load_singlecell_data,
    load_tf_binding,
    compute_tf_activity_linger,
    load_omnipath,
    IntegratedGRNModelV7,
    MultiTFDataset,
)
from torch.utils.data import DataLoader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    values = values[~np.isnan(values)]
    if len(values) < 10:
        return np.nan, np.nan, np.nan
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    mean = np.mean(values)
    
    return mean, lower, upper


def compute_per_gene_pcc(pred_delta: np.ndarray, actual_delta: np.ndarray) -> np.ndarray:
    """Compute PCC for each gene across samples."""
    n_genes = pred_delta.shape[1]
    pccs = np.zeros(n_genes)
    
    for g in range(n_genes):
        pred_g = pred_delta[:, g]
        actual_g = actual_delta[:, g]
        
        if np.std(pred_g) > 1e-8 and np.std(actual_g) > 1e-8:
            pccs[g] = np.corrcoef(pred_g, actual_g)[0, 1]
        else:
            pccs[g] = 0.0
    
    return pccs


def compute_direction_accuracy_per_gene(pred_delta: np.ndarray, actual_delta: np.ndarray, 
                                         threshold: float = 0.1) -> np.ndarray:
    """Compute direction accuracy for each gene (only for non-trivial deltas)."""
    n_genes = pred_delta.shape[1]
    dir_acc = np.zeros(n_genes)
    
    for g in range(n_genes):
        pred_g = pred_delta[:, g]
        actual_g = actual_delta[:, g]
        
        # Only consider non-trivial deltas
        mask = np.abs(actual_g) > threshold
        if mask.sum() > 10:
            correct = ((pred_g[mask] > 0) == (actual_g[mask] > 0)).astype(float)
            dir_acc[g] = correct.mean()
        else:
            dir_acc[g] = np.nan
    
    return dir_acc


# =============================================================================
# BASELINE MODELS
# =============================================================================

class NoChangeBaseline:
    """Baseline: predict no change (delta = 0)."""
    def predict(self, X):
        return np.zeros_like(X)


class CopySourceBaseline:
    """Baseline: predict delta = source - mean(source)."""
    def __init__(self, mean_expression):
        self.mean_expr = mean_expression
    
    def predict(self, source_expr):
        return source_expr - self.mean_expr


class RidgeDeltaBaseline:
    """Ridge regression on TF activity changes."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = None
    
    def fit(self, tf_changes: np.ndarray, deltas: np.ndarray):
        """Fit one ridge model per gene."""
        n_genes = deltas.shape[1]
        self.models = []
        
        for g in tqdm(range(n_genes), desc="Fitting Ridge baseline"):
            model = Ridge(alpha=self.alpha)
            model.fit(tf_changes, deltas[:, g])
            self.models.append(model)
    
    def predict(self, tf_changes: np.ndarray) -> np.ndarray:
        if self.models is None:
            raise ValueError("Must fit first")
        
        preds = np.zeros((tf_changes.shape[0], len(self.models)))
        for g, model in enumerate(self.models):
            preds[:, g] = model.predict(tf_changes)
        return preds


class ChIPOnlyBaseline:
    """Linear model using only ChIP binding matrix."""
    def __init__(self, binding_matrix: np.ndarray):
        self.binding = binding_matrix  # (n_genes, n_tfs)
    
    def predict(self, tf_changes: np.ndarray) -> np.ndarray:
        """delta = binding @ tf_changes.T"""
        # tf_changes: (n_samples, n_tfs)
        # binding: (n_genes, n_tfs)
        # output: (n_samples, n_genes)
        return tf_changes @ self.binding.T


# =============================================================================
# MAIN EVALUATION CLASS
# =============================================================================

class NatureMetricsEvaluator:
    """Comprehensive evaluation for Nature-level metrics."""
    
    def __init__(
        self,
        model_path: str,
        singlecell_path: str,
        tf_binding_path: str,
        omnipath_path: Optional[str] = None,
        ppi_path: Optional[str] = None,
        celltype_key: str = 'sclc_subcluster',
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print("NATURE-LEVEL METRICS EVALUATION")
        print("=" * 70)
        
        # Load data
        print("\n[1/5] Loading data...")
        self.expression, self.gene_names, self.celltypes = load_singlecell_data(
            singlecell_path, celltype_key
        )
        self.n_celltypes, self.n_genes = self.expression.shape
        
        print("\n[2/5] Loading TF binding...")
        self.binding_matrix, self.tf_names, self.tf_tf_adj = load_tf_binding(
            tf_binding_path, self.gene_names
        )
        self.n_tfs = len(self.tf_names)
        
        print("\n[3/5] Computing TF activity...")
        self.tf_activity = compute_tf_activity_linger(self.expression, self.binding_matrix)
        
        # Load OmniPath
        print("\n[4/5] Loading OmniPath...")
        self.has_omnipath = False
        if omnipath_path:
            self.gene_gene, self.gene_tf, self.tf_tf_omnipath = load_omnipath(
                omnipath_path, len(self.gene_names), len(self.tf_names),
                gene_names=self.gene_names, tf_names=self.tf_names
            )
            if self.tf_tf_omnipath is not None:
                self.tf_tf_combined = self.tf_tf_adj + 0.3 * self.tf_tf_omnipath
            else:
                self.tf_tf_combined = self.tf_tf_adj
            self.has_omnipath = True
        else:
            self.gene_gene = np.zeros((self.n_genes, self.n_genes), dtype=np.float32)
            self.gene_tf = np.zeros((self.n_genes, self.n_tfs), dtype=np.float32)
            self.tf_tf_combined = self.tf_tf_adj
        
        # Load model
        print("\n[5/5] Loading model...")
        self.model = self._load_model(model_path, ppi_path)
        
        # Create datasets
        self._create_datasets()
        
        # Storage for results
        self.results = {}
    
    def _load_model(self, model_path: str, ppi_path: Optional[str]) -> IntegratedGRNModelV7:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
        
        model = IntegratedGRNModelV7(
            n_genes=self.n_genes,
            n_tfs=self.n_tfs,
            hidden_dim=config.get('hidden_dim', 512),
            use_gene_propagation=self.has_omnipath,
            use_gene_tf_modulation=self.has_omnipath
        )
        
        # Load PPI
        cof_to_tf = None
        cof_indices = None
        if ppi_path:
            ppi_dir = Path(ppi_path)
            if (ppi_dir / 'cof_to_tf_adj.npy').exists():
                cof_to_tf = np.load(ppi_dir / 'cof_to_tf_adj.npy')
                cof_indices = np.load(ppi_dir / 'cofactor_gene_indices.npy')
        
        model.set_priors(
            chip_binding=torch.from_numpy(self.binding_matrix),
            tf_tf_adjacency=torch.from_numpy(self.tf_tf_combined),
            cofactor_tf_adjacency=torch.from_numpy(cof_to_tf) if cof_to_tf is not None else None,
            cofactor_gene_indices=torch.from_numpy(cof_indices) if cof_indices is not None else None
        )
        
        if self.has_omnipath:
            model.set_omnipath_priors(
                gene_gene=torch.from_numpy(self.gene_gene),
                gene_tf=torch.from_numpy(self.gene_tf)
            )
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def _create_datasets(self):
        """Create train/val splits for evaluation."""
        n_holdout = max(2, self.n_celltypes // 6)
        np.random.seed(42)
        holdout = set(np.random.choice(self.n_celltypes, n_holdout, replace=False))
        train_idx = [i for i in range(self.n_celltypes) if i not in holdout]
        
        self.train_pairs = [(i, j) for i in train_idx for j in train_idx if i != j]
        self.val_pairs = [(i, j) for i in holdout for j in holdout if i != j]
        
        print(f"  Train pairs: {len(self.train_pairs)}, Val pairs: {len(self.val_pairs)}")
    
    def _collect_predictions(self, pairs: List[Tuple[int, int]], max_samples: int = 5000
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect model predictions for given pairs."""
        dataset = MultiTFDataset(
            expression_means=self.expression,
            tf_activity=self.tf_activity,
            celltype_names=list(self.celltypes),
            allowed_pairs=pairs,
            augment_factor=1,
            noise_std=0.0
        )
        
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        all_pred = []
        all_actual = []
        all_tf_change = []
        all_source = []
        
        n_collected = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions"):
                if n_collected >= max_samples:
                    break
                
                tf_change = batch['tf_activity_change'].to(self.device)
                source_expr = batch['source_expression'].to(self.device)
                actual_delta = batch['delta_expression']
                
                pred_delta, _ = self.model.predict_delta_multi_tf(tf_change, source_expr)
                
                all_pred.append(pred_delta.cpu().numpy())
                all_actual.append(actual_delta.numpy())
                all_tf_change.append(batch['tf_activity_change'].numpy())
                all_source.append(batch['source_expression'].numpy())
                
                n_collected += len(pred_delta)
        
        return (
            np.concatenate(all_pred, axis=0)[:max_samples],
            np.concatenate(all_actual, axis=0)[:max_samples],
            np.concatenate(all_tf_change, axis=0)[:max_samples],
            np.concatenate(all_source, axis=0)[:max_samples]
        )
    
    # =========================================================================
    # 1. PER-GENE ΔPCC DISTRIBUTION
    # =========================================================================
    
    def evaluate_per_gene_pcc(self) -> Dict:
        """Compute per-gene ΔPCC distribution."""
        print("\n" + "=" * 70)
        print("1. PER-GENE ΔPCC DISTRIBUTION")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        pccs = compute_per_gene_pcc(pred, actual)
        
        # Filter valid
        valid_pccs = pccs[~np.isnan(pccs)]
        
        results = {
            'median': float(np.median(valid_pccs)),
            'mean': float(np.mean(valid_pccs)),
            'std': float(np.std(valid_pccs)),
            'q25': float(np.percentile(valid_pccs, 25)),
            'q75': float(np.percentile(valid_pccs, 75)),
            'frac_above_0.3': float((valid_pccs > 0.3).mean()),
            'frac_above_0.5': float((valid_pccs > 0.5).mean()),
            'frac_above_0.7': float((valid_pccs > 0.7).mean()),
            'n_genes': len(valid_pccs)
        }
        
        print(f"  Median ΔPCC: {results['median']:.4f}")
        print(f"  IQR: [{results['q25']:.4f}, {results['q75']:.4f}]")
        print(f"  Fraction >0.3: {results['frac_above_0.3']:.1%}")
        print(f"  Fraction >0.5: {results['frac_above_0.5']:.1%}")
        print(f"  Fraction >0.7: {results['frac_above_0.7']:.1%}")
        
        self.results['per_gene_pcc'] = results
        self.gene_pccs = pccs
        
        return results
    
    # =========================================================================
    # 2. AUPRC/AUROC FOR UP VS DOWN
    # =========================================================================
    
    def evaluate_classification_metrics(self, threshold: float = 0.1) -> Dict:
        """Compute AUPRC/AUROC for up vs down regulation."""
        print("\n" + "=" * 70)
        print("2. AUPRC/AUROC FOR UP VS DOWN REGULATION")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        # Binary labels: up (1) vs down (0)
        # Only consider non-trivial changes
        mask = np.abs(actual) > threshold
        
        y_true = (actual[mask] > 0).astype(int).flatten()
        y_score = pred[mask].flatten()
        
        # Compute metrics
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        
        # Direction accuracy
        y_pred = (y_score > 0).astype(int)
        dir_acc = (y_true == y_pred).mean()
        
        results = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'direction_accuracy': float(dir_acc),
            'n_samples': int(y_true.sum()),
            'threshold': threshold
        }
        
        print(f"  AUROC: {results['auroc']:.4f}")
        print(f"  AUPRC: {results['auprc']:.4f}")
        print(f"  Direction Accuracy: {results['direction_accuracy']:.1%}")
        
        self.results['classification'] = results
        
        return results
    
    # =========================================================================
    # 3. ABLATIONS WITH 95% CI
    # =========================================================================
    
    def evaluate_ablations(self, n_bootstrap: int = 100) -> Dict:
        """Run ablations with bootstrap 95% CI."""
        print("\n" + "=" * 70)
        print("3. ABLATIONS WITH 95% BOOTSTRAP CI")
        print("=" * 70)
        
        pred_full, actual, tf_change, source = self._collect_predictions(self.val_pairs)
        
        # Full model baseline
        full_pccs = compute_per_gene_pcc(pred_full, actual)
        full_mean, full_lo, full_hi = bootstrap_ci(full_pccs, n_bootstrap)
        
        results = {
            'full_model': {
                'mean_pcc': full_mean,
                'ci_lower': full_lo,
                'ci_upper': full_hi
            }
        }
        
        print(f"  Full model: {full_mean:.4f} [{full_lo:.4f}, {full_hi:.4f}]")
        
        # Ablation 1: No Gene→Gene propagation
        if hasattr(self.model, 'use_gene_propagation') and self.model.use_gene_propagation:
            saved = self.model.use_gene_propagation
            self.model.use_gene_propagation = False
            
            with torch.no_grad():
                pred_no_gg = []
                for i in range(0, len(tf_change), 256):
                    batch_tf = torch.FloatTensor(tf_change[i:i+256]).to(self.device)
                    batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                    p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                    pred_no_gg.append(p.cpu().numpy())
                pred_no_gg = np.concatenate(pred_no_gg, axis=0)
            
            self.model.use_gene_propagation = saved
            
            no_gg_pccs = compute_per_gene_pcc(pred_no_gg, actual)
            no_gg_mean, no_gg_lo, no_gg_hi = bootstrap_ci(no_gg_pccs, n_bootstrap)
            
            delta = full_mean - no_gg_mean
            results['no_gene_gene'] = {
                'mean_pcc': no_gg_mean,
                'ci_lower': no_gg_lo,
                'ci_upper': no_gg_hi,
                'delta_vs_full': delta
            }
            print(f"  -Gene→Gene: {no_gg_mean:.4f} [{no_gg_lo:.4f}, {no_gg_hi:.4f}] (Δ={delta:+.4f})")
        
        # Ablation 2: No Gene→TF modulation
        if hasattr(self.model, 'use_gene_tf_modulation') and self.model.use_gene_tf_modulation:
            saved = self.model.use_gene_tf_modulation
            self.model.use_gene_tf_modulation = False
            
            with torch.no_grad():
                pred_no_gtf = []
                for i in range(0, len(tf_change), 256):
                    batch_tf = torch.FloatTensor(tf_change[i:i+256]).to(self.device)
                    batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                    p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                    pred_no_gtf.append(p.cpu().numpy())
                pred_no_gtf = np.concatenate(pred_no_gtf, axis=0)
            
            self.model.use_gene_tf_modulation = saved
            
            no_gtf_pccs = compute_per_gene_pcc(pred_no_gtf, actual)
            no_gtf_mean, no_gtf_lo, no_gtf_hi = bootstrap_ci(no_gtf_pccs, n_bootstrap)
            
            delta = full_mean - no_gtf_mean
            results['no_gene_tf'] = {
                'mean_pcc': no_gtf_mean,
                'ci_lower': no_gtf_lo,
                'ci_upper': no_gtf_hi,
                'delta_vs_full': delta
            }
            print(f"  -Gene→TF: {no_gtf_mean:.4f} [{no_gtf_lo:.4f}, {no_gtf_hi:.4f}] (Δ={delta:+.4f})")
        
        # Ablation 3: No Cofactor gating
        if hasattr(self.model, 'cof_to_tf_adj') and self.model.cof_to_tf_adj is not None:
            saved = self.model.cof_to_tf_adj
            self.model.cof_to_tf_adj = None
            
            with torch.no_grad():
                pred_no_cof = []
                for i in range(0, len(tf_change), 256):
                    batch_tf = torch.FloatTensor(tf_change[i:i+256]).to(self.device)
                    batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                    p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                    pred_no_cof.append(p.cpu().numpy())
                pred_no_cof = np.concatenate(pred_no_cof, axis=0)
            
            self.model.cof_to_tf_adj = saved
            
            no_cof_pccs = compute_per_gene_pcc(pred_no_cof, actual)
            no_cof_mean, no_cof_lo, no_cof_hi = bootstrap_ci(no_cof_pccs, n_bootstrap)
            
            delta = full_mean - no_cof_mean
            results['no_cofactor'] = {
                'mean_pcc': no_cof_mean,
                'ci_lower': no_cof_lo,
                'ci_upper': no_cof_hi,
                'delta_vs_full': delta
            }
            print(f"  -Cofactor: {no_cof_mean:.4f} [{no_cof_lo:.4f}, {no_cof_hi:.4f}] (Δ={delta:+.4f})")
        
        self.results['ablations'] = results
        
        return results
    
    # =========================================================================
    # 4. HELD-OUT TF TEST
    # =========================================================================
    
    def evaluate_held_out_tfs(self, n_holdout_tfs: int = 50, n_bootstrap: int = 100) -> Dict:
        """Evaluate on held-out TFs (zero out during prediction)."""
        print("\n" + "=" * 70)
        print("4. HELD-OUT TF TEST")
        print("=" * 70)
        
        pred_full, actual, tf_change, source = self._collect_predictions(self.val_pairs)
        
        # Select random TFs to hold out
        np.random.seed(42)
        holdout_tfs = np.random.choice(self.n_tfs, n_holdout_tfs, replace=False)
        
        # Zero out held-out TFs in tf_change
        tf_change_masked = tf_change.copy()
        tf_change_masked[:, holdout_tfs] = 0
        
        with torch.no_grad():
            pred_masked = []
            for i in range(0, len(tf_change_masked), 256):
                batch_tf = torch.FloatTensor(tf_change_masked[i:i+256]).to(self.device)
                batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                pred_masked.append(p.cpu().numpy())
            pred_masked = np.concatenate(pred_masked, axis=0)
        
        # Compute metrics
        full_pccs = compute_per_gene_pcc(pred_full, actual)
        masked_pccs = compute_per_gene_pcc(pred_masked, actual)
        
        full_mean, full_lo, full_hi = bootstrap_ci(full_pccs, n_bootstrap)
        masked_mean, masked_lo, masked_hi = bootstrap_ci(masked_pccs, n_bootstrap)
        
        results = {
            'n_held_out_tfs': n_holdout_tfs,
            'full_model': {
                'mean_pcc': full_mean,
                'ci_lower': full_lo,
                'ci_upper': full_hi
            },
            'held_out_tfs': {
                'mean_pcc': masked_mean,
                'ci_lower': masked_lo,
                'ci_upper': masked_hi
            },
            'drop': full_mean - masked_mean
        }
        
        print(f"  Full model: {full_mean:.4f} [{full_lo:.4f}, {full_hi:.4f}]")
        print(f"  {n_holdout_tfs} TFs held out: {masked_mean:.4f} [{masked_lo:.4f}, {masked_hi:.4f}]")
        print(f"  Performance drop: {results['drop']:.4f}")
        
        self.results['held_out_tfs'] = results
        
        return results
    
    # =========================================================================
    # 5. DELTA BASELINES
    # =========================================================================
    
    def evaluate_baselines(self, n_bootstrap: int = 100) -> Dict:
        """Compare against delta baselines."""
        print("\n" + "=" * 70)
        print("5. DELTA BASELINES COMPARISON")
        print("=" * 70)
        
        pred_model, actual, tf_change, source = self._collect_predictions(self.val_pairs)
        
        results = {}
        
        # Our model
        model_pccs = compute_per_gene_pcc(pred_model, actual)
        model_mean, model_lo, model_hi = bootstrap_ci(model_pccs, n_bootstrap)
        results['bridge_model'] = {
            'mean_pcc': model_mean,
            'ci_lower': model_lo,
            'ci_upper': model_hi
        }
        print(f"  BRIDGE model: {model_mean:.4f} [{model_lo:.4f}, {model_hi:.4f}]")
        
        # Baseline 1: No change (delta = 0)
        pred_nochange = np.zeros_like(actual)
        nochange_pccs = compute_per_gene_pcc(pred_nochange, actual)
        nochange_mean, nochange_lo, nochange_hi = bootstrap_ci(nochange_pccs, n_bootstrap)
        results['no_change'] = {
            'mean_pcc': nochange_mean,
            'ci_lower': nochange_lo,
            'ci_upper': nochange_hi
        }
        print(f"  No-change: {nochange_mean:.4f} [{nochange_lo:.4f}, {nochange_hi:.4f}]")
        
        # Baseline 2: ChIP-only linear
        chip_baseline = ChIPOnlyBaseline(self.binding_matrix)
        pred_chip = chip_baseline.predict(tf_change)
        chip_pccs = compute_per_gene_pcc(pred_chip, actual)
        chip_mean, chip_lo, chip_hi = bootstrap_ci(chip_pccs, n_bootstrap)
        results['chip_only'] = {
            'mean_pcc': chip_mean,
            'ci_lower': chip_lo,
            'ci_upper': chip_hi
        }
        print(f"  ChIP-only: {chip_mean:.4f} [{chip_lo:.4f}, {chip_hi:.4f}]")
        
        # Baseline 3: Ridge regression
        print("  Fitting Ridge baseline...")
        # Use train data for fitting
        train_pred, train_actual, train_tf, train_src = self._collect_predictions(
            self.train_pairs, max_samples=2000
        )
        
        ridge = RidgeDeltaBaseline(alpha=1.0)
        ridge.fit(train_tf, train_actual)
        pred_ridge = ridge.predict(tf_change)
        
        ridge_pccs = compute_per_gene_pcc(pred_ridge, actual)
        ridge_mean, ridge_lo, ridge_hi = bootstrap_ci(ridge_pccs, n_bootstrap)
        results['ridge'] = {
            'mean_pcc': ridge_mean,
            'ci_lower': ridge_lo,
            'ci_upper': ridge_hi
        }
        print(f"  Ridge: {ridge_mean:.4f} [{ridge_lo:.4f}, {ridge_hi:.4f}]")
        
        # Improvement over best baseline
        best_baseline = max(nochange_mean, chip_mean, ridge_mean)
        improvement = model_mean - best_baseline
        results['improvement_vs_best_baseline'] = improvement
        print(f"\n  Improvement vs best baseline: +{improvement:.4f}")
        
        self.results['baselines'] = results
        
        return results
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def create_figures(self, output_dir: Path):
        """Create all figures."""
        print("\n" + "=" * 70)
        print("CREATING FIGURES")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Figure A: Per-gene PCC distribution
        ax = axes[0, 0]
        if hasattr(self, 'gene_pccs'):
            valid = self.gene_pccs[~np.isnan(self.gene_pccs)]
            ax.hist(valid, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.median(valid), color='red', linestyle='--', label=f'Median={np.median(valid):.3f}')
            ax.axvline(0.3, color='orange', linestyle=':', label='Threshold 0.3')
            ax.axvline(0.5, color='green', linestyle=':', label='Threshold 0.5')
            ax.set_xlabel('Per-Gene ΔPCC')
            ax.set_ylabel('Count')
            ax.set_title('A. Per-Gene ΔPCC Distribution')
            ax.legend()
        
        # Figure B: Ablations
        ax = axes[0, 1]
        if 'ablations' in self.results:
            abl = self.results['ablations']
            names = []
            means = []
            errors = []
            
            for key, val in abl.items():
                if isinstance(val, dict) and 'mean_pcc' in val:
                    names.append(key.replace('_', '\n'))
                    means.append(val['mean_pcc'])
                    errors.append([val['mean_pcc'] - val['ci_lower'], 
                                   val['ci_upper'] - val['mean_pcc']])
            
            x = np.arange(len(names))
            ax.bar(x, means, yerr=np.array(errors).T, capsize=5, color='steelblue', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel('Mean Per-Gene ΔPCC')
            ax.set_title('B. Ablation Study (95% CI)')
        
        # Figure C: Baseline comparison
        ax = axes[1, 0]
        if 'baselines' in self.results:
            base = self.results['baselines']
            names = []
            means = []
            errors = []
            
            for key, val in base.items():
                if isinstance(val, dict) and 'mean_pcc' in val:
                    names.append(key.replace('_', '\n'))
                    means.append(val['mean_pcc'])
                    errors.append([val['mean_pcc'] - val['ci_lower'], 
                                   val['ci_upper'] - val['mean_pcc']])
            
            x = np.arange(len(names))
            colors = ['#2ca02c' if 'bridge' in n.lower() else '#1f77b4' for n in names]
            ax.bar(x, means, yerr=np.array(errors).T, capsize=5, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel('Mean Per-Gene ΔPCC')
            ax.set_title('C. Baseline Comparison (95% CI)')
        
        # Figure D: Held-out TF test
        ax = axes[1, 1]
        if 'held_out_tfs' in self.results:
            hot = self.results['held_out_tfs']
            names = ['Full Model', f'{hot["n_held_out_tfs"]} TFs\nHeld Out']
            means = [hot['full_model']['mean_pcc'], hot['held_out_tfs']['mean_pcc']]
            errors = [
                [hot['full_model']['mean_pcc'] - hot['full_model']['ci_lower'],
                 hot['full_model']['ci_upper'] - hot['full_model']['mean_pcc']],
                [hot['held_out_tfs']['mean_pcc'] - hot['held_out_tfs']['ci_lower'],
                 hot['held_out_tfs']['ci_upper'] - hot['held_out_tfs']['mean_pcc']]
            ]
            
            x = np.arange(len(names))
            ax.bar(x, means, yerr=np.array(errors).T, capsize=5, color=['#2ca02c', '#d62728'], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel('Mean Per-Gene ΔPCC')
            ax.set_title('D. Held-Out TF Generalization')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'nature_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: nature_metrics_summary.png")
    
    # =========================================================================
    # RUN ALL
    # =========================================================================
    
    def run_all(self, output_dir: Path):
        """Run all evaluations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluations
        self.evaluate_per_gene_pcc()
        self.evaluate_classification_metrics()
        self.evaluate_ablations()
        self.evaluate_held_out_tfs()
        self.evaluate_baselines()
        
        # Create figures
        self.create_figures(output_dir)
        
        # Save results
        with open(output_dir / 'nature_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save per-gene PCCs
        if hasattr(self, 'gene_pccs'):
            df = pd.DataFrame({
                'gene': self.gene_names,
                'delta_pcc': self.gene_pccs
            }).sort_values('delta_pcc', ascending=False)
            df.to_csv(output_dir / 'per_gene_pcc.csv', index=False)
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)
        print(f"Results saved to: {output_dir}")
        
        return self.results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BRIDGE 2.0: Nature-Level Metrics Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--singlecell', type=str, required=True, help='Path to h5ad file')
    parser.add_argument('--tf-binding', type=str, required=True, help='TF binding directory')
    parser.add_argument('--omnipath', type=str, default=None, help='OmniPath directory')
    parser.add_argument('--ppi', type=str, default=None, help='PPI directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--celltype-key', type=str, default='sclc_subcluster')
    
    args = parser.parse_args()
    
    evaluator = NatureMetricsEvaluator(
        model_path=args.model,
        singlecell_path=args.singlecell,
        tf_binding_path=args.tf_binding,
        omnipath_path=args.omnipath,
        ppi_path=args.ppi,
        celltype_key=args.celltype_key
    )
    
    results = evaluator.run_all(Path(args.output))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PUBLICATION")
    print("=" * 70)
    
    if 'per_gene_pcc' in results:
        pgp = results['per_gene_pcc']
        print(f"\nPer-Gene ΔPCC: {pgp['median']:.3f} (median), IQR [{pgp['q25']:.3f}, {pgp['q75']:.3f}]")
        print(f"  Genes with ΔPCC>0.5: {pgp['frac_above_0.5']:.1%}")
    
    if 'classification' in results:
        cls = results['classification']
        print(f"\nClassification: AUROC={cls['auroc']:.3f}, AUPRC={cls['auprc']:.3f}")
    
    if 'baselines' in results:
        base = results['baselines']
        print(f"\nImprovement vs best baseline: +{base['improvement_vs_best_baseline']:.3f}")


if __name__ == '__main__':
    main()


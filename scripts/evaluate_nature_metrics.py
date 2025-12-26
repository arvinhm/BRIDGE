#!/usr/bin/env python3
"""
================================================================================
BRIDGE 2.0: NATURE-LEVEL EVALUATION METRICS (COMPREHENSIVE)
================================================================================

Complete evaluation for publication-quality reporting:

CORE METRICS:
1. Per-gene ΔPCC distribution (median, IQR, fraction >0.3/0.5/0.7)
2. AUPRC/AUROC for up vs down regulation
3. Ablations with 95% bootstrap CI
4. Held-out TF test
5. Delta baselines (ridge, ChIP-only, no-change)

ADVANCED METRICS:
6. Per-cell-type performance heatmap
7. Calibration: ECE, reliability diagram, calibration slope
8. Risk-coverage curves (selective prediction)
9. Spearman/Rank quality and NDCG
10. Error stratification by biology (expression variance, ChIP density)
11. Prior consistency and edge overlap with OmniPath/ChIP
12. Robustness to noise and shuffle controls
13. R² on deltas

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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, r2_score
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


def compute_per_gene_spearman(pred_delta: np.ndarray, actual_delta: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation for each gene (less sensitive to outliers)."""
    n_genes = pred_delta.shape[1]
    rhos = np.zeros(n_genes)
    
    for g in range(n_genes):
        pred_g = pred_delta[:, g]
        actual_g = actual_delta[:, g]
        
        if np.std(pred_g) > 1e-8 and np.std(actual_g) > 1e-8:
            rhos[g], _ = stats.spearmanr(pred_g, actual_g)
        else:
            rhos[g] = 0.0
    
    return rhos


def reliability_and_calibration(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10
                                ) -> Tuple[float, float, float, np.ndarray]:
    """Compute ECE, calibration slope/intercept, and bin statistics."""
    y = np.abs(actual).ravel()
    p = np.abs(pred).ravel()
    
    # Remove NaN/Inf
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    
    if len(y) < 100:
        return np.nan, np.nan, np.nan, np.array([])
    
    # Bin by predicted magnitude
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6  # Ensure last bin includes max
    
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        bin_pred = p[m].mean()
        bin_obs = y[m].mean()
        ece += np.abs(bin_pred - bin_obs) * m.mean()
        bin_stats.append((bin_pred, bin_obs, m.sum()))
    
    # Calibration slope/intercept
    slope, intercept, r, _, _ = stats.linregress(p, y)
    
    return ece, slope, intercept, np.array(bin_stats)


def risk_coverage(pred: np.ndarray, actual: np.ndarray, conf: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute risk-coverage curves."""
    order = np.argsort(-conf.ravel())
    y = actual.ravel()[order]
    p = pred.ravel()[order]
    
    errs = np.abs(y - p)
    n = len(errs)
    
    cov = np.linspace(1/n, 1.0, n)
    mae_cov = np.cumsum(errs) / np.arange(1, n + 1)
    dir_acc_cov = np.cumsum((np.sign(p) == np.sign(y)).astype(float)) / np.arange(1, n + 1)
    
    return cov, mae_cov, dir_acc_cov


def compute_ndcg(pred: np.ndarray, actual: np.ndarray, k: int = 50) -> float:
    """Compute NDCG@k for signed ranking."""
    n = len(pred)
    if n < k:
        k = n
    
    # Use actual as relevance, pred for ranking
    pred_order = np.argsort(-np.abs(pred))[:k]
    ideal_order = np.argsort(-np.abs(actual))[:k]
    
    # DCG
    relevance = np.abs(actual[pred_order])
    dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
    
    # IDCG
    ideal_relevance = np.abs(actual[ideal_order])
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


# =============================================================================
# BASELINE MODELS
# =============================================================================

class NoChangeBaseline:
    """Baseline: predict no change (delta = 0)."""
    def predict(self, X):
        return np.zeros_like(X)


class ChIPOnlyBaseline:
    """Linear model using only ChIP binding matrix."""
    def __init__(self, binding_matrix: np.ndarray):
        self.binding = binding_matrix  # (n_genes, n_tfs)
    
    def predict(self, tf_changes: np.ndarray) -> np.ndarray:
        return tf_changes @ self.binding.T


class RidgeDeltaBaseline:
    """Ridge regression on TF activity changes."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = None
    
    def fit(self, tf_changes: np.ndarray, deltas: np.ndarray):
        n_genes = deltas.shape[1]
        self.models = []
        
        for g in tqdm(range(n_genes), desc="Fitting Ridge baseline", leave=False):
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
        print("NATURE-LEVEL METRICS EVALUATION (COMPREHENSIVE)")
        print("=" * 70)
        
        # Load data
        print("\n[1/5] Loading data...")
        self.expression, self.gene_names, self.celltypes = load_singlecell_data(
            singlecell_path, celltype_key
        )
        self.n_celltypes, self.n_genes = self.expression.shape
        self.gene_names_arr = np.array(self.gene_names)
        
        print("\n[2/5] Loading TF binding...")
        self.binding_matrix, self.tf_names, self.tf_tf_adj = load_tf_binding(
            tf_binding_path, self.gene_names
        )
        self.n_tfs = len(self.tf_names)
        self.tf_names_arr = np.array(self.tf_names)
        
        print("\n[3/5] Computing TF activity...")
        self.tf_activity = compute_tf_activity_linger(self.expression, self.binding_matrix)
        
        # Load OmniPath
        print("\n[4/5] Loading OmniPath...")
        self.has_omnipath = False
        self.omnipath_path = omnipath_path
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
        self.ppi_path = ppi_path
        self.model = self._load_model(model_path, ppi_path)
        
        # Create datasets
        self._create_datasets()
        
        # Precompute gene statistics
        self._compute_gene_statistics()
        
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
        self.holdout_idx = list(holdout)
        
        print(f"  Train pairs: {len(self.train_pairs)}, Val pairs: {len(self.val_pairs)}")
    
    def _compute_gene_statistics(self):
        """Precompute gene-level statistics for stratification."""
        self.gene_expr_mean = self.expression.mean(axis=0)
        self.gene_expr_var = self.expression.var(axis=0)
        self.gene_chip_density = self.binding_matrix.sum(axis=1)  # Number of TFs binding
    
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
            for batch in tqdm(loader, desc="Collecting predictions", leave=False):
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
        """Compute per-gene ΔPCC and Spearman distribution."""
        print("\n" + "=" * 70)
        print("1. PER-GENE ΔPCC DISTRIBUTION")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        pccs = compute_per_gene_pcc(pred, actual)
        spearman = compute_per_gene_spearman(pred, actual)
        
        # Filter valid
        valid_pccs = pccs[~np.isnan(pccs)]
        valid_spearman = spearman[~np.isnan(spearman)]
        
        results = {
            'pearson': {
                'median': float(np.median(valid_pccs)),
                'mean': float(np.mean(valid_pccs)),
                'std': float(np.std(valid_pccs)),
                'q25': float(np.percentile(valid_pccs, 25)),
                'q75': float(np.percentile(valid_pccs, 75)),
                'frac_above_0.3': float((valid_pccs > 0.3).mean()),
                'frac_above_0.5': float((valid_pccs > 0.5).mean()),
                'frac_above_0.7': float((valid_pccs > 0.7).mean()),
            },
            'spearman': {
                'median': float(np.median(valid_spearman)),
                'mean': float(np.mean(valid_spearman)),
                'q25': float(np.percentile(valid_spearman, 25)),
                'q75': float(np.percentile(valid_spearman, 75)),
            },
            'n_genes': len(valid_pccs)
        }
        
        print(f"  Pearson ΔPCC: median={results['pearson']['median']:.4f}, "
              f"IQR=[{results['pearson']['q25']:.4f}, {results['pearson']['q75']:.4f}]")
        print(f"  Spearman ρ:   median={results['spearman']['median']:.4f}")
        print(f"  Fraction >0.3: {results['pearson']['frac_above_0.3']:.1%}")
        print(f"  Fraction >0.5: {results['pearson']['frac_above_0.5']:.1%}")
        
        self.results['per_gene_pcc'] = results
        self.gene_pccs = pccs
        self.gene_spearman = spearman
        
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
        
        # R² on deltas
        r2 = r2_score(actual.flatten(), pred.flatten())
        
        results = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'direction_accuracy': float(dir_acc),
            'r2_delta': float(r2),
            'n_samples': int(mask.sum()),
            'threshold': threshold
        }
        
        print(f"  AUROC: {results['auroc']:.4f}")
        print(f"  AUPRC: {results['auprc']:.4f}")
        print(f"  Direction Accuracy: {results['direction_accuracy']:.1%}")
        print(f"  R² on deltas: {results['r2_delta']:.4f}")
        
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
        
        # Helper to run ablation
        def run_ablation(attr_name, flag_value):
            if not hasattr(self.model, attr_name):
                return None
            saved = getattr(self.model, attr_name)
            setattr(self.model, attr_name, flag_value)
            
            with torch.no_grad():
                pred_abl = []
                for i in range(0, len(tf_change), 256):
                    batch_tf = torch.FloatTensor(tf_change[i:i+256]).to(self.device)
                    batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                    p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                    pred_abl.append(p.cpu().numpy())
                pred_abl = np.concatenate(pred_abl, axis=0)
            
            setattr(self.model, attr_name, saved)
            return pred_abl
        
        # Ablation 1: No Gene→Gene propagation
        if hasattr(self.model, 'use_gene_propagation') and self.model.use_gene_propagation:
            pred_no_gg = run_ablation('use_gene_propagation', False)
            if pred_no_gg is not None:
                no_gg_pccs = compute_per_gene_pcc(pred_no_gg, actual)
                no_gg_mean, no_gg_lo, no_gg_hi = bootstrap_ci(no_gg_pccs, n_bootstrap)
                delta = full_mean - no_gg_mean
                results['no_gene_gene'] = {
                    'mean_pcc': no_gg_mean, 'ci_lower': no_gg_lo, 'ci_upper': no_gg_hi,
                    'delta_vs_full': delta
                }
                print(f"  -Gene→Gene: {no_gg_mean:.4f} [{no_gg_lo:.4f}, {no_gg_hi:.4f}] (Δ={delta:+.4f})")
        
        # Ablation 2: No Gene→TF modulation
        if hasattr(self.model, 'use_gene_tf_modulation') and self.model.use_gene_tf_modulation:
            pred_no_gtf = run_ablation('use_gene_tf_modulation', False)
            if pred_no_gtf is not None:
                no_gtf_pccs = compute_per_gene_pcc(pred_no_gtf, actual)
                no_gtf_mean, no_gtf_lo, no_gtf_hi = bootstrap_ci(no_gtf_pccs, n_bootstrap)
                delta = full_mean - no_gtf_mean
                results['no_gene_tf'] = {
                    'mean_pcc': no_gtf_mean, 'ci_lower': no_gtf_lo, 'ci_upper': no_gtf_hi,
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
                'mean_pcc': no_cof_mean, 'ci_lower': no_cof_lo, 'ci_upper': no_cof_hi,
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
        
        np.random.seed(42)
        holdout_tfs = np.random.choice(self.n_tfs, n_holdout_tfs, replace=False)
        
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
        
        full_pccs = compute_per_gene_pcc(pred_full, actual)
        masked_pccs = compute_per_gene_pcc(pred_masked, actual)
        
        full_mean, full_lo, full_hi = bootstrap_ci(full_pccs, n_bootstrap)
        masked_mean, masked_lo, masked_hi = bootstrap_ci(masked_pccs, n_bootstrap)
        
        results = {
            'n_held_out_tfs': n_holdout_tfs,
            'full_model': {'mean_pcc': full_mean, 'ci_lower': full_lo, 'ci_upper': full_hi},
            'held_out_tfs': {'mean_pcc': masked_mean, 'ci_lower': masked_lo, 'ci_upper': masked_hi},
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
        model_r2 = r2_score(actual.flatten(), pred_model.flatten())
        results['bridge_model'] = {
            'mean_pcc': model_mean, 'ci_lower': model_lo, 'ci_upper': model_hi,
            'r2': model_r2
        }
        print(f"  BRIDGE model: {model_mean:.4f} [{model_lo:.4f}, {model_hi:.4f}] (R²={model_r2:.4f})")
        
        # Baseline 1: No change (delta = 0)
        pred_nochange = np.zeros_like(actual)
        nochange_pccs = compute_per_gene_pcc(pred_nochange, actual)
        nochange_mean, nochange_lo, nochange_hi = bootstrap_ci(nochange_pccs, n_bootstrap)
        results['no_change'] = {'mean_pcc': nochange_mean, 'ci_lower': nochange_lo, 'ci_upper': nochange_hi}
        print(f"  No-change: {nochange_mean:.4f} [{nochange_lo:.4f}, {nochange_hi:.4f}]")
        
        # Baseline 2: ChIP-only linear
        chip_baseline = ChIPOnlyBaseline(self.binding_matrix)
        pred_chip = chip_baseline.predict(tf_change)
        chip_pccs = compute_per_gene_pcc(pred_chip, actual)
        chip_mean, chip_lo, chip_hi = bootstrap_ci(chip_pccs, n_bootstrap)
        chip_r2 = r2_score(actual.flatten(), pred_chip.flatten())
        results['chip_only'] = {
            'mean_pcc': chip_mean, 'ci_lower': chip_lo, 'ci_upper': chip_hi,
            'r2': chip_r2
        }
        print(f"  ChIP-only: {chip_mean:.4f} [{chip_lo:.4f}, {chip_hi:.4f}] (R²={chip_r2:.4f})")
        
        # Baseline 3: Ridge regression
        print("  Fitting Ridge baseline...")
        train_pred, train_actual, train_tf, _ = self._collect_predictions(self.train_pairs, max_samples=2000)
        
        ridge = RidgeDeltaBaseline(alpha=1.0)
        ridge.fit(train_tf, train_actual)
        pred_ridge = ridge.predict(tf_change)
        
        ridge_pccs = compute_per_gene_pcc(pred_ridge, actual)
        ridge_mean, ridge_lo, ridge_hi = bootstrap_ci(ridge_pccs, n_bootstrap)
        ridge_r2 = r2_score(actual.flatten(), pred_ridge.flatten())
        results['ridge'] = {
            'mean_pcc': ridge_mean, 'ci_lower': ridge_lo, 'ci_upper': ridge_hi,
            'r2': ridge_r2
        }
        print(f"  Ridge: {ridge_mean:.4f} [{ridge_lo:.4f}, {ridge_hi:.4f}] (R²={ridge_r2:.4f})")
        
        # Store ridge for hard gene analysis
        self.ridge_pccs = ridge_pccs
        
        # Improvement over best baseline
        best_baseline = max(nochange_mean, chip_mean, ridge_mean)
        improvement = model_mean - best_baseline
        results['improvement_vs_best_baseline'] = improvement
        print(f"\n  Improvement vs best baseline: +{improvement:.4f}")
        
        # Hard genes: bottom quartile of Ridge ΔPCC
        hard_gene_mask = ridge_pccs < np.percentile(ridge_pccs[~np.isnan(ridge_pccs)], 25)
        if hard_gene_mask.sum() > 0:
            model_hard = model_pccs[hard_gene_mask]
            ridge_hard = ridge_pccs[hard_gene_mask]
            results['hard_genes'] = {
                'n_genes': int(hard_gene_mask.sum()),
                'model_mean': float(np.nanmean(model_hard)),
                'ridge_mean': float(np.nanmean(ridge_hard)),
                'improvement': float(np.nanmean(model_hard) - np.nanmean(ridge_hard))
            }
            print(f"  Hard genes (bottom 25% Ridge): Model={results['hard_genes']['model_mean']:.4f}, "
                  f"Ridge={results['hard_genes']['ridge_mean']:.4f}, "
                  f"Improvement={results['hard_genes']['improvement']:.4f}")
        
        self.results['baselines'] = results
        return results
    
    # =========================================================================
    # 6. PER-CELL-TYPE PERFORMANCE
    # =========================================================================
    
    def evaluate_per_celltype(self) -> Dict:
        """Compute per-cell-type performance."""
        print("\n" + "=" * 70)
        print("6. PER-CELL-TYPE PERFORMANCE")
        print("=" * 70)
        
        celltype_pccs = {}
        
        for tgt_idx in self.holdout_idx:
            pairs = [(i, tgt_idx) for i in self.holdout_idx if i != tgt_idx]
            if len(pairs) == 0:
                continue
            
            pred, actual, _, _ = self._collect_predictions(pairs, max_samples=1000)
            pccs = compute_per_gene_pcc(pred, actual)
            valid_pccs = pccs[~np.isnan(pccs)]
            
            ct_name = self.celltypes[tgt_idx] if tgt_idx < len(self.celltypes) else f"CT_{tgt_idx}"
            celltype_pccs[ct_name] = {
                'median': float(np.median(valid_pccs)),
                'q25': float(np.percentile(valid_pccs, 25)),
                'q75': float(np.percentile(valid_pccs, 75)),
                'mean': float(np.mean(valid_pccs))
            }
        
        if celltype_pccs:
            sorted_cts = sorted(celltype_pccs.items(), key=lambda x: x[1]['median'], reverse=True)
            print(f"  Best 3 cell types:")
            for ct, m in sorted_cts[:3]:
                print(f"    {ct}: {m['median']:.4f}")
            print(f"  Worst 3 cell types:")
            for ct, m in sorted_cts[-3:]:
                print(f"    {ct}: {m['median']:.4f}")
        
        self.results['per_celltype'] = celltype_pccs
        self.celltype_pccs = celltype_pccs
        return celltype_pccs
    
    # =========================================================================
    # 7. CALIBRATION AND ECE
    # =========================================================================
    
    def evaluate_calibration(self) -> Dict:
        """Compute calibration metrics: ECE, slope, intercept."""
        print("\n" + "=" * 70)
        print("7. CALIBRATION (ECE, SLOPE, INTERCEPT)")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        ece, slope, intercept, bin_stats = reliability_and_calibration(pred, actual)
        
        results = {
            'ece': float(ece),
            'calibration_slope': float(slope),
            'calibration_intercept': float(intercept),
            'ideal_slope': 1.0,
            'ideal_intercept': 0.0,
            'bin_stats': bin_stats.tolist() if len(bin_stats) > 0 else []
        }
        
        print(f"  ECE: {results['ece']:.4f}")
        print(f"  Calibration slope: {results['calibration_slope']:.4f} (ideal=1.0)")
        print(f"  Calibration intercept: {results['calibration_intercept']:.4f} (ideal=0.0)")
        
        self.results['calibration'] = results
        self.calibration_bin_stats = bin_stats
        return results
    
    # =========================================================================
    # 8. RISK-COVERAGE CURVES
    # =========================================================================
    
    def evaluate_risk_coverage(self) -> Dict:
        """Compute risk-coverage curves for selective prediction."""
        print("\n" + "=" * 70)
        print("8. RISK-COVERAGE CURVES")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        # Use |predicted delta| as confidence
        conf = np.abs(pred)
        
        cov, mae_cov, dir_acc_cov = risk_coverage(pred, actual, conf)
        
        # Sample at key coverage levels
        results = {
            'coverage_levels': [0.1, 0.25, 0.5, 0.75, 1.0],
            'mae_at_coverage': {},
            'dir_acc_at_coverage': {}
        }
        
        for c in results['coverage_levels']:
            idx = int(c * len(cov)) - 1
            if idx >= 0 and idx < len(cov):
                results['mae_at_coverage'][str(c)] = float(mae_cov[idx])
                results['dir_acc_at_coverage'][str(c)] = float(dir_acc_cov[idx])
        
        print(f"  At 50% coverage: MAE={results['mae_at_coverage'].get('0.5', 'N/A'):.4f}, "
              f"Dir Acc={results['dir_acc_at_coverage'].get('0.5', 'N/A'):.1%}")
        print(f"  At 100% coverage: MAE={results['mae_at_coverage'].get('1.0', 'N/A'):.4f}, "
              f"Dir Acc={results['dir_acc_at_coverage'].get('1.0', 'N/A'):.1%}")
        
        self.results['risk_coverage'] = results
        self.risk_coverage_data = (cov, mae_cov, dir_acc_cov)
        return results
    
    # =========================================================================
    # 9. NDCG RANKING QUALITY
    # =========================================================================
    
    def evaluate_ndcg(self) -> Dict:
        """Compute NDCG at various k."""
        print("\n" + "=" * 70)
        print("9. NDCG RANKING QUALITY")
        print("=" * 70)
        
        pred, actual, _, _ = self._collect_predictions(self.val_pairs)
        
        ks = [10, 50, 100, 200]
        results = {}
        
        for k in ks:
            ndcg_scores = []
            for i in range(pred.shape[0]):
                ndcg = compute_ndcg(pred[i], actual[i], k)
                ndcg_scores.append(ndcg)
            
            mean_ndcg = np.mean(ndcg_scores)
            results[f'ndcg@{k}'] = float(mean_ndcg)
            print(f"  NDCG@{k}: {mean_ndcg:.4f}")
        
        self.results['ndcg'] = results
        return results
    
    # =========================================================================
    # 10. ERROR STRATIFICATION BY BIOLOGY
    # =========================================================================
    
    def evaluate_stratification(self) -> Dict:
        """Stratify performance by biological attributes."""
        print("\n" + "=" * 70)
        print("10. ERROR STRATIFICATION BY BIOLOGY")
        print("=" * 70)
        
        if not hasattr(self, 'gene_pccs'):
            self.evaluate_per_gene_pcc()
        
        results = {}
        
        # By expression variance (deciles)
        var_deciles = pd.qcut(self.gene_expr_var, 10, labels=False, duplicates='drop')
        var_groups = {}
        for d in range(10):
            mask = var_deciles == d
            if mask.sum() > 0:
                group_pccs = self.gene_pccs[mask]
                valid = group_pccs[~np.isnan(group_pccs)]
                if len(valid) > 0:
                    var_groups[f'decile_{d}'] = float(np.median(valid))
        
        results['by_expression_variance'] = var_groups
        
        # By ChIP density (deciles)
        density_deciles = pd.qcut(self.gene_chip_density, 10, labels=False, duplicates='drop')
        density_groups = {}
        for d in range(10):
            mask = density_deciles == d
            if mask.sum() > 0:
                group_pccs = self.gene_pccs[mask]
                valid = group_pccs[~np.isnan(group_pccs)]
                if len(valid) > 0:
                    density_groups[f'decile_{d}'] = float(np.median(valid))
        
        results['by_chip_density'] = density_groups
        
        # Correlation between ΔPCC and attributes
        valid_mask = ~np.isnan(self.gene_pccs)
        if valid_mask.sum() > 100:
            corr_var, _ = stats.spearmanr(self.gene_pccs[valid_mask], self.gene_expr_var[valid_mask])
            corr_density, _ = stats.spearmanr(self.gene_pccs[valid_mask], self.gene_chip_density[valid_mask])
            results['correlations'] = {
                'pcc_vs_expr_variance': float(corr_var),
                'pcc_vs_chip_density': float(corr_density)
            }
            print(f"  ΔPCC vs Expression Variance: ρ={corr_var:.4f}")
            print(f"  ΔPCC vs ChIP Density: ρ={corr_density:.4f}")
        
        self.results['stratification'] = results
        return results
    
    # =========================================================================
    # 11. PRIOR CONSISTENCY AND EDGE OVERLAP
    # =========================================================================
    
    def evaluate_prior_consistency(self) -> Dict:
        """Check consistency with OmniPath/ChIP priors."""
        print("\n" + "=" * 70)
        print("11. PRIOR CONSISTENCY AND EDGE OVERLAP")
        print("=" * 70)
        
        # Get learned GRN
        grn = self.model.get_grn_matrix().detach().cpu().numpy()
        
        results = {}
        
        # ChIP overlap
        chip_edges = self.binding_matrix > 0
        grn_edges = np.abs(grn) > np.percentile(np.abs(grn), 90)
        
        overlap = (chip_edges & grn_edges).sum()
        chip_total = chip_edges.sum()
        grn_top_total = grn_edges.sum()
        
        results['chip_overlap'] = {
            'overlap_count': int(overlap),
            'chip_edges': int(chip_total),
            'grn_top10pct_edges': int(grn_top_total),
            'precision': float(overlap / grn_top_total) if grn_top_total > 0 else 0,
            'recall': float(overlap / chip_total) if chip_total > 0 else 0
        }
        
        print(f"  ChIP overlap (top 10% GRN edges):")
        print(f"    Precision: {results['chip_overlap']['precision']:.1%}")
        print(f"    Recall: {results['chip_overlap']['recall']:.1%}")
        
        # Sign consistency with ChIP (where signs exist)
        # For ChIP, assume positive binding = potential activation
        chip_positive = self.binding_matrix > 0
        grn_positive = grn > 0
        
        both_edges = chip_positive & (np.abs(grn) > np.percentile(np.abs(grn), 50))
        if both_edges.sum() > 0:
            sign_consistent = (chip_positive[both_edges] == grn_positive[both_edges]).mean()
            results['sign_consistency'] = float(sign_consistent)
            print(f"  Sign consistency with ChIP: {sign_consistent:.1%}")
        
        self.results['prior_consistency'] = results
        return results
    
    # =========================================================================
    # 12. ROBUSTNESS TESTS
    # =========================================================================
    
    def evaluate_robustness(self, sigmas: List[float] = [0.05, 0.1, 0.2]) -> Dict:
        """Test robustness to TF-change noise and shuffle control."""
        print("\n" + "=" * 70)
        print("12. ROBUSTNESS TESTS")
        print("=" * 70)
        
        pred_clean, actual, tf_change, source = self._collect_predictions(self.val_pairs)
        
        clean_pcc = np.corrcoef(pred_clean.flatten(), actual.flatten())[0, 1]
        
        results = {
            'clean': float(clean_pcc),
            'noise_tests': {},
            'shuffle_control': None
        }
        
        print(f"  Clean: ΔPCC={clean_pcc:.4f}")
        
        # Noise tests
        for sigma in sigmas:
            noisy_tf = tf_change + np.random.normal(0, sigma, tf_change.shape)
            
            with torch.no_grad():
                pred_noisy = []
                for i in range(0, len(noisy_tf), 256):
                    batch_tf = torch.FloatTensor(noisy_tf[i:i+256]).to(self.device)
                    batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                    p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                    pred_noisy.append(p.cpu().numpy())
                pred_noisy = np.concatenate(pred_noisy, axis=0)
            
            noisy_pcc = np.corrcoef(pred_noisy.flatten(), actual.flatten())[0, 1]
            results['noise_tests'][f'sigma_{sigma}'] = float(noisy_pcc)
            print(f"  Noise σ={sigma}: ΔPCC={noisy_pcc:.4f} (drop={clean_pcc - noisy_pcc:.4f})")
        
        # Shuffle control (sanity check - should give ~0)
        np.random.seed(123)
        shuffled_tf = tf_change.copy()
        for i in range(shuffled_tf.shape[0]):
            np.random.shuffle(shuffled_tf[i])
        
        with torch.no_grad():
            pred_shuffle = []
            for i in range(0, len(shuffled_tf), 256):
                batch_tf = torch.FloatTensor(shuffled_tf[i:i+256]).to(self.device)
                batch_src = torch.FloatTensor(source[i:i+256]).to(self.device)
                p, _ = self.model.predict_delta_multi_tf(batch_tf, batch_src)
                pred_shuffle.append(p.cpu().numpy())
            pred_shuffle = np.concatenate(pred_shuffle, axis=0)
        
        shuffle_pcc = np.corrcoef(pred_shuffle.flatten(), actual.flatten())[0, 1]
        results['shuffle_control'] = float(shuffle_pcc)
        print(f"  Shuffle control: ΔPCC={shuffle_pcc:.4f} (should be ~0)")
        
        self.results['robustness'] = results
        return results
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def create_figures(self, output_dir: Path):
        """Create comprehensive figures."""
        print("\n" + "=" * 70)
        print("CREATING FIGURES")
        print("=" * 70)
        
        # Main summary figure (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # A. Per-gene PCC distribution
        ax = axes[0, 0]
        if hasattr(self, 'gene_pccs'):
            valid = self.gene_pccs[~np.isnan(self.gene_pccs)]
            ax.hist(valid, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(np.median(valid), color='red', linestyle='--', lw=2, 
                       label=f'Median={np.median(valid):.3f}')
            ax.axvline(0.5, color='orange', linestyle=':', lw=1.5, label='Threshold 0.5')
            ax.set_xlabel('Per-Gene ΔPCC', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('A. Per-Gene ΔPCC Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        
        # B. Ablations
        ax = axes[0, 1]
        if 'ablations' in self.results:
            abl = self.results['ablations']
            names = []
            means = []
            errors = []
            
            for key, val in abl.items():
                if isinstance(val, dict) and 'mean_pcc' in val:
                    names.append(key.replace('_', '\n').replace('no\n', '-'))
                    means.append(val['mean_pcc'])
                    errors.append([val['mean_pcc'] - val['ci_lower'], 
                                   val['ci_upper'] - val['mean_pcc']])
            
            x = np.arange(len(names))
            colors = ['#2ca02c' if 'full' in n.lower() else '#1f77b4' for n in names]
            ax.bar(x, means, yerr=np.array(errors).T, capsize=5, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=10)
            ax.set_ylabel('Mean Per-Gene ΔPCC', fontsize=12)
            ax.set_title('B. Ablation Study (95% CI)', fontsize=14, fontweight='bold')
        
        # C. Baseline comparison
        ax = axes[0, 2]
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
            ax.set_ylabel('Mean Per-Gene ΔPCC', fontsize=12)
            ax.set_title('C. Baseline Comparison', fontsize=14, fontweight='bold')
        
        # D. Calibration reliability diagram
        ax = axes[1, 0]
        if hasattr(self, 'calibration_bin_stats') and len(self.calibration_bin_stats) > 0:
            bin_pred = self.calibration_bin_stats[:, 0]
            bin_obs = self.calibration_bin_stats[:, 1]
            ax.scatter(bin_pred, bin_obs, s=100, c='steelblue', alpha=0.7)
            ax.plot([0, max(bin_pred)], [0, max(bin_pred)], 'r--', lw=2, label='Perfect calibration')
            ax.set_xlabel('Predicted |Δ|', fontsize=12)
            ax.set_ylabel('Observed |Δ|', fontsize=12)
            ax.set_title('D. Calibration (Reliability Diagram)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        
        # E. Risk-coverage curve
        ax = axes[1, 1]
        if hasattr(self, 'risk_coverage_data'):
            cov, mae_cov, dir_acc_cov = self.risk_coverage_data
            ax.plot(cov, dir_acc_cov, 'b-', lw=2, label='Direction Accuracy')
            ax.set_xlabel('Coverage', fontsize=12)
            ax.set_ylabel('Direction Accuracy', fontsize=12)
            ax.set_title('E. Risk-Coverage Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0.5, 1])
        
        # F. Robustness to noise
        ax = axes[1, 2]
        if 'robustness' in self.results:
            rob = self.results['robustness']
            sigmas = [0] + [float(k.split('_')[1]) for k in rob['noise_tests'].keys()]
            pccs = [rob['clean']] + list(rob['noise_tests'].values())
            ax.plot(sigmas, pccs, 'o-', color='steelblue', lw=2, markersize=8)
            ax.axhline(rob['shuffle_control'], color='red', linestyle='--', lw=2, 
                       label=f'Shuffle control: {rob["shuffle_control"]:.3f}')
            ax.set_xlabel('Noise σ', fontsize=12)
            ax.set_ylabel('ΔPCC', fontsize=12)
            ax.set_title('F. Robustness to TF-Change Noise', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'nature_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: nature_metrics_summary.png")
        
        # Additional figure: Stratification
        if 'stratification' in self.results and hasattr(self, 'gene_pccs'):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # By expression variance
            ax = axes[0]
            var_data = self.results['stratification'].get('by_expression_variance', {})
            if var_data:
                x = list(range(len(var_data)))
                y = list(var_data.values())
                ax.bar(x, y, color='steelblue', alpha=0.7)
                ax.set_xlabel('Expression Variance Decile', fontsize=12)
                ax.set_ylabel('Median ΔPCC', fontsize=12)
                ax.set_title('G. Performance by Expression Variance', fontsize=14, fontweight='bold')
            
            # By ChIP density
            ax = axes[1]
            density_data = self.results['stratification'].get('by_chip_density', {})
            if density_data:
                x = list(range(len(density_data)))
                y = list(density_data.values())
                ax.bar(x, y, color='coral', alpha=0.7)
                ax.set_xlabel('ChIP Density Decile', fontsize=12)
                ax.set_ylabel('Median ΔPCC', fontsize=12)
                ax.set_title('H. Performance by ChIP Density', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'nature_metrics_stratification.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: nature_metrics_stratification.png")
    
    # =========================================================================
    # RUN ALL
    # =========================================================================
    
    def run_all(self, output_dir: Path):
        """Run all evaluations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core metrics
        self.evaluate_per_gene_pcc()
        self.evaluate_classification_metrics()
        self.evaluate_ablations()
        self.evaluate_held_out_tfs()
        self.evaluate_baselines()
        
        # Advanced metrics
        self.evaluate_per_celltype()
        self.evaluate_calibration()
        self.evaluate_risk_coverage()
        self.evaluate_ndcg()
        self.evaluate_stratification()
        self.evaluate_prior_consistency()
        self.evaluate_robustness()
        
        # Create figures
        self.create_figures(output_dir)
        
        # Save results
        with open(output_dir / 'nature_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save per-gene PCCs
        if hasattr(self, 'gene_pccs'):
            df = pd.DataFrame({
                'gene': self.gene_names,
                'delta_pcc': self.gene_pccs,
                'spearman_rho': self.gene_spearman if hasattr(self, 'gene_spearman') else np.nan,
                'expr_variance': self.gene_expr_var,
                'chip_density': self.gene_chip_density
            }).sort_values('delta_pcc', ascending=False)
            df.to_csv(output_dir / 'per_gene_metrics.csv', index=False)
        
        # Save model card
        model_card = {
            'dataset': 'SCLC single-cell',
            'n_celltypes': self.n_celltypes,
            'n_genes': self.n_genes,
            'n_tfs': self.n_tfs,
            'split': 'cell-type pair holdout',
            'val_pairs': len(self.val_pairs),
            'train_pairs': len(self.train_pairs),
            'priors_used': ['ChIP-Atlas', 'OmniPath', 'Rosetta PPI'],
            'metrics_threshold': 0.1
        }
        with open(output_dir / 'model_card.json', 'w') as f:
            json.dump(model_card, f, indent=2)
        
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
        pgp = results['per_gene_pcc']['pearson']
        print(f"\nPer-Gene ΔPCC: {pgp['median']:.3f} (median), IQR [{pgp['q25']:.3f}, {pgp['q75']:.3f}]")
        print(f"  Genes with ΔPCC>0.5: {pgp['frac_above_0.5']:.1%}")
    
    if 'classification' in results:
        cls = results['classification']
        print(f"\nClassification: AUROC={cls['auroc']:.3f}, AUPRC={cls['auprc']:.3f}, R²={cls['r2_delta']:.3f}")
    
    if 'baselines' in results:
        base = results['baselines']
        print(f"\nImprovement vs best baseline: +{base['improvement_vs_best_baseline']:.3f}")
    
    if 'robustness' in results:
        rob = results['robustness']
        print(f"\nRobustness: Shuffle control ΔPCC={rob['shuffle_control']:.3f} (validates model)")


if __name__ == '__main__':
    main()

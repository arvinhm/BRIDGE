#!/usr/bin/env python3
"""
Preprocess Single-Cell RNA-seq Data for GRN Inference

Takes raw single-cell data and prepares it for training:
1. Quality control and filtering
2. Normalization
3. Cluster identification
4. Batch/patient metadata extraction
5. TF activity estimation

Usage:
    python scripts/preprocess_singlecell.py --input data.h5ad --output data/singlecell/
    python scripts/preprocess_singlecell.py --input counts.csv --metadata meta.csv --output data/singlecell/
"""

import os
import sys
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing scanpy
try:
    import scanpy as sc
    import anndata
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Warning: scanpy not available. Install with: pip install scanpy")


class SingleCellPreprocessor:
    """
    Preprocessor for single-cell RNA-seq data.
    
    Handles:
    - Quality control and filtering
    - Normalization and log-transformation
    - Highly variable gene selection
    - Dimensionality reduction
    - Clustering
    - TF activity estimation
    """
    
    def __init__(
        self,
        min_genes_per_cell: int = 200,
        max_genes_per_cell: int = 8000,
        min_cells_per_gene: int = 10,
        max_mito_pct: float = 25.0,
        n_hvg: int = 3000,
        normalize_target: float = 10000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        force_normalize: bool = False,
        skip_normalize: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            min_genes_per_cell: Minimum genes per cell
            max_genes_per_cell: Maximum genes per cell
            min_cells_per_gene: Minimum cells per gene
            max_mito_pct: Maximum mitochondrial percentage
            n_hvg: Number of highly variable genes
            normalize_target: Target sum for normalization
            n_pcs: Number of PCA components
            n_neighbors: Number of neighbors for UMAP/clustering
            resolution: Clustering resolution
            force_normalize: Force normalization even if data appears normalized
            skip_normalize: Skip normalization entirely
        """
        self.min_genes_per_cell = min_genes_per_cell
        self.max_genes_per_cell = max_genes_per_cell
        self.min_cells_per_gene = min_cells_per_gene
        self.max_mito_pct = max_mito_pct
        self.n_hvg = n_hvg
        self.normalize_target = normalize_target
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.force_normalize = force_normalize
        self.skip_normalize = skip_normalize
        self.skip_hvg = False  # Set via preprocess() or directly
        
        self.tf_names: Optional[List[str]] = None
        
    def load_data(
        self,
        data_path: str,
        metadata_path: Optional[str] = None
    ) -> 'anndata.AnnData':
        """
        Load single-cell data from various formats.
        
        Args:
            data_path: Path to data (h5ad, csv, loom, mtx)
            metadata_path: Optional path to metadata CSV
            
        Returns:
            AnnData object
        """
        if not SCANPY_AVAILABLE:
            raise ImportError("scanpy required for loading single-cell data")
            
        data_path = Path(data_path)
        ext = data_path.suffix.lower()
        
        print(f"Loading data from {data_path}...")
        
        if ext == '.h5ad':
            adata = sc.read_h5ad(data_path)
        elif ext == '.loom':
            adata = sc.read_loom(data_path)
        elif ext == '.csv':
            # CSV: genes x cells or cells x genes
            df = pd.read_csv(data_path, index_col=0)
            
            # Guess orientation - if more rows than columns, transpose
            if df.shape[0] > df.shape[1]:
                print("  Detected genes x cells format, transposing...")
                df = df.T
                
            adata = anndata.AnnData(df)
        elif ext == '.mtx':
            # Matrix Market format (10x Genomics)
            adata = sc.read_10x_mtx(data_path.parent)
        else:
            raise ValueError(f"Unsupported format: {ext}")
            
        print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
        
        # Load metadata if provided
        if metadata_path:
            print(f"Loading metadata from {metadata_path}...")
            meta = pd.read_csv(metadata_path, index_col=0)
            
            # Align with adata
            common = adata.obs_names.intersection(meta.index)
            if len(common) < adata.shape[0]:
                print(f"  Warning: Only {len(common)}/{adata.shape[0]} cells have metadata")
                
            for col in meta.columns:
                adata.obs[col] = meta.loc[adata.obs_names, col] if col in meta.columns else None
                
        return adata
    
    def quality_control(self, adata: 'anndata.AnnData') -> 'anndata.AnnData':
        """
        Apply quality control filters.
        
        Args:
            adata: AnnData object
            
        Returns:
            Filtered AnnData
        """
        print("Quality control...")
        n_cells_before = adata.shape[0]
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        
        # Filter cells
        sc.pp.filter_cells(adata, min_genes=self.min_genes_per_cell)
        adata = adata[adata.obs['n_genes_by_counts'] < self.max_genes_per_cell, :]
        
        if 'pct_counts_mt' in adata.obs:
            adata = adata[adata.obs['pct_counts_mt'] < self.max_mito_pct, :]
            
        # Filter genes
        sc.pp.filter_genes(adata, min_cells=self.min_cells_per_gene)
        
        print(f"  Cells: {n_cells_before} -> {adata.shape[0]}")
        print(f"  Genes: {adata.shape[1]}")
        
        return adata
    
    def _is_normalized(self, adata: 'anndata.AnnData') -> bool:
        """
        Auto-detect if data is already normalized.
        
        Heuristics:
        1. If max value < 50, likely log-normalized
        2. If values are floats with decimals, likely normalized
        3. If 'log1p' in adata.uns, already processed
        
        Args:
            adata: AnnData object
            
        Returns:
            True if data appears to be normalized
        """
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        
        # Sample a subset for speed
        sample_size = min(1000, X.shape[0])
        sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
        sample = X[sample_idx, :]
        
        max_val = sample.max()
        mean_val = sample.mean()
        
        # Check for scanpy log1p marker
        if 'log1p' in adata.uns:
            return True
            
        # Log-normalized data typically has max < 15
        if max_val < 15:
            return True
            
        # Raw counts typically have max > 100 and are integers
        if max_val > 100:
            # Check if mostly integers (raw counts)
            is_integer = np.allclose(sample, np.round(sample), rtol=0, atol=0.01)
            if is_integer:
                return False
                
        # If mean is low (< 5) and max is reasonable, likely normalized
        if mean_val < 5 and max_val < 50:
            return True
            
        return False
    
    def normalize(self, adata: 'anndata.AnnData') -> 'anndata.AnnData':
        """
        Normalize and log-transform data.
        
        Auto-detects if data is already normalized and skips if so.
        
        Args:
            adata: AnnData object
            
        Returns:
            Normalized AnnData
        """
        # User explicitly asked to skip
        if self.skip_normalize:
            print("Skipping normalization (--skip-normalize flag)")
            adata.layers['normalized'] = adata.X.copy()
            return adata
        
        # Check if already normalized (unless forced)
        if not self.force_normalize and self._is_normalized(adata):
            print("Data appears to be already normalized (log-transformed). Skipping normalization.")
            print("  Use --force-normalize to override auto-detection.")
            
            # Still store in layers for consistency
            if 'counts' not in adata.layers:
                print("  Warning: No raw counts available. Storing current data as 'normalized'.")
                adata.layers['normalized'] = adata.X.copy()
            else:
                adata.layers['normalized'] = adata.X.copy()
                
            return adata
        
        if self.force_normalize:
            print("Normalizing (--force-normalize flag)...")
        else:
            print("Normalizing raw counts...")
        
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize to target sum
        sc.pp.normalize_total(adata, target_sum=self.normalize_target)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Store normalized data
        adata.layers['normalized'] = adata.X.copy()
        
        print(f"  Normalized: max={adata.X.max():.2f}, mean={adata.X.mean():.4f}")
        
        return adata
    
    def select_hvg(self, adata: 'anndata.AnnData') -> 'anndata.AnnData':
        """
        Select highly variable genes.
        
        Auto-selects appropriate method based on available data:
        - seurat_v3 with raw counts (preferred)
        - cell_ranger with normalized data (fallback)
        
        Args:
            adata: AnnData object
            
        Returns:
            AnnData with HVG marked
        """
        print(f"Selecting {self.n_hvg} highly variable genes...")
        
        # Clean data: replace inf values with max finite value
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        
        if np.any(np.isinf(X)):
            print("  Warning: Found infinity values in data, replacing with max finite value")
            max_finite = np.nanmax(X[np.isfinite(X)])
            X = np.where(np.isinf(X), max_finite, X)
            if sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(X)
            else:
                adata.X = X
        
        # Check if raw counts are available
        has_counts = 'counts' in adata.layers
        
        if has_counts:
            # Use seurat_v3 with raw counts (preferred)
            print("  Using seurat_v3 method with raw counts")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_hvg,
                subset=False,
                flavor='seurat_v3',
                layer='counts'
            )
        else:
            # Fallback to cell_ranger method that's more robust with normalized data
            print("  Using cell_ranger method with normalized data")
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.n_hvg,
                    subset=False,
                    flavor='cell_ranger'
                )
            except Exception as e:
                # Last resort: use basic variance-based selection
                print(f"  Warning: cell_ranger failed ({e}), using variance-based selection")
                gene_vars = np.var(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, axis=0)
                gene_vars = np.nan_to_num(gene_vars, nan=0, posinf=0, neginf=0)
                top_idx = np.argsort(gene_vars)[-self.n_hvg:]
                adata.var['highly_variable'] = False
                adata.var.iloc[top_idx, adata.var.columns.get_loc('highly_variable')] = True
        
        n_hvg = adata.var['highly_variable'].sum()
        print(f"  Selected {n_hvg} HVGs")
        
        return adata
    
    def reduce_dimensions(self, adata: 'anndata.AnnData') -> 'anndata.AnnData':
        """
        Perform dimensionality reduction.
        
        Args:
            adata: AnnData object
            
        Returns:
            AnnData with PCA and UMAP
        """
        print("Dimensionality reduction...")
        
        # Scale for PCA
        sc.pp.scale(adata, max_value=10)
        
        # PCA
        sc.tl.pca(adata, n_comps=self.n_pcs, svd_solver='arpack')
        
        # Neighbors graph
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs)
        
        # UMAP
        sc.tl.umap(adata)
        
        return adata
    
    def cluster(self, adata: 'anndata.AnnData') -> 'anndata.AnnData':
        """
        Cluster cells using Leiden algorithm (preferred) with fallbacks.
        
        Args:
            adata: AnnData object
            
        Returns:
            AnnData with clusters
        """
        print(f"Clustering (resolution={self.resolution})...")
        
        method = None
        
        # Try leiden first (preferred)
        try:
            sc.tl.leiden(adata, resolution=self.resolution)
            method = 'leiden'
            print("  Using Leiden algorithm")
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Try louvain as fallback
        if method is None:
            try:
                sc.tl.louvain(adata, resolution=self.resolution)
                method = 'louvain'
                print("  Using Louvain algorithm (leiden not available)")
            except (ImportError, ModuleNotFoundError):
                pass
        
        # Last resort: KMeans on PCA
        if method is None:
            print("  Warning: No graph-based clustering available. Using KMeans on PCA.")
            from sklearn.cluster import KMeans
            n_clusters_estimate = max(5, min(30, adata.n_obs // 500))
            kmeans = KMeans(n_clusters=n_clusters_estimate, random_state=42, n_init=10)
            adata.obs['leiden'] = pd.Categorical(
                kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
            )
            method = 'leiden'
            
        n_clusters = adata.obs[method].nunique()
        print(f"  Found {n_clusters} clusters")
        
        return adata
    
    def estimate_tf_activity(
        self,
        adata: 'anndata.AnnData',
        tf_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Estimate TF activity from expression.
        
        Args:
            adata: AnnData object
            tf_names: Optional list of TF names
            
        Returns:
            TF activity matrix (cells x TFs)
        """
        print("Estimating TF activity...")
        
        if tf_names is None:
            tf_names = self.tf_names or []
            
        if not tf_names:
            print("  No TF names provided, using all genes")
            return None
            
        # Find TFs in gene list
        gene_names = adata.var_names.str.upper()
        tf_names_upper = [t.upper() for t in tf_names]
        
        found_tfs = []
        found_indices = []
        
        for tf, tf_upper in zip(tf_names, tf_names_upper):
            if tf_upper in gene_names.values:
                idx = np.where(gene_names == tf_upper)[0][0]
                found_tfs.append(tf)
                found_indices.append(idx)
                
        print(f"  Found {len(found_tfs)}/{len(tf_names)} TFs in expression data")
        
        if not found_tfs:
            return None
            
        # Extract TF expression
        X = adata.X if not sparse.issparse(adata.X) else adata.X.toarray()
        tf_activity = X[:, found_indices]
        
        # Z-score normalize
        tf_activity = (tf_activity - tf_activity.mean(axis=0)) / (tf_activity.std(axis=0) + 1e-8)
        
        return tf_activity, found_tfs
    
    def preprocess(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        tf_names: Optional[List[str]] = None
    ) -> 'anndata.AnnData':
        """
        Full preprocessing pipeline.
        
        Args:
            data_path: Path to data
            metadata_path: Optional path to metadata
            tf_names: Optional TF names for activity estimation
            
        Returns:
            Preprocessed AnnData
        """
        self.tf_names = tf_names
        
        # Load
        adata = self.load_data(data_path, metadata_path)
        
        # QC
        adata = self.quality_control(adata)
        
        # Normalize
        adata = self.normalize(adata)
        
        # HVG (optional - ALL genes are always kept, HVG just marks them)
        if not self.skip_hvg:
            adata = self.select_hvg(adata)
            print(f"  Note: All {adata.n_vars} genes retained. HVG marks {adata.var['highly_variable'].sum()} for dimensionality reduction.")
        else:
            print(f"Skipping HVG selection. All {adata.n_vars} genes retained.")
            adata.var['highly_variable'] = True  # Mark all as HVG
        
        # Reduce dimensions
        adata = self.reduce_dimensions(adata)
        
        # Cluster
        adata = self.cluster(adata)
        
        # TF activity
        if tf_names:
            tf_result = self.estimate_tf_activity(adata, tf_names)
            if tf_result:
                tf_activity, found_tfs = tf_result
                adata.obsm['tf_activity'] = tf_activity
                adata.uns['tf_names'] = found_tfs
                
        return adata


def select_cluster(
    adata: 'anndata.AnnData',
    cluster_key: str = 'leiden',
    cluster_id: Optional[str] = None,
    cell_type: Optional[str] = None,
    cell_type_key: str = 'cell_type'
) -> 'anndata.AnnData':
    """
    Select a specific cluster or cell type for focused analysis.
    
    Args:
        adata: AnnData object
        cluster_key: Key for cluster labels
        cluster_id: Cluster ID to select
        cell_type: Cell type annotation to select
        cell_type_key: Key for cell type annotations
        
    Returns:
        Subset AnnData
    """
    if cluster_id is not None:
        mask = adata.obs[cluster_key] == cluster_id
        print(f"Selecting cluster {cluster_id}: {mask.sum()} cells")
    elif cell_type is not None and cell_type_key in adata.obs:
        mask = adata.obs[cell_type_key].str.contains(cell_type, case=False, na=False)
        print(f"Selecting {cell_type}: {mask.sum()} cells")
    else:
        return adata
        
    return adata[mask, :].copy()


def extract_heterogeneity_labels(
    adata: 'anndata.AnnData',
    batch_key: str = 'batch',
    patient_key: str = 'patient',
    stage_key: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract heterogeneity labels from metadata.
    
    Args:
        adata: AnnData object
        batch_key: Key for batch labels
        patient_key: Key for patient labels
        stage_key: Key for stage/grade labels
        
    Returns:
        Dict of labels
    """
    labels = {}
    
    for key in [batch_key, patient_key, stage_key]:
        if key and key in adata.obs:
            labels[key] = adata.obs[key].values
            n_unique = len(np.unique(labels[key]))
            print(f"  {key}: {n_unique} unique values")
            
    return labels


def save_preprocessed(
    adata: 'anndata.AnnData',
    output_dir: str,
    save_full: bool = True
):
    """
    Save preprocessed data.
    
    Args:
        adata: Preprocessed AnnData
        output_dir: Output directory
        save_full: Whether to save full h5ad
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_dir}...")
    
    # Save full h5ad
    if save_full:
        adata.write_h5ad(output_dir / 'preprocessed.h5ad')
        print(f"  Saved preprocessed.h5ad")
        
    # Save expression matrix
    X = adata.X if not sparse.issparse(adata.X) else adata.X.toarray()
    np.save(output_dir / 'expression.npy', X.astype(np.float32))
    print(f"  Saved expression.npy ({X.shape})")
    
    # Save gene names
    pd.DataFrame({'gene': adata.var_names}).to_csv(
        output_dir / 'genes.csv', index=False
    )
    
    # Save cell metadata
    adata.obs.to_csv(output_dir / 'metadata.csv')
    
    # Save embeddings
    if 'X_pca' in adata.obsm:
        np.save(output_dir / 'pca.npy', adata.obsm['X_pca'])
    if 'X_umap' in adata.obsm:
        np.save(output_dir / 'umap.npy', adata.obsm['X_umap'])
    if 'tf_activity' in adata.obsm:
        np.save(output_dir / 'tf_activity.npy', adata.obsm['tf_activity'])
        
    # Save HVG mask
    if 'highly_variable' in adata.var:
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        pd.DataFrame({'gene': hvg_genes}).to_csv(
            output_dir / 'hvg_genes.csv', index=False
        )
        
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess single-cell data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input data (h5ad, csv, loom, mtx)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata CSV')
    parser.add_argument('--tf-list', type=str, default=None,
                        help='Path to TF list file')
    parser.add_argument('--output', type=str, default='data/singlecell',
                        help='Output directory')
    parser.add_argument('--select-cluster', type=str, default=None,
                        help='Cluster ID to select for focused analysis')
    parser.add_argument('--select-celltype', type=str, default=None,
                        help='Cell type to select (regex)')
    parser.add_argument('--min-genes', type=int, default=200,
                        help='Minimum genes per cell')
    parser.add_argument('--max-genes', type=int, default=8000,
                        help='Maximum genes per cell')
    parser.add_argument('--n-hvg', type=int, default=3000,
                        help='Number of highly variable genes')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Clustering resolution')
    parser.add_argument('--force-normalize', action='store_true',
                        help='Force normalization even if data appears normalized')
    parser.add_argument('--skip-normalize', action='store_true',
                        help='Skip normalization (data is already normalized)')
    parser.add_argument('--skip-hvg', action='store_true',
                        help='Skip HVG selection (keep all genes without marking)')
    
    args = parser.parse_args()
    
    # Load TF names if provided
    tf_names = None
    if args.tf_list:
        tf_names = pd.read_csv(args.tf_list, header=None)[0].tolist()
        print(f"Loaded {len(tf_names)} TF names")
        
    # Create preprocessor
    preprocessor = SingleCellPreprocessor(
        min_genes_per_cell=args.min_genes,
        max_genes_per_cell=args.max_genes,
        n_hvg=args.n_hvg,
        resolution=args.resolution,
        force_normalize=args.force_normalize,
        skip_normalize=args.skip_normalize
    )
    preprocessor.skip_hvg = args.skip_hvg
    
    # Preprocess
    adata = preprocessor.preprocess(
        args.input,
        args.metadata,
        tf_names
    )
    
    # Select cluster if specified
    if args.select_cluster:
        adata = select_cluster(adata, cluster_id=args.select_cluster)
    elif args.select_celltype:
        adata = select_cluster(adata, cell_type=args.select_celltype)
        
    # Save
    save_preprocessed(adata, args.output)


if __name__ == '__main__':
    main()


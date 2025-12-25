"""
Cell State Encoder for SingleCell-GRN

Encodes single-cell RNA-seq data to capture cell state and heterogeneity.
Inspired by LINGER's neural network approach for gene regulatory inference.

Key features:
- Per-gene neural networks (LINGER-style)
- Cell heterogeneity encoding (batch/patient prediction)
- TF activity estimation from expression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union

import numpy as np


class GenePredictor(nn.Module):
    """
    Per-gene neural network predictor.
    
    LINGER uses individual models for each gene, predicting gene expression
    from TF and RE (regulatory element) features.
    
    Architecture from LINGER:
    - Input: All TFs + REs within 1Mb of TSS
    - Hidden 1: 64 neurons, ReLU
    - Hidden 2: 16 neurons, ReLU
    - Output: Single value (gene expression)
    
    Args:
        n_input_features: Number of input features (TFs + REs)
        hidden1: First hidden layer size
        hidden2: Second hidden layer size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_input_features: int,
        hidden1: int = 64,
        hidden2: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_input_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict gene expression.
        
        Args:
            x: Input features (batch, n_features)
            
        Returns:
            Predicted expression (batch, 1)
        """
        return self.network(x)


class SharedGeneEncoder(nn.Module):
    """
    Shared encoder for all genes with gene-specific output heads.
    
    More efficient than per-gene models for large gene sets.
    Uses shared representations with gene-specific final layers.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs (input features)
        hidden_dim: Hidden dimension
        n_layers: Number of shared layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        
        # Shared encoder layers
        layers = [
            nn.Linear(n_tfs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
        self.shared_encoder = nn.Sequential(*layers)
        
        # Gene-specific output heads (as single matrix for efficiency)
        self.gene_heads = nn.Linear(hidden_dim, n_genes)
        
    def forward(self, tf_expression: torch.Tensor) -> torch.Tensor:
        """
        Predict all gene expressions from TF activities.
        
        Args:
            tf_expression: TF expression/activity (batch, n_tfs)
            
        Returns:
            Predicted gene expression (batch, n_genes)
        """
        hidden = self.shared_encoder(tf_expression)
        return self.gene_heads(hidden)
    
    def get_regulatory_weights(self) -> torch.Tensor:
        """
        Get TF-Gene regulatory weights from first layer.
        
        Returns:
            Weight matrix (n_tfs, hidden_dim)
        """
        return self.shared_encoder[0].weight.detach()


class CellStateEncoder(nn.Module):
    """
    Main cell state encoder for single-cell data.
    
    Captures:
    1. Cell heterogeneity (patient/batch/cell type differences)
    2. TF activity inference from expression
    3. Gene-gene co-expression patterns
    
    Inspired by LINGER's approach but extended for:
    - Direct expression input (without ATAC-seq)
    - Batch effect modeling
    - Multi-sample integration
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        n_batches: Number of batches/patients for heterogeneity
        hidden_dim: Hidden dimension
        n_layers: Number of encoder layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        n_batches: int = 1,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.n_batches = n_batches
        self.hidden_dim = hidden_dim
        
        # Gene expression encoder
        # Compress high-dim expression to cell embedding
        self.expression_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # TF activity encoder
        # Learns to extract TF activities from expression
        self.tf_activity_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tfs),
            nn.Tanh()  # Activities in [-1, 1]
        )
        
        # Batch/heterogeneity embedding (learned per batch)
        if n_batches > 1:
            self.batch_embedding = nn.Embedding(n_batches, hidden_dim // 4)
        else:
            self.batch_embedding = None
            
        # Cell state transformer
        # Self-attention to model gene-gene relationships
        encoder_dim = hidden_dim + (hidden_dim // 4 if n_batches > 1 else 0)
        
        self.cell_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=8,
                dim_feedforward=encoder_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Batch prediction head (for heterogeneity learning)
        if n_batches > 1:
            self.batch_predictor = nn.Linear(hidden_dim, n_batches)
        else:
            self.batch_predictor = None
            
    def forward(
        self,
        expression: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        return_tf_activity: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode cell states from expression.
        
        Args:
            expression: Gene expression matrix (batch, n_genes)
            batch_ids: Batch/patient IDs (batch,)
            return_tf_activity: Whether to return TF activities
            
        Returns:
            Dict with:
                - 'cell_embedding': Cell state embedding (batch, hidden_dim)
                - 'tf_activity': TF activity scores (batch, n_tfs) if requested
                - 'batch_logits': Batch prediction logits if available
        """
        # Encode expression to cell embedding
        cell_emb = self.expression_encoder(expression)  # (batch, hidden_dim)
        
        # Add batch embedding if batch encoding is enabled
        if self.batch_embedding is not None:
            if batch_ids is not None:
                batch_emb = self.batch_embedding(batch_ids)  # (batch, hidden//4)
            else:
                # Use zeros when batch_ids not provided
                batch_size = expression.size(0)
                batch_emb = torch.zeros(
                    batch_size, self.hidden_dim // 4, 
                    device=expression.device, dtype=expression.dtype
                )
            cell_emb = torch.cat([cell_emb, batch_emb], dim=-1)
            
        # Apply transformer (treating cell as single token)
        cell_emb = cell_emb.unsqueeze(1)  # (batch, 1, dim)
        cell_emb = self.cell_transformer(cell_emb)
        cell_emb = cell_emb.squeeze(1)  # (batch, dim)
        
        # Project to output
        output_emb = self.output_proj(cell_emb)
        
        outputs = {'cell_embedding': output_emb}
        
        # TF activity inference
        if return_tf_activity:
            tf_activity = self.tf_activity_encoder(expression)
            outputs['tf_activity'] = tf_activity
            
        # Batch prediction (for heterogeneity learning)
        if self.batch_predictor is not None:
            batch_logits = self.batch_predictor(output_emb)
            outputs['batch_logits'] = batch_logits
            
        return outputs
    
    def get_cell_heterogeneity_loss(
        self,
        expression: torch.Tensor,
        batch_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute heterogeneity loss (batch prediction).
        
        This encourages the model to learn batch-specific patterns,
        capturing cell heterogeneity across patients/conditions.
        
        Args:
            expression: Gene expression (batch, n_genes)
            batch_ids: True batch labels (batch,)
            
        Returns:
            Cross-entropy loss for batch prediction
        """
        outputs = self.forward(expression, batch_ids)
        
        if 'batch_logits' not in outputs:
            return torch.tensor(0.0, device=expression.device)
            
        return F.cross_entropy(outputs['batch_logits'], batch_ids)


class SingleCellDataEncoder(nn.Module):
    """
    Complete single-cell data encoder combining cell state and TF encoders.
    
    This is the main entry point for encoding single-cell data.
    
    Architecture:
    1. Cell State Encoder: Expression -> Cell embedding
    2. TF Activity Inference: Expression -> TF activities
    3. Gene-Gene Co-expression: Transformer attention
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        n_batches: int = 1,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_gene_coexpression: bool = False  # Disabled by default - too memory intensive
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        
        # Cell state encoder
        self.cell_encoder = CellStateEncoder(
            n_genes=n_genes,
            n_tfs=n_tfs,
            n_batches=n_batches,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Gene co-expression features (optional) - uses efficient method
        self.use_gene_coexpression = use_gene_coexpression
        if use_gene_coexpression:
            # Use efficient linear projection instead of full attention
            # Full gene-gene attention is O(n_genes^2) which is too expensive
            self.coexp_proj = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            
        # Final fusion
        fusion_dim = hidden_dim * (2 if use_gene_coexpression else 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        expression: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode single-cell data.
        
        Args:
            expression: Gene expression (batch, n_genes)
            batch_ids: Batch/patient IDs (batch,)
            
        Returns:
            Dict with:
                - 'cell_embedding': Fused cell embedding
                - 'tf_activity': Inferred TF activities
                - 'heterogeneity_loss': Batch prediction loss
        """
        # Get cell state encoding
        cell_outputs = self.cell_encoder(
            expression, batch_ids, return_tf_activity=True
        )
        
        cell_emb = cell_outputs['cell_embedding']
        
        # Gene co-expression features (optional)
        if self.use_gene_coexpression:
            # Efficient projection-based co-expression features
            # Avoids O(n_genes^2) attention matrix
            coexp_features = self.coexp_proj(expression)
            
            # Fuse
            fused = torch.cat([cell_emb, coexp_features], dim=-1)
            cell_emb = self.fusion(fused)
        
        outputs = {
            'cell_embedding': cell_emb,
            'tf_activity': cell_outputs['tf_activity']
        }
        
        # Add heterogeneity loss if applicable
        if 'batch_logits' in cell_outputs:
            outputs['batch_logits'] = cell_outputs['batch_logits']
            
        return outputs


def create_cell_encoder_from_adata(
    adata,
    batch_key: str = 'batch',
    tf_list: Optional[List[str]] = None,
    hidden_dim: int = 512
) -> SingleCellDataEncoder:
    """
    Create cell encoder from AnnData object.
    
    Args:
        adata: AnnData object with expression data
        batch_key: Key in adata.obs for batch labels
        tf_list: List of TF gene names (if None, auto-detect)
        hidden_dim: Hidden dimension
        
    Returns:
        Configured SingleCellDataEncoder
    """
    n_genes = adata.n_vars
    
    # Get number of batches
    if batch_key in adata.obs.columns:
        n_batches = adata.obs[batch_key].nunique()
    else:
        n_batches = 1
        
    # Get TF count
    if tf_list is not None:
        n_tfs = len(tf_list)
    else:
        # Estimate from typical TF count
        n_tfs = min(1400, n_genes // 10)
        
    return SingleCellDataEncoder(
        n_genes=n_genes,
        n_tfs=n_tfs,
        n_batches=n_batches,
        hidden_dim=hidden_dim
    )


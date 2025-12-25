"""
Monster GRN Model - Complete Integration for SingleCell-GRN

Integrates all components from multiple projects:

1. LINGER (Nature Biotechnology 2024):
   - Per-gene neural networks for expression prediction
   - TF potential calculation from binding + expression
   - In silico perturbation simulation

2. DeepCRE:
   - CNN-based cis-regulatory element prediction
   - DNA sequence feature extraction

3. DPFunc/Pro2Func:
   - Multi-modal GCN with attention
   - PPI network integration
   - Transformer fusion blocks

4. TFNET Monster Model:
   - GenSAN (Self-Attention Network)
   - FunDNN (Functional Deep Neural Network)
   - Multi-network gene embeddings

This is the main model for gene regulatory network inference.

Architecture:
==============================================================================
                           MONSTER GRN MODEL                              
==============================================================================
  Single-Cell      TF Binding        Gene Networks      DNA Sequences
  Expression       (ChIP-Atlas)      (STRING, etc)      (optional)
      |                |                   |                  |
      v                v                   v                  v
  Cell State       TF Regulatory      Multi-Network       Sequence
    Encoder           Encoder          GCN Encoder        Encoder
      |                   |                |                  |
      +-------------------+----------------+------------------+
                          |
                          v
                 Transformer Fusion
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
      TF Activity    Edge Pred     Pathway
         Head          Head          Head
            |             |             |
            +-------------+-------------+
                          |
                          v
              Perturbation Simulator
==============================================================================
"""

import math
import copy
from typing import Optional, Dict, List, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell_encoder import CellStateEncoder, SingleCellDataEncoder
from .tf_encoder import TFRegulatoryEncoder, ChIPAtlasLoader

# Try to import DGL
try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    DGL_AVAILABLE = False
    dgl = None
    dglnn = None


class LayerNorm(nn.Module):
    """Layer normalization with learnable parameters."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class MultiNetworkGCNEncoder(nn.Module):
    """
    Multi-network GCN encoder from DPFunc/Pro2Func.
    
    Processes multiple gene networks in parallel and fuses them.
    
    Args:
        n_genes: Number of genes
        hidden_dim: Hidden dimension
        num_layers: Number of GCN layers
        num_heads: Number of attention heads for fusion
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.use_dgl = DGL_AVAILABLE
        
        # Gene embeddings (learnable baseline)
        self.gene_embeddings = nn.Embedding(n_genes, hidden_dim)
        nn.init.xavier_uniform_(self.gene_embeddings.weight)
        
        # Network names
        self.network_names = ['string', 'corum', 'pathway', 'disease', 'drug']
        
        # Per-network GCN layers
        if self.use_dgl:
            self.network_gcns = nn.ModuleDict()
            self.network_bns = nn.ModuleDict()
            
            for name in self.network_names:
                self.network_gcns[name] = nn.ModuleList([
                    dglnn.GraphConv(hidden_dim, hidden_dim)
                    for _ in range(num_layers)
                ])
                self.network_bns[name] = nn.ModuleList([
                    nn.BatchNorm1d(hidden_dim)
                    for _ in range(num_layers)
                ])
        else:
            # Fallback: Simple linear message passing
            self.network_gcns = nn.ModuleDict()
            for name in self.network_names:
                layers = []
                for _ in range(num_layers):
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ])
                self.network_gcns[name] = nn.Sequential(*layers)
                
        # Cross-network fusion with multi-head attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        network_graphs: Optional[Dict[str, Any]] = None,
        network_adjs: Optional[Dict[str, torch.Tensor]] = None,
        gene_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode genes using multiple networks.
        
        Args:
            network_graphs: Dict of DGL graphs (for DGL mode)
            network_adjs: Dict of adjacency matrices (for fallback)
            gene_indices: Specific gene indices to return
            
        Returns:
            Gene embeddings (n_genes, hidden_dim)
        """
        # Get base embeddings
        if gene_indices is not None:
            h_base = self.gene_embeddings(gene_indices)
            n_nodes = len(gene_indices)
        else:
            h_base = self.gene_embeddings.weight
            n_nodes = self.n_genes
            
        # Process each network
        network_features = []
        
        for name in self.network_names:
            h = h_base
            
            if self.use_dgl and network_graphs is not None and name in network_graphs:
                g = network_graphs[name]
                for conv, bn in zip(self.network_gcns[name], self.network_bns[name]):
                    h_prev = h
                    h = conv(g, h)
                    h = bn(h)
                    h = F.gelu(h)
                    h = self.dropout(h)
                    h = h + h_prev  # Residual
            elif network_adjs is not None and name in network_adjs:
                adj = network_adjs[name]
                h = torch.mm(adj, h)
                h = self.network_gcns[name](h)
            else:
                # No network data, use base embedding
                pass
                
            network_features.append(h)
            
        # Stack: (n_genes, n_networks, hidden)
        stacked = torch.stack(network_features, dim=1)
        
        # Cross-network attention
        attended, _ = self.fusion_attention(stacked, stacked, stacked)
        fused = stacked.mean(dim=1) + self.dropout(attended.mean(dim=1))
        fused = self.fusion_norm(fused)
        
        # Feed-forward
        ff_out = self.fusion_ff(fused)
        output = self.ff_norm(fused + self.dropout(ff_out))
        
        return output


class TransformerFusionBlock(nn.Module):
    """
    Transformer fusion block for combining multiple embeddings.
    
    Uses cross-attention to fuse:
    - Cell state embeddings
    - TF regulatory embeddings
    - Network-derived gene embeddings
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Self-attention for each input type
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(3)  # cell, tf, network
        ])
        
        # Cross-attention layers
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2 + 3)
        ])
        
        # Feed-forward
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        cell_emb: torch.Tensor,
        tf_emb: torch.Tensor,
        network_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse three types of embeddings.
        
        Args:
            cell_emb: Cell state embedding (batch, hidden)
            tf_emb: TF regulatory embedding (batch, hidden)
            network_emb: Network-derived embedding (batch, hidden)
            
        Returns:
            Fused embedding (batch, hidden)
        """
        # Reshape to sequence format
        cell_emb = cell_emb.unsqueeze(1)  # (batch, 1, hidden)
        tf_emb = tf_emb.unsqueeze(1)
        network_emb = network_emb.unsqueeze(1)
        
        # Self-attention on each
        cell_emb, _ = self.self_attns[0](cell_emb, cell_emb, cell_emb)
        cell_emb = self.norms[0](cell_emb)
        
        tf_emb, _ = self.self_attns[1](tf_emb, tf_emb, tf_emb)
        tf_emb = self.norms[1](tf_emb)
        
        network_emb, _ = self.self_attns[2](network_emb, network_emb, network_emb)
        network_emb = self.norms[2](network_emb)
        
        # Concatenate for cross-attention
        x = torch.cat([cell_emb, tf_emb, network_emb], dim=1)  # (batch, 3, hidden)
        
        norm_idx = 3
        for cross_attn, ff in zip(self.cross_attns, self.ffs):
            # Cross-attention
            attended, _ = cross_attn(x, x, x)
            x = x + self.dropout(attended)
            x = self.norms[norm_idx](x)
            norm_idx += 1
            
            # Feed-forward
            ff_out = ff(x)
            x = x + self.dropout(ff_out)
            x = self.norms[norm_idx](x)
            norm_idx += 1
            
        # Pool across the 3 tokens
        return x.mean(dim=1)


class GRNInferenceHead(nn.Module):
    """
    GRN inference head for predicting regulatory relationships.
    
    Outputs:
    1. TF activity scores per gene
    2. TF-Gene edge probabilities
    3. Pathway enrichment scores
    
    Args:
        hidden_dim: Input hidden dimension
        n_tfs: Number of TFs
        n_genes: Number of genes
        n_pathways: Number of pathways (KEGG/Reactome)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        n_tfs: int = 1390,
        n_genes: int = 12554,
        n_pathways: int = 330,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.n_genes = n_genes
        
        # TF activity predictor
        self.tf_activity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tfs),
            nn.Tanh()  # Activities in [-1, 1]
        )
        
        # Edge prediction head (TF-Gene)
        # Uses bilinear attention for edge scores
        self.edge_query = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, 1)
        
        # Pathway enrichment head
        self.pathway_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_pathways)
        )
        
    def forward(
        self,
        fused_emb: torch.Tensor,
        tf_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict GRN components.
        
        Args:
            fused_emb: Fused gene/cell embedding (batch, hidden)
            tf_features: TF embeddings (n_tfs, hidden) for edge prediction
            
        Returns:
            Dict with:
                - 'tf_activity': TF activity scores (batch, n_tfs)
                - 'edge_logits': Edge probabilities (batch, n_tfs) if tf_features given
                - 'pathway_logits': Pathway enrichment (batch, n_pathways)
        """
        outputs = {}
        
        # TF activity prediction
        outputs['tf_activity'] = self.tf_activity_head(fused_emb)
        
        # Edge prediction (gene queries TFs)
        if tf_features is not None:
            query = self.edge_query(fused_emb)  # (batch, hidden)
            key = self.edge_key(tf_features)  # (n_tfs, hidden)
            
            # Bilinear attention scores
            scores = torch.mm(query, key.t())  # (batch, n_tfs)
            outputs['edge_logits'] = scores
            
        # Pathway enrichment
        outputs['pathway_logits'] = self.pathway_head(fused_emb)
        
        return outputs


class PerturbationSimulator(nn.Module):
    """
    Perturbation simulator for in-silico TF perturbations.
    
    Given a TF knockout/overexpression, predicts the effect on:
    - All target gene expressions
    - Cell state changes
    
    Inspired by LINGER's in-silico perturbation approach.
    
    Args:
        hidden_dim: Hidden dimension
        n_tfs: Number of TFs
        n_genes: Number of genes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        n_tfs: int = 1390,
        n_genes: int = 12554,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.n_genes = n_genes
        
        # Perturbation encoder
        # Encodes which TFs are perturbed and how
        self.pert_type_emb = nn.Embedding(3, hidden_dim // 4)  # KO, OE, normal
        self.pert_tf_emb = nn.Embedding(n_tfs, hidden_dim)
        
        # Effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
            nn.Tanh()  # Expression changes in [-1, 1]
        )
        
        # Cell state change predictor
        self.state_change_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        base_embedding: torch.Tensor,
        tf_indices: torch.Tensor,
        perturbation_type: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate perturbation effects.
        
        Args:
            base_embedding: Base cell/gene embedding (batch, hidden)
            tf_indices: TF indices being perturbed (batch,)
            perturbation_type: 0=KO, 1=OE, 2=normal (batch,)
            
        Returns:
            Dict with:
                - 'expression_change': Predicted change for all genes (batch, n_genes)
                - 'state_change': Change in cell state embedding (batch, hidden)
        """
        # Encode perturbation
        pert_type = self.pert_type_emb(perturbation_type)  # (batch, hidden//4)
        pert_tf = self.pert_tf_emb(tf_indices)  # (batch, hidden)
        
        # Pad type embedding
        padding = torch.zeros(
            pert_type.size(0), 
            pert_tf.size(1) - pert_type.size(1),
            device=pert_type.device
        )
        pert_type = torch.cat([pert_type, padding], dim=-1)
        
        # Combine perturbation with base
        pert_emb = pert_tf + pert_type
        combined = torch.cat([base_embedding, pert_emb], dim=-1)
        
        # Predict effects
        expression_change = self.effect_predictor(combined)
        state_change = self.state_change_predictor(combined)
        
        return {
            'expression_change': expression_change,
            'state_change': state_change
        }


class MonsterGRNModel(nn.Module):
    """
    Monster GRN Model - Complete integration of all components.
    
    This is the main model for SingleCell-GRN.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs  
        n_batches: Number of batches/patients
        n_pathways: Number of pathways
        hidden_dim: Hidden dimension
        num_gcn_layers: Number of GCN layers
        num_transformer_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int = 12554,
        n_tfs: int = 1390,
        n_batches: int = 1,
        n_pathways: int = 330,
        hidden_dim: int = 512,
        num_gcn_layers: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.n_batches = n_batches
        self.hidden_dim = hidden_dim
        
        # Store config for checkpointing
        self.config = {
            'n_genes': n_genes,
            'n_tfs': n_tfs,
            'n_batches': n_batches,
            'n_pathways': n_pathways,
            'hidden_dim': hidden_dim,
            'num_gcn_layers': num_gcn_layers,
            'num_transformer_layers': num_transformer_layers,
            'num_heads': num_heads,
            'dropout': dropout
        }
        
        # ============================================================
        # Component 1: Cell State Encoder
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
        # Component 2: TF Regulatory Encoder (optional - used when binding_matrix provided)
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
        # Component 3: Multi-Network GCN Encoder
        # ============================================================
        self.network_encoder = MultiNetworkGCNEncoder(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            num_layers=num_gcn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ============================================================
        # Component 4: Transformer Fusion
        # ============================================================
        self.fusion = TransformerFusionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # ============================================================
        # Component 5: GRN Inference Head
        # ============================================================
        self.grn_head = GRNInferenceHead(
            hidden_dim=hidden_dim,
            n_tfs=n_tfs,
            n_genes=n_genes,
            n_pathways=n_pathways,
            dropout=dropout
        )
        
        # ============================================================
        # Component 6: Perturbation Simulator
        # ============================================================
        self.perturbation_sim = PerturbationSimulator(
            hidden_dim=hidden_dim,
            n_tfs=n_tfs,
            n_genes=n_genes,
            dropout=dropout
        )
        
        # ============================================================
        # Component 7: Expression Decoder (for reconstruction loss)
        # ============================================================
        self.expression_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_genes)
        )
        
        # Cached binding matrix for simplified forward
        self.register_buffer('_cached_binding', None)
        self.register_buffer('_cached_tf_tf_adj', None)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_binding_matrix(self, binding_matrix: torch.Tensor, tf_tf_adjacency: Optional[torch.Tensor] = None):
        """
        Cache binding matrix for simplified forward calls.
        
        Args:
            binding_matrix: TF-Gene binding matrix (n_genes, n_tfs)
            tf_tf_adjacency: Optional TF-TF adjacency
        """
        self._cached_binding = binding_matrix
        self._cached_tf_tf_adj = tf_tf_adjacency
                    
    def forward(
        self,
        expression: torch.Tensor,
        binding_matrix: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        tf_tf_adjacency: Optional[torch.Tensor] = None,
        tf_activity: Optional[torch.Tensor] = None,
        network_graphs: Optional[Dict[str, Any]] = None,
        gene_indices: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Args:
            expression: Gene expression (batch, n_genes)
            binding_matrix: TF-Gene binding matrix (n_genes, n_tfs). Optional if cached.
            batch_ids: Batch/patient IDs (batch,)
            tf_tf_adjacency: TF-TF regulatory adjacency
            tf_activity: Optional pre-computed TF activity (batch, n_tfs)
            network_graphs: Dict of gene network graphs
            gene_indices: Specific genes to encode
            return_embeddings: Alias for return_all (for train.py compatibility)
            return_all: Return all intermediate outputs
            
        Returns:
            Dict with:
                - 'reconstructed': Reconstructed expression (batch, n_genes)
                - 'predicted_expression': Same as reconstructed (for compatibility)
                - 'tf_activity': TF activity scores (batch, n_tfs)
                - 'pathway_logits': Pathway predictions (batch, n_pathways)
                - 'fused_embedding': Fused representation (batch, hidden)
                - 'batch_logits': Batch/patient classification (if n_batches > 1)
                - 'edge_logits': TF-Gene edge predictions (batch, n_tfs)
        """
        batch_size = expression.size(0)
        device = expression.device
        return_all = return_all or return_embeddings
        
        # Use cached binding if not provided
        if binding_matrix is None:
            binding_matrix = self._cached_binding
        if tf_tf_adjacency is None:
            tf_tf_adjacency = self._cached_tf_tf_adj
        
        # Step 1: Cell state encoding
        cell_outputs = self.cell_encoder(expression, batch_ids)
        cell_emb = cell_outputs['cell_embedding']  # (batch, hidden)
        
        # Step 2: TF regulatory encoding (if binding matrix available)
        if binding_matrix is not None:
            tf_outputs = self.tf_encoder(
                binding_matrix, tf_tf_adjacency, gene_indices
            )
            tf_emb = tf_outputs['gene_features']  # (n_genes, hidden)
            tf_features = tf_outputs['tf_features']  # (n_tfs, hidden)
            tf_importance = tf_outputs.get('tf_importance', None)
        else:
            # Fallback: use cell embedding as TF context
            tf_emb = cell_emb
            # Access TF embeddings through the correct path
            if hasattr(self.tf_encoder, 'binding_encoder') and hasattr(self.tf_encoder.binding_encoder, 'tf_embeddings'):
                tf_features = self.tf_encoder.binding_encoder.tf_embeddings.embeddings.weight
            else:
                tf_features = None
            tf_importance = None
        
        # Handle dimension mismatch (tf_emb might be per-gene, not per-sample)
        if tf_emb.dim() == 2 and tf_emb.size(0) != batch_size:
            # Attention pooling: use cell_emb to attend over gene features
            attn_weights = torch.softmax(
                torch.matmul(cell_emb, tf_emb.T) / math.sqrt(self.hidden_dim), 
                dim=-1
            )  # (batch, n_genes)
            tf_emb_pooled = torch.matmul(attn_weights, tf_emb)  # (batch, hidden)
        else:
            tf_emb_pooled = tf_emb
            
        # Step 3: Network encoding  
        network_emb = self.network_encoder(network_graphs, gene_indices=gene_indices)
        
        # Handle dimension mismatch with attention pooling
        if network_emb.dim() == 2 and network_emb.size(0) != batch_size:
            attn_weights = torch.softmax(
                torch.matmul(cell_emb, network_emb.T) / math.sqrt(self.hidden_dim),
                dim=-1
            )
            network_emb_pooled = torch.matmul(attn_weights, network_emb)
        else:
            network_emb_pooled = network_emb
            
        # Step 4: Transformer fusion
        fused = self.fusion(cell_emb, tf_emb_pooled, network_emb_pooled)  # (batch, hidden)
        
        # Step 5: Expression reconstruction (decoder)
        reconstructed = self.expression_decoder(fused)  # (batch, n_genes)
        
        # Step 6: GRN inference
        grn_outputs = self.grn_head(fused, tf_features)
        
        # Build outputs
        outputs = {
            'reconstructed': reconstructed,
            'predicted_expression': reconstructed,  # Alias for compatibility
            'tf_activity': grn_outputs['tf_activity'],
            'pathway_logits': grn_outputs['pathway_logits'],
            'fused_embedding': fused
        }
        
        # Add edge logits
        if 'edge_logits' in grn_outputs:
            outputs['edge_logits'] = grn_outputs['edge_logits']
            
        # Add batch classification logits (for heterogeneity loss)
        if 'batch_logits' in cell_outputs:
            outputs['batch_logits'] = cell_outputs['batch_logits']
            
        # Add intermediate outputs if requested
        if return_all:
            outputs['cell_embedding'] = cell_emb
            outputs['tf_embedding'] = tf_emb_pooled
            outputs['network_embedding'] = network_emb_pooled
            if tf_importance is not None:
                outputs['tf_importance'] = tf_importance
            
        return outputs
    
    def simulate_perturbation(
        self,
        expression: torch.Tensor,
        binding_matrix: torch.Tensor,
        tf_indices: torch.Tensor,
        perturbation_type: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate TF perturbation effects.
        
        Args:
            expression: Base expression (batch, n_genes)
            binding_matrix: TF-Gene binding matrix
            tf_indices: TF indices to perturb (batch,)
            perturbation_type: 0=KO, 1=OE (batch,)
            batch_ids: Batch IDs
            
        Returns:
            Dict with:
                - 'original_outputs': Model outputs before perturbation
                - 'expression_change': Predicted expression changes
                - 'new_tf_activity': TF activity after perturbation
        """
        # Get base embedding
        outputs = self.forward(
            expression, binding_matrix, batch_ids, return_all=True
        )
        base_emb = outputs['fused_embedding']
        
        # Simulate perturbation
        pert_outputs = self.perturbation_sim(
            base_emb, tf_indices, perturbation_type
        )
        
        # New cell state after perturbation
        new_emb = base_emb + pert_outputs['state_change']
        
        # Predict new TF activity
        new_activity = self.grn_head.tf_activity_head(new_emb)
        
        return {
            'original_outputs': outputs,
            'expression_change': pert_outputs['expression_change'],
            'new_tf_activity': new_activity,
            'state_change': pert_outputs['state_change']
        }
    
    def find_regulators(
        self,
        expression: torch.Tensor,
        binding_matrix: torch.Tensor,
        target_gene_idx: int,
        direction: str = 'increase',
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Find TFs that regulate a target gene.
        
        Args:
            expression: Gene expression (batch, n_genes)
            binding_matrix: TF-Gene binding matrix
            target_gene_idx: Index of target gene
            direction: 'increase' or 'decrease'
            top_k: Number of top regulators to return
            
        Returns:
            Dict with top regulators and their scores
        """
        batch_size = expression.size(0)
        device = expression.device
        
        # Get base predictions
        outputs = self.forward(expression, binding_matrix)
        base_activity = outputs['tf_activity']  # (batch, n_tfs)
        
        # Try each TF perturbation
        regulator_scores = []
        
        for tf_idx in range(self.n_tfs):
            tf_indices = torch.full((batch_size,), tf_idx, device=device)
            
            # KO for increase, OE for decrease
            if direction == 'increase':
                pert_type = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                pert_type = torch.ones(batch_size, dtype=torch.long, device=device)
                
            # Simulate
            pert_outputs = self.simulate_perturbation(
                expression, binding_matrix, tf_indices, pert_type
            )
            
            # Get effect on target gene
            effect = pert_outputs['expression_change'][:, target_gene_idx]
            
            if direction == 'increase':
                score = effect.mean().item()
            else:
                score = -effect.mean().item()
                
            regulator_scores.append((tf_idx, score))
            
        # Sort by score
        regulator_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'top_regulators': regulator_scores[:top_k],
            'all_scores': regulator_scores
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_monster_grn(
    n_genes: int = 12554,
    n_tfs: int = 1390,
    n_batches: int = 1,
    n_pathways: int = 330,
    hidden_dim: Optional[int] = None,
    num_gcn_layers: Optional[int] = None,
    num_transformer_layers: Optional[int] = None,
    n_layers: Optional[int] = None,  # Alias for num_transformer_layers
    num_heads: Optional[int] = None,
    n_heads: Optional[int] = None,  # Alias for num_heads
    dropout: Optional[float] = None,
    config: str = 'medium',
    tf_tf_adjacency: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> MonsterGRNModel:
    """
    Factory function to create Monster GRN model.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        n_batches: Number of batches
        n_pathways: Number of pathways
        hidden_dim: Hidden dimension (overrides config)
        num_gcn_layers: GCN layers (overrides config)
        num_transformer_layers: Transformer layers (overrides config)
        n_layers: Alias for num_transformer_layers
        num_heads: Attention heads (overrides config)
        n_heads: Alias for num_heads
        dropout: Dropout rate (overrides config)
        config: 'small', 'medium', 'large' - base configuration
        tf_tf_adjacency: Optional TF-TF adjacency to cache
        device: Compute device
        
    Returns:
        Configured MonsterGRNModel
    """
    # Preset configurations
    configs = {
        'small': {
            'hidden_dim': 256,
            'num_gcn_layers': 2,
            'num_transformer_layers': 1,
            'num_heads': 4,
            'dropout': 0.1
        },
        'medium': {
            'hidden_dim': 512,
            'num_gcn_layers': 3,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        },
        'large': {
            'hidden_dim': 1024,
            'num_gcn_layers': 4,
            'num_transformer_layers': 3,
            'num_heads': 16,
            'dropout': 0.1
        }
    }
    
    # Start with preset config
    model_config = configs.get(config, configs['medium']).copy()
    
    # Override with explicit args
    if hidden_dim is not None:
        model_config['hidden_dim'] = hidden_dim
    if num_gcn_layers is not None:
        model_config['num_gcn_layers'] = num_gcn_layers
    if num_transformer_layers is not None:
        model_config['num_transformer_layers'] = num_transformer_layers
    if n_layers is not None:  # Alias
        model_config['num_transformer_layers'] = n_layers
    if num_heads is not None:
        model_config['num_heads'] = num_heads
    if n_heads is not None:  # Alias
        model_config['num_heads'] = n_heads
    if dropout is not None:
        model_config['dropout'] = dropout
    
    model = MonsterGRNModel(
        n_genes=n_genes,
        n_tfs=n_tfs,
        n_batches=n_batches,
        n_pathways=n_pathways,
        **model_config
    )
    
    # Cache TF-TF adjacency if provided
    if tf_tf_adjacency is not None:
        model._cached_tf_tf_adj = tf_tf_adjacency
    
    if device is not None:
        model = model.to(device)
        
    print(f"Created Monster GRN Model (base: {config})")
    print(f"  Genes: {n_genes}, TFs: {n_tfs}, Batches: {n_batches}")
    print(f"  Hidden: {model_config['hidden_dim']}, Heads: {model_config['num_heads']}")
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    return model


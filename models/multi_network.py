"""
Multi-Network Graph Encoder for SingleCell-GRN

Combines multiple biological networks for GRN inference:
- Gene co-expression networks
- Protein-protein interaction (PPI) networks
- TF-Gene regulatory networks
- Pathway membership networks

Inspired by:
- DPFunc: GCN layers with attention and transformer blocks
- Pro2Func: Multi-modal feature integration
- TFNET: Multi-network fusion

Architecture:
1. Per-network GCN encoders
2. Cross-network attention
3. Transformer fusion
4. Unified gene embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union

import numpy as np

# Try to import DGL
try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    DGL_AVAILABLE = False
    dgl = None
    dglnn = None


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer.
    
    Basic GCN with optional skip connections and layer normalization.
    Compatible with both DGL and simple adjacency matrix.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        dropout: Dropout rate
        use_skip: Use skip connection
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_skip = use_skip
        
        if use_skip and in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        else:
            self.skip_proj = None
            
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (n_nodes, in_dim)
            adj: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated features (n_nodes, out_dim)
        """
        # Normalize adjacency
        degree = adj.sum(dim=1, keepdim=True) + 1e-8
        adj_norm = adj / degree
        
        # Message passing
        h = torch.matmul(adj_norm, x)
        h = self.linear(h)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Skip connection
        if self.use_skip:
            if self.skip_proj is not None:
                x = self.skip_proj(x)
            h = h + x
            
        return h


class DGLGCNLayer(nn.Module):
    """
    DGL-based GCN layer with attention.
    
    Uses DGL's efficient graph operations when available.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("DGL required for DGLGCNLayer")
            
        self.conv = dglnn.GATConv(
            in_dim, out_dim // n_heads, n_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=F.elu
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, g: 'dgl.DGLGraph', h: torch.Tensor) -> torch.Tensor:
        """Forward pass with DGL graph."""
        h = self.conv(g, h).flatten(1)
        h = self.norm(h)
        return h


class TransformerBlock(nn.Module):
    """
    Transformer block for graph feature aggregation.
    
    From DPFunc: Combines residue features with protein-level features
    using attention mechanism.
    
    Args:
        hidden_dim: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query features (batch, seq_len, hidden)
            context: Context features (batch, ctx_len, hidden)
            
        Returns:
            Updated query features
        """
        # Cross-attention
        attn_out, _ = self.attention(query, context, context)
        query = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


class SingleNetworkEncoder(nn.Module):
    """
    Encoder for a single biological network.
    
    Processes one network (e.g., PPI, co-expression) to generate
    node embeddings.
    
    Args:
        n_nodes: Number of nodes
        hidden_dim: Hidden dimension
        n_layers: Number of GCN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # Initial node embeddings
        self.node_embeddings = nn.Embedding(n_nodes, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        adj: torch.Tensor,
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode network.
        
        Args:
            adj: Adjacency matrix (n_nodes, n_nodes)
            node_features: Optional input features (n_nodes, feat_dim)
            
        Returns:
            Node embeddings (n_nodes, hidden_dim)
        """
        if node_features is None:
            # Use learned embeddings
            node_indices = torch.arange(self.n_nodes, device=adj.device)
            h = self.node_embeddings(node_indices)
        else:
            h = node_features
            
        # GCN propagation
        for gcn in self.gcn_layers:
            h = gcn(h, adj)
            
        # Project output
        h = self.output_proj(h)
        
        return h


class MultiNetworkEncoder(nn.Module):
    """
    Multi-Network Graph Encoder.
    
    Processes multiple biological networks in parallel and fuses them
    into unified gene embeddings.
    
    Networks:
    1. Gene co-expression
    2. PPI (protein-protein interaction)
    3. TF-Gene regulatory
    4. Pathway membership
    
    From DPFunc + TFNET: Multi-modal feature integration with attention.
    
    Args:
        n_genes: Number of genes
        hidden_dim: Hidden dimension per network
        n_networks: Number of networks
        n_gcn_layers: GCN layers per network
        n_fusion_heads: Attention heads for fusion
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        n_networks: int = 4,
        n_gcn_layers: int = 2,
        n_fusion_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_networks = n_networks
        
        # Per-network encoders
        self.network_encoders = nn.ModuleList([
            SingleNetworkEncoder(n_genes, hidden_dim, n_gcn_layers, dropout)
            for _ in range(n_networks)
        ])
        
        # Network type embeddings
        self.network_type_emb = nn.Embedding(n_networks, hidden_dim)
        
        # Cross-network attention
        self.cross_attention = TransformerBlock(
            hidden_dim, n_fusion_heads, dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * n_networks, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        adjacencies: List[torch.Tensor],
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multiple networks.
        
        Args:
            adjacencies: List of adjacency matrices, one per network
                        Each is (n_genes, n_genes)
            node_features: Optional shared input features (n_genes, feat_dim)
            
        Returns:
            Fused gene embeddings (n_genes, hidden_dim)
        """
        device = adjacencies[0].device
        
        # Encode each network
        network_embeddings = []
        
        for i, (encoder, adj) in enumerate(zip(self.network_encoders, adjacencies)):
            h = encoder(adj, node_features)
            
            # Add network type embedding
            network_type = torch.tensor([i], device=device)
            type_emb = self.network_type_emb(network_type)  # (1, hidden)
            h = h + type_emb
            
            network_embeddings.append(h)
            
        # Stack for attention: (n_genes, n_networks, hidden)
        stacked = torch.stack(network_embeddings, dim=1)
        
        # Cross-network attention (each gene attends to its representations across networks)
        attended = self.cross_attention(stacked, stacked)  # (n_genes, n_networks, hidden)
        
        # Flatten and fuse
        flattened = attended.view(self.n_genes, -1)  # (n_genes, n_networks * hidden)
        fused = self.fusion(flattened)  # (n_genes, hidden)
        
        # Final projection
        output = self.output_proj(fused)
        
        return output


class PPIEncoder(nn.Module):
    """
    PPI-specific encoder (from DPFunc).
    
    Encodes protein-protein interaction features for each gene/protein.
    
    Args:
        ppi_dim: Input PPI feature dimension
        hidden_dim: Output dimension
    """
    
    def __init__(
        self,
        ppi_dim: int = 20,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(ppi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, ppi_features: torch.Tensor) -> torch.Tensor:
        """Encode PPI features."""
        return self.encoder(ppi_features)


class CoexpressionNetworkBuilder(nn.Module):
    """
    Builds co-expression network from single-cell expression data.
    
    Creates a gene-gene adjacency matrix based on expression correlation.
    
    Args:
        n_genes: Number of genes
        k_neighbors: Number of top correlated genes to connect
        min_correlation: Minimum correlation threshold
    """
    
    def __init__(
        self,
        n_genes: int,
        k_neighbors: int = 20,
        min_correlation: float = 0.3
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.k_neighbors = k_neighbors
        self.min_correlation = min_correlation
        
        # Cache for built network
        self.register_buffer('cached_adjacency', None)
        
    @torch.no_grad()
    def build_network(self, expression: torch.Tensor) -> torch.Tensor:
        """
        Build co-expression network from expression matrix.
        
        Args:
            expression: Gene expression (n_cells, n_genes)
            
        Returns:
            Adjacency matrix (n_genes, n_genes)
        """
        # Normalize expression
        expr_mean = expression.mean(dim=0, keepdim=True)
        expr_std = expression.std(dim=0, keepdim=True) + 1e-8
        expr_norm = (expression - expr_mean) / expr_std
        
        # Compute correlation matrix
        n_cells = expression.shape[0]
        correlation = torch.matmul(expr_norm.T, expr_norm) / n_cells  # (n_genes, n_genes)
        
        # Apply threshold
        adj = (correlation >= self.min_correlation).float()
        
        # Keep only top-k neighbors per gene
        if self.k_neighbors < self.n_genes:
            top_values, top_indices = torch.topk(
                correlation, self.k_neighbors, dim=1
            )
            
            mask = torch.zeros_like(correlation)
            mask.scatter_(1, top_indices, 1.0)
            adj = adj * mask
            
        # Make symmetric
        adj = (adj + adj.T) / 2
        
        # Remove self-loops
        adj = adj * (1 - torch.eye(self.n_genes, device=adj.device))
        
        return adj
        
    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """Build or return cached network."""
        if self.cached_adjacency is None:
            self.cached_adjacency = self.build_network(expression)
        return self.cached_adjacency


class IntegratedNetworkEncoder(nn.Module):
    """
    Complete integrated network encoder for SingleCell-GRN.
    
    Combines:
    - TF-Gene regulatory network (from ChIP-Atlas)
    - Gene co-expression network (from single-cell data)
    - PPI network (from STRING/BioGRID)
    - Pathway membership (from GO/KEGG)
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Hidden dimension
        use_coexpression: Build co-expression network from data
        use_ppi: Use PPI features
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int = 0,
        hidden_dim: int = 256,
        use_coexpression: bool = True,
        use_ppi: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.hidden_dim = hidden_dim
        self.use_coexpression = use_coexpression
        self.use_ppi = use_ppi
        
        # Network builders
        if use_coexpression:
            self.coexpr_builder = CoexpressionNetworkBuilder(n_genes)
            
        # Multi-network encoder
        n_networks = 2 + int(use_coexpression) + int(use_ppi)
        self.network_encoder = MultiNetworkEncoder(
            n_genes, hidden_dim, n_networks,
            n_gcn_layers=2, dropout=dropout
        )
        
        # TF-Gene regulatory adjacency (set externally)
        self.register_buffer('tf_gene_adj', torch.zeros(n_genes, n_genes))
        
        # PPI encoder
        if use_ppi:
            self.ppi_encoder = PPIEncoder(20, hidden_dim)
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def set_tf_adjacency(self, tf_gene_binding: torch.Tensor, tf_indices: List[int]):
        """
        Set TF-Gene regulatory adjacency from binding matrix.
        
        Args:
            tf_gene_binding: TF binding scores (n_genes, n_tfs)
            tf_indices: Indices of TF genes
        """
        adj = torch.zeros(self.n_genes, self.n_genes, device=tf_gene_binding.device)
        
        for tf_idx, gene_idx in enumerate(tf_indices):
            if gene_idx < self.n_genes:
                # TF -> target genes
                adj[gene_idx, :] = tf_gene_binding[:, tf_idx]
                
        self.tf_gene_adj = adj
        
    def forward(
        self,
        expression: Optional[torch.Tensor] = None,
        ppi_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode genes using all networks.
        
        Args:
            expression: Gene expression for co-expression network (n_cells, n_genes)
            ppi_features: PPI features (n_genes, ppi_dim)
            
        Returns:
            Gene embeddings (n_genes, hidden_dim)
        """
        adjacencies = []
        
        # 1. TF-Gene regulatory network
        adjacencies.append(self.tf_gene_adj)
        
        # 2. Identity/baseline network
        adjacencies.append(torch.eye(self.n_genes, device=self.tf_gene_adj.device) * 0.1)
        
        # 3. Co-expression network
        if self.use_coexpression and expression is not None:
            coexpr_adj = self.coexpr_builder(expression)
            adjacencies.append(coexpr_adj)
            
        # 4. PPI network (placeholder - set adjacency externally)
        if self.use_ppi:
            # Placeholder identity - should be set with real PPI data
            adjacencies.append(torch.eye(self.n_genes, device=self.tf_gene_adj.device) * 0.05)
            
        # Multi-network encoding
        gene_embeddings = self.network_encoder(adjacencies)
        
        # Add PPI features if available
        if self.use_ppi and ppi_features is not None:
            ppi_emb = self.ppi_encoder(ppi_features)
            gene_embeddings = gene_embeddings + ppi_emb
            
        # Final projection
        gene_embeddings = self.output_proj(gene_embeddings)
        
        return gene_embeddings


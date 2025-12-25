"""
TF Regulatory Encoder for SingleCell-GRN

Encodes transcription factor regulatory networks from ChIP-Atlas data.
Handles TF-TF cascades and TF-Gene relationships.

Architecture inspired by:
- DPFunc GCN layers
- LINGER TF motif integration
- Pro2Func multi-head attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# Try to import DGL
try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    DGL_AVAILABLE = False
    dgl = None
    dglnn = None


class TFEmbedding(nn.Module):
    """
    Learnable TF embeddings with optional ChIP-seq feature initialization.
    
    Args:
        n_tfs: Number of transcription factors
        embedding_dim: Dimension of TF embeddings
        use_pretrained: Whether to initialize from ChIP-seq data
    """
    
    def __init__(
        self,
        n_tfs: int,
        embedding_dim: int = 512,
        use_pretrained: bool = False
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.embedding_dim = embedding_dim
        
        # Learnable TF embeddings
        self.embeddings = nn.Embedding(n_tfs, embedding_dim)
        
        # TF type encoding (activator/repressor/dual)
        self.tf_type_embedding = nn.Embedding(3, embedding_dim // 4)
        
        # Initialize
        nn.init.xavier_uniform_(self.embeddings.weight)
        
    def forward(
        self, 
        tf_indices: torch.Tensor,
        tf_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get TF embeddings.
        
        Args:
            tf_indices: TF indices (batch,) or (batch, n_tfs)
            tf_types: Optional TF type encoding
            
        Returns:
            TF embeddings (batch, embedding_dim)
        """
        emb = self.embeddings(tf_indices)
        
        if tf_types is not None:
            type_emb = self.tf_type_embedding(tf_types)
            # Pad to match embedding_dim and add
            padding = torch.zeros(
                *type_emb.shape[:-1], 
                self.embedding_dim - type_emb.shape[-1],
                device=type_emb.device
            )
            type_emb = torch.cat([type_emb, padding], dim=-1)
            emb = emb + type_emb
            
        return emb


class TFGeneBindingEncoder(nn.Module):
    """
    Encodes TF-Gene binding relationships from ChIP-Atlas.
    
    Uses the transcription_factor_gene_matrix.csv where:
    - Columns: TFs (1,390 TFs)
    - Rows: Genes (12,554 genes)
    - Values: Binding strength/presence
    
    Args:
        n_tfs: Number of TFs
        n_genes: Number of genes
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_tfs: int = 1390,
        n_genes: int = 12554,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # TF embeddings
        self.tf_embeddings = TFEmbedding(n_tfs, hidden_dim)
        
        # Gene embeddings  
        self.gene_embeddings = nn.Embedding(n_genes, hidden_dim)
        nn.init.xavier_uniform_(self.gene_embeddings.weight)
        
        # Binding strength encoder
        # Takes sparse binding matrix and produces dense features
        self.binding_encoder = nn.Sequential(
            nn.Linear(n_tfs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memory-efficient TF aggregation (instead of expensive cross-attention)
        # Use learned weighted sum instead of full attention
        self.tf_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.tf_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        binding_matrix: torch.Tensor,
        gene_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode TF-Gene binding relationships.
        
        Args:
            binding_matrix: TF-Gene binding matrix (n_genes, n_tfs)
            gene_indices: Optional specific gene indices to process
            
        Returns:
            Tuple of (gene_features, tf_features)
        """
        # Get all TF embeddings
        tf_indices = torch.arange(self.n_tfs, device=binding_matrix.device)
        tf_features = self.tf_embeddings(tf_indices)  # (n_tfs, hidden)
        
        # Encode binding patterns for each gene
        if gene_indices is not None:
            binding_patterns = binding_matrix[gene_indices]  # (batch, n_tfs)
        else:
            binding_patterns = binding_matrix  # (n_genes, n_tfs)
            
        # Dense binding features
        binding_features = self.binding_encoder(binding_patterns)  # (batch/n_genes, hidden)
        
        # Get gene base embeddings
        if gene_indices is not None:
            gene_base = self.gene_embeddings(gene_indices)
        else:
            gene_base = self.gene_embeddings.weight
            
        # Combine binding features with gene embeddings
        gene_features = gene_base + binding_features
        
        # Memory-efficient TF aggregation
        # Instead of cross-attention (O(n_genes * n_tfs * hidden)), 
        # use weighted sum of TF features based on binding patterns
        # This is O(n_genes * hidden) - much more memory efficient
        
        # Compute TF weights based on binding patterns (n_genes/batch, n_tfs)
        # Weight each TF by how strongly it binds
        binding_weights = torch.softmax(binding_patterns + 1e-8, dim=-1)  # (batch, n_tfs)
        
        # Aggregate TF features weighted by binding
        # tf_features: (n_tfs, hidden) -> weighted sum -> (batch, hidden)
        tf_context = torch.matmul(binding_weights, tf_features)  # (batch, hidden)
        tf_context = self.tf_proj(tf_context)
        
        # Residual connection
        gene_features = self.layer_norm(gene_features + self.dropout(tf_context))
        
        return gene_features, tf_features


class TFCascadeGCN(nn.Module):
    """
    Graph Convolutional Network for TF-TF regulatory cascades.
    
    Models relationships like: TF1 -> TF2 -> TF3 -> Gene
    Uses the fact that TFs can regulate other TFs.
    
    Architecture from DPFunc with modifications for TF networks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_dgl: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_dgl = use_dgl and DGL_AVAILABLE
        
        if self.use_dgl:
            # DGL-based GCN layers with residual connections
            self.gcn_layers = nn.ModuleList([
                dglnn.GraphConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        else:
            # Fallback: Simple message passing
            self.gcn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor = None,
        graph: 'dgl.DGLGraph' = None
    ) -> torch.Tensor:
        """
        Apply GCN on TF network.
        
        Args:
            node_features: Node features (n_nodes, hidden)
            adjacency: Adjacency matrix (n_nodes, n_nodes) for non-DGL
            graph: DGL graph for DGL mode
            
        Returns:
            Updated node features
        """
        h = node_features
        
        if self.use_dgl:
            # DGL mode - need to create graph from adjacency if not provided
            if graph is None and adjacency is not None and DGL_AVAILABLE:
                # Create DGL graph from adjacency matrix
                src, dst = adjacency.nonzero(as_tuple=True)
                graph = dgl.graph((src, dst), num_nodes=adjacency.size(0))
                graph = dgl.add_self_loop(graph)  # Add self-loops for 0-in-degree nodes
                graph = graph.to(h.device)
            
            if graph is not None:
                for i, (conv, bn) in enumerate(zip(self.gcn_layers, self.bns)):
                    h_prev = h
                    h = conv(graph, h)
                    h = bn(h)
                    h = F.gelu(h)
                    h = self.dropout(h)
                    if h.shape == h_prev.shape:
                        h = h + h_prev  # Residual
            else:
                # Fallback: simple MLP without message passing
                for i, (conv, bn) in enumerate(zip(self.gcn_layers, self.bns)):
                    h_prev = h
                    # Use the linear projection from GraphConv
                    if hasattr(conv, 'fc'):
                        h = conv.fc(h)
                    elif hasattr(conv, 'linear'):
                        h = conv.linear(h)
                    else:
                        # Skip if we can't extract the linear layer
                        pass
                    h = bn(h)
                    h = F.gelu(h)
                    h = self.dropout(h)
                    if h.shape == h_prev.shape:
                        h = h + h_prev  # Residual
        else:
            # Non-DGL mode: Simple message passing with nn.Linear
            for i, (linear, bn) in enumerate(zip(self.gcn_layers, self.bns)):
                h_prev = h
                if adjacency is not None:
                    # Normalized adjacency message passing
                    h = torch.mm(adjacency, h)
                h = linear(h)
                h = bn(h)
                h = F.gelu(h)
                h = self.dropout(h)
                if h.shape == h_prev.shape:
                    h = h + h_prev  # Residual
                    
        return h


class TFRegulatoryEncoder(nn.Module):
    """
    Complete TF Regulatory Network Encoder.
    
    Combines:
    1. TF-Gene binding encoder (from ChIP-Atlas)
    2. TF-TF cascade GCN
    3. Multi-head attention for TF importance
    
    This is the main encoder for the TF regulatory network component.
    """
    
    def __init__(
        self,
        n_tfs: int = 1390,
        n_genes: int = 12554,
        hidden_dim: int = 512,
        num_gcn_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # TF-Gene binding encoder
        self.binding_encoder = TFGeneBindingEncoder(
            n_tfs=n_tfs,
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # TF-TF cascade GCN
        self.tf_cascade_gcn = TFCascadeGCN(
            hidden_dim=hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # Memory-efficient TF importance (instead of expensive cross-attention)
        # Uses gated combination instead of full attention
        self.tf_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        binding_matrix: torch.Tensor,
        tf_tf_adjacency: Optional[torch.Tensor] = None,
        gene_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode TF regulatory network.
        
        Args:
            binding_matrix: TF-Gene binding matrix (n_genes, n_tfs)
            tf_tf_adjacency: TF-TF regulatory adjacency (n_tfs, n_tfs)
            gene_indices: Specific genes to encode
            
        Returns:
            Dict with:
                - 'gene_features': Gene embeddings with TF context
                - 'tf_features': TF embeddings after cascade
                - 'tf_importance': Global TF importance weights
        """
        # Step 1: Encode TF-Gene binding
        gene_features, tf_features = self.binding_encoder(
            binding_matrix, gene_indices
        )
        
        # Step 2: Apply TF cascade GCN
        if tf_tf_adjacency is not None:
            tf_features = self.tf_cascade_gcn(tf_features, tf_tf_adjacency)
        
        # Step 3: Memory-efficient TF importance 
        # Instead of expanding TF features to (batch, n_tfs, hidden),
        # compute a global TF importance and gate gene features
        
        # Global TF importance weights (computed once, not per gene)
        tf_importance = self.tf_gate(tf_features).squeeze(-1)  # (n_tfs,)
        tf_importance = torch.softmax(tf_importance, dim=0)  # normalize
        
        # Weighted TF summary
        tf_summary = torch.sum(tf_features * tf_importance.unsqueeze(-1), dim=0)  # (hidden,)
        tf_summary = tf_summary.unsqueeze(0).expand(gene_features.size(0), -1)  # (batch, hidden)
        
        # Step 4: Combine gene features with TF summary
        combined = torch.cat([gene_features, tf_summary], dim=-1)
        output = self.output_proj(combined)
        
        return {
            'gene_features': output,
            'tf_features': tf_features,
            'tf_importance': tf_importance  # (n_tfs,) global importance
        }
    
    def get_tf_gene_edges(
        self,
        binding_matrix: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """
        Extract TF-Gene regulatory edges above threshold.
        
        Args:
            binding_matrix: TF-Gene binding matrix
            threshold: Minimum binding score
            
        Returns:
            List of (tf_idx, gene_idx, score) tuples
        """
        edges = []
        mask = binding_matrix > threshold
        nonzero = torch.nonzero(mask)
        
        for idx in nonzero:
            gene_idx = idx[0].item()
            tf_idx = idx[1].item()
            score = binding_matrix[gene_idx, tf_idx].item()
            edges.append((tf_idx, gene_idx, score))
            
        return edges


class ChIPAtlasLoader:
    """
    Utility class to load and process ChIP-Atlas TF-Gene binding data.
    """
    
    @staticmethod
    def load_binding_matrix(
        csv_path: str,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Load TF-Gene binding matrix from CSV.
        
        Args:
            csv_path: Path to transcription_factor_gene_matrix.csv
            normalize: Whether to normalize values
            
        Returns:
            Tuple of (binding_matrix, gene_names, tf_names)
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path, index_col=0)
        
        # First row is TF names
        tf_names = list(df.columns)
        gene_names = list(df.index)
        
        # Convert to tensor
        matrix = torch.tensor(df.values, dtype=torch.float32)
        
        if normalize:
            # Min-max normalization per TF
            min_vals = matrix.min(dim=0, keepdim=True)[0]
            max_vals = matrix.max(dim=0, keepdim=True)[0]
            matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-8)
            
        return matrix, gene_names, tf_names
    
    @staticmethod
    def build_tf_tf_adjacency(
        binding_matrix: torch.Tensor,
        tf_gene_map: Dict[str, int],
        tf_names: List[str]
    ) -> torch.Tensor:
        """
        Build TF-TF adjacency from TF-Gene binding.
        
        If TF1 binds to gene(TF2), then TF1 -> TF2 edge exists.
        
        Args:
            binding_matrix: TF-Gene binding matrix
            tf_gene_map: Map from gene name to gene index
            tf_names: List of TF names
            
        Returns:
            TF-TF adjacency matrix
        """
        n_tfs = len(tf_names)
        adjacency = torch.zeros(n_tfs, n_tfs)
        
        for tf_idx, tf_name in enumerate(tf_names):
            if tf_name in tf_gene_map:
                gene_idx = tf_gene_map[tf_name]
                # TF binding to this TF's gene
                adjacency[:, tf_idx] = binding_matrix[gene_idx, :]
                
        # Normalize
        row_sum = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (row_sum + 1e-8)
        
        return adjacency


"""
Sequence Encoder for SingleCell-GRN

CNN-based encoder for DNA sequences, inspired by DeepCRE.
Extracts features from cis-regulatory elements near gene TSS.

Architecture from DeepCRE:
- 1D CNN layers with different kernel sizes
- Max pooling
- Dense layers for sequence embeddings

This module is optional - used when DNA sequence data is available.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class DNATokenizer:
    """
    Tokenize DNA sequences for CNN input.
    
    Converts ACGT to one-hot encoding.
    """
    
    NUCLEOTIDES = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    def __init__(self, max_length: int = 2000, include_n: bool = True):
        """
        Initialize tokenizer.
        
        Args:
            max_length: Maximum sequence length
            include_n: Include N (unknown) nucleotide
        """
        self.max_length = max_length
        self.n_tokens = 5 if include_n else 4
        
    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encode DNA sequence to one-hot tensor.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot tensor (n_tokens x seq_length)
        """
        seq = sequence.upper()
        
        # Pad or truncate
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        elif len(seq) < self.max_length:
            seq = seq + 'N' * (self.max_length - len(seq))
            
        # One-hot encode
        encoding = torch.zeros(self.n_tokens, self.max_length)
        
        for i, nuc in enumerate(seq):
            if nuc in self.NUCLEOTIDES:
                idx = self.NUCLEOTIDES[nuc]
                if idx < self.n_tokens:
                    encoding[idx, i] = 1.0
                    
        return encoding
        
    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """Encode batch of sequences."""
        return torch.stack([self.encode(seq) for seq in sequences])


class ConvBlock(nn.Module):
    """
    Convolutional block for sequence processing.
    
    From DeepCRE: Conv1D + BatchNorm + ReLU + MaxPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SequenceCNN(nn.Module):
    """
    CNN for DNA sequence feature extraction.
    
    Based on DeepCRE architecture:
    - Multiple convolutional layers with increasing filters
    - Global max pooling
    - Dense layers
    
    Args:
        n_tokens: Number of input tokens (4 for ACGT, 5 with N)
        seq_length: Input sequence length
        n_filters: List of filter sizes for conv layers
        kernel_sizes: Kernel sizes for each conv layer
        hidden_dim: Hidden dimension for dense layers
        output_dim: Output embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_tokens: int = 5,
        seq_length: int = 2000,
        n_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [8, 4, 4],
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.n_tokens = n_tokens
        self.seq_length = seq_length
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        
        in_channels = n_tokens
        for n_filter, kernel_size in zip(n_filters, kernel_sizes):
            self.conv_layers.append(
                ConvBlock(in_channels, n_filter, kernel_size, dropout=dropout)
            )
            in_channels = n_filter
            
        # Calculate output size after convolutions
        test_input = torch.zeros(1, n_tokens, seq_length)
        with torch.no_grad():
            for conv in self.conv_layers:
                test_input = conv(test_input)
        conv_output_size = test_input.shape[1] * test_input.shape[2]
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: One-hot encoded sequences (batch, n_tokens, seq_length)
            
        Returns:
            Sequence embeddings (batch, output_dim)
        """
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
            
        # Global features
        x = self.flatten(x)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MultiKernelCNN(nn.Module):
    """
    Multi-kernel CNN for motif detection.
    
    Uses multiple kernel sizes in parallel to capture motifs of different lengths.
    Similar to Kim (2014) CNN for NLP, adapted for DNA sequences.
    
    Args:
        n_tokens: Number of input tokens
        seq_length: Sequence length
        kernel_sizes: List of kernel sizes (e.g., [4, 8, 16] for short/medium/long motifs)
        n_filters: Filters per kernel size
        output_dim: Output dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_tokens: int = 5,
        seq_length: int = 2000,
        kernel_sizes: List[int] = [4, 8, 16, 24],
        n_filters: int = 64,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(n_tokens, n_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.bn = nn.BatchNorm1d(n_filters * len(kernel_sizes))
        self.fc = nn.Linear(n_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-kernel processing.
        
        Args:
            x: Input sequences (batch, n_tokens, seq_length)
            
        Returns:
            Sequence embeddings (batch, output_dim)
        """
        # Apply each kernel
        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))
            # Global max pooling over sequence
            out = F.adaptive_max_pool1d(out, 1).squeeze(-1)
            conv_outputs.append(out)
            
        # Concatenate
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CisRegulatoryEncoder(nn.Module):
    """
    Complete cis-regulatory element encoder.
    
    Combines:
    - Promoter sequence CNN
    - Enhancer sequence CNN (optional)
    - Distance-weighted attention
    
    From DeepCRE: Uses separate models for different regulatory regions.
    
    Args:
        promoter_length: Length of promoter region to encode
        enhancer_length: Length of enhancer region (optional)
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
        use_enhancer: Whether to use enhancer sequences
    """
    
    def __init__(
        self,
        promoter_length: int = 2000,
        enhancer_length: int = 1000,
        hidden_dim: int = 256,
        output_dim: int = 128,
        use_enhancer: bool = False,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.use_enhancer = use_enhancer
        
        # Promoter CNN
        self.promoter_cnn = MultiKernelCNN(
            n_tokens=5,
            seq_length=promoter_length,
            kernel_sizes=[4, 8, 16],
            n_filters=64,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Enhancer CNN (optional)
        if use_enhancer:
            self.enhancer_cnn = MultiKernelCNN(
                n_tokens=5,
                seq_length=enhancer_length,
                kernel_sizes=[4, 8, 16],
                n_filters=32,
                output_dim=hidden_dim // 2,
                dropout=dropout
            )
            combined_dim = hidden_dim + hidden_dim // 2
        else:
            self.enhancer_cnn = None
            combined_dim = hidden_dim
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self,
        promoter_seq: torch.Tensor,
        enhancer_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode cis-regulatory sequences.
        
        Args:
            promoter_seq: Promoter sequences (batch, n_tokens, promoter_length)
            enhancer_seq: Optional enhancer sequences
            
        Returns:
            CRE embeddings (batch, output_dim)
        """
        # Encode promoter
        promoter_emb = self.promoter_cnn(promoter_seq)
        
        if self.use_enhancer and enhancer_seq is not None:
            # Encode enhancer
            enhancer_emb = self.enhancer_cnn(enhancer_seq)
            # Concatenate
            combined = torch.cat([promoter_emb, enhancer_emb], dim=1)
        else:
            combined = promoter_emb
            
        # Project to output
        output = self.output_proj(combined)
        
        return output


class GeneSequenceEncoder(nn.Module):
    """
    Encode genes using their cis-regulatory sequences.
    
    For each gene, encodes the promoter region and optional enhancers,
    producing a sequence-based embedding that complements expression data.
    
    Args:
        n_genes: Number of genes
        promoter_length: Promoter sequence length
        hidden_dim: Hidden dimension
        embedding_dim: Final embedding dimension per gene
    """
    
    def __init__(
        self,
        n_genes: int,
        promoter_length: int = 2000,
        hidden_dim: int = 256,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.embedding_dim = embedding_dim
        
        # CRE encoder (shared across genes)
        self.cre_encoder = CisRegulatoryEncoder(
            promoter_length=promoter_length,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        )
        
        # Pre-computed gene embeddings (for efficiency)
        self.gene_embeddings = nn.Embedding(n_genes, embedding_dim)
        
        # Flag for using sequences vs pre-computed
        self.use_precomputed = True
        
    def precompute_embeddings(
        self,
        promoter_sequences: Dict[int, torch.Tensor]
    ):
        """
        Pre-compute embeddings from sequences.
        
        Args:
            promoter_sequences: Dict mapping gene_idx to promoter sequence tensor
        """
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for gene_idx, seq in promoter_sequences.items():
                if gene_idx < self.n_genes:
                    seq = seq.unsqueeze(0).to(device)
                    emb = self.cre_encoder(seq).squeeze(0)
                    self.gene_embeddings.weight.data[gene_idx] = emb
                    
        self.use_precomputed = True
        
    def forward(
        self,
        gene_indices: Optional[torch.Tensor] = None,
        promoter_sequences: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get gene sequence embeddings.
        
        Args:
            gene_indices: Gene indices for lookup (batch,) or None for all
            promoter_sequences: Optional sequences to encode on-the-fly
            
        Returns:
            Gene embeddings (batch/n_genes, embedding_dim)
        """
        if promoter_sequences is not None:
            # Encode sequences directly
            return self.cre_encoder(promoter_sequences)
        elif gene_indices is not None:
            # Lookup pre-computed
            return self.gene_embeddings(gene_indices)
        else:
            # Return all embeddings
            return self.gene_embeddings.weight


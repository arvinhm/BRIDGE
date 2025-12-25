"""
Perturbation Module for SingleCell-GRN

Implements LINGER-style perturbation analysis and GRN inference.

From LINGER (Nature Biotechnology 2024):
- Per-gene neural networks for expression prediction
- TF potential calculation from binding + expression
- In silico perturbation simulation
- GRN edge scoring

This module provides:
1. Perturbation effect prediction
2. TF influence scoring
3. GRN edge weight calculation
4. Counterfactual analysis
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union

import numpy as np


class PerGenePredictor(nn.Module):
    """
    Per-gene neural network predictor (LINGER-style).
    
    For each gene, predicts its expression from:
    - TF expression/activity
    - Regulatory element accessibility (optional)
    
    Args:
        n_tfs: Number of transcription factors
        n_res: Number of regulatory elements (optional)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_tfs: int,
        n_res: int = 0,
        hidden_dims: List[int] = [64, 16],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_tfs = n_tfs
        self.n_res = n_res
        input_dim = n_tfs + n_res
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(
        self,
        tf_features: torch.Tensor,
        re_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict gene expression.
        
        Args:
            tf_features: TF expression/activity (batch, n_tfs)
            re_features: Optional RE accessibility (batch, n_res)
            
        Returns:
            Predicted expression (batch, 1)
        """
        if re_features is not None:
            x = torch.cat([tf_features, re_features], dim=-1)
        else:
            x = tf_features
            
        return self.net(x)


class SharedPerturbationEncoder(nn.Module):
    """
    Shared encoder for perturbation features.
    
    Instead of per-gene models (expensive), uses a shared encoder
    with gene-specific output heads.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Shared hidden dimension
        n_layers: Number of hidden layers
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        
        # Shared encoder
        layers = []
        prev_dim = n_tfs
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Gene-specific output projection
        self.gene_proj = nn.Linear(hidden_dim, n_genes)
        
    def forward(self, tf_features: torch.Tensor) -> torch.Tensor:
        """
        Predict all gene expressions from TF features.
        
        Args:
            tf_features: TF activity (batch, n_tfs)
            
        Returns:
            Predicted expression for all genes (batch, n_genes)
        """
        h = self.encoder(tf_features)
        return self.gene_proj(h)


class TFInfluenceCalculator(nn.Module):
    """
    Calculate TF influence on target genes.
    
    Combines:
    - ChIP-seq binding scores (TF-gene proximity)
    - Expression correlation
    - Learnable influence weights
    
    From LINGER: TF potential = binding * expression * learnable_weight
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Hidden dimension for influence calculation
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        
        # Learnable influence weights (TF -> Gene)
        self.influence_weights = nn.Parameter(torch.randn(n_tfs, n_genes) * 0.01)
        
        # Binding mask (from ChIP-seq, can be updated)
        self.register_buffer('binding_mask', torch.ones(n_genes, n_tfs))
        
        # Influence encoder
        self.influence_encoder = nn.Sequential(
            nn.Linear(n_tfs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes)
        )
        
    def set_binding_mask(self, binding_matrix: torch.Tensor):
        """
        Set TF-gene binding mask from ChIP-seq data.
        
        Args:
            binding_matrix: Binary/continuous binding (n_genes, n_tfs)
        """
        self.binding_mask = binding_matrix.to(self.binding_mask.device)
        
    def forward(
        self,
        tf_expression: torch.Tensor,
        return_influence_matrix: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculate TF influence on gene expression.
        
        Args:
            tf_expression: TF expression (batch, n_tfs)
            return_influence_matrix: Whether to return influence matrix
            
        Returns:
            Predicted gene expression (batch, n_genes)
            Optionally: influence matrix (n_tfs, n_genes)
        """
        # Effective influence = learned weights * binding mask
        effective_influence = self.influence_weights * self.binding_mask.T  # (n_tfs, n_genes)
        
        # Gene expression = TF expression weighted by influence
        gene_expr = torch.matmul(tf_expression, effective_influence)  # (batch, n_genes)
        
        # Add non-linear component
        gene_expr = gene_expr + self.influence_encoder(tf_expression)
        
        if return_influence_matrix:
            return gene_expr, effective_influence
        return gene_expr


class PerturbationSimulator(nn.Module):
    """
    Simulate gene perturbation effects.
    
    Given a perturbation (KO/OE of a gene), predicts the effect
    on all other genes in the network.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Hidden dimension
        n_propagation_steps: Number of effect propagation steps
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int = 0,
        hidden_dim: int = 256,
        n_propagation_steps: int = 3
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs if n_tfs > 0 else n_genes
        self.n_propagation_steps = n_propagation_steps
        
        # Gene embeddings
        self.gene_embeddings = nn.Embedding(n_genes, hidden_dim)
        
        # Perturbation encoder
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for direction
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Effect propagation (like message passing)
        self.propagation_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_propagation_steps)
        ])
        
        # Effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Gene-gene adjacency (learnable or from prior)
        self.register_buffer(
            'adjacency',
            torch.eye(n_genes) * 0.1 + torch.rand(n_genes, n_genes) * 0.01
        )
        
    def set_adjacency(self, adjacency: torch.Tensor):
        """Set gene-gene regulatory adjacency."""
        self.adjacency = adjacency.to(self.adjacency.device)
        
    def forward(
        self,
        perturbed_gene_idx: torch.Tensor,
        perturbation_direction: torch.Tensor,
        return_propagated: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simulate perturbation effects.
        
        Args:
            perturbed_gene_idx: Index of perturbed gene (batch,)
            perturbation_direction: Direction (-1 for KO, +1 for OE) (batch,)
            return_propagated: Return intermediate propagation states
            
        Returns:
            Effect on all genes (batch, n_genes)
        """
        batch_size = perturbed_gene_idx.shape[0]
        
        # Get perturbed gene embedding
        gene_emb = self.gene_embeddings(perturbed_gene_idx)  # (batch, hidden)
        
        # Encode perturbation
        direction = perturbation_direction.unsqueeze(-1)  # (batch, 1)
        perturbation = self.perturbation_encoder(
            torch.cat([gene_emb, direction], dim=-1)
        )  # (batch, hidden)
        
        # Propagate effects through network
        all_gene_embs = self.gene_embeddings.weight  # (n_genes, hidden)
        
        propagated = perturbation
        propagation_states = [propagated]
        
        for layer in self.propagation_layers:
            # Effect propagates via adjacency
            # This is simplified - full version would use GNN
            propagated = layer(propagated)
            propagation_states.append(propagated)
            
        # Predict effect on each gene
        effects = []
        for gene_idx in range(self.n_genes):
            target_emb = all_gene_embs[gene_idx].unsqueeze(0).expand(batch_size, -1)
            combined = torch.cat([propagated, target_emb], dim=-1)
            effect = self.effect_predictor(combined)
            effects.append(effect)
            
        effects = torch.cat(effects, dim=-1)  # (batch, n_genes)
        
        if return_propagated:
            return effects, propagation_states
        return effects


class LINGERStyleGRN(nn.Module):
    """
    Complete LINGER-style GRN inference module.
    
    Combines:
    - TF activity estimation
    - Gene expression prediction
    - Perturbation simulation
    - GRN edge weight inference
    
    From LINGER methodology:
    1. For each gene, train a neural network to predict expression from TF features
    2. Use gradient-based attribution to determine TF->Gene edges
    3. Simulate perturbations by modifying TF activities
    4. Score edges by perturbation effect
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Hidden dimension
        use_per_gene_models: Use separate model per gene (expensive but accurate)
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 256,
        use_per_gene_models: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.use_per_gene_models = use_per_gene_models
        
        # TF influence calculator
        self.tf_influence = TFInfluenceCalculator(n_genes, n_tfs, hidden_dim)
        
        # Expression predictor
        if use_per_gene_models:
            # Create separate model per gene (LINGER original, expensive)
            self.gene_predictors = nn.ModuleList([
                PerGenePredictor(n_tfs, hidden_dims=[64, 16], dropout=dropout)
                for _ in range(n_genes)
            ])
        else:
            # Use shared encoder (efficient)
            self.shared_predictor = SharedPerturbationEncoder(
                n_genes, n_tfs, hidden_dim, dropout=dropout
            )
            
        # Perturbation simulator
        self.perturbation_sim = PerturbationSimulator(n_genes, n_tfs, hidden_dim)
        
        # GRN edge scorer
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Gene embeddings for edge scoring
        self.gene_emb_for_edges = nn.Embedding(n_genes, hidden_dim)
        
    def predict_expression(
        self,
        tf_expression: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict gene expression from TF expression.
        
        Args:
            tf_expression: TF expression (batch, n_tfs)
            
        Returns:
            Predicted gene expression (batch, n_genes)
        """
        if self.use_per_gene_models:
            predictions = []
            for predictor in self.gene_predictors:
                pred = predictor(tf_expression)
                predictions.append(pred)
            return torch.cat(predictions, dim=-1)
        else:
            return self.shared_predictor(tf_expression)
            
    def simulate_perturbation(
        self,
        tf_expression: torch.Tensor,
        perturbed_gene_idx: int,
        perturbation_direction: float = -1.0
    ) -> torch.Tensor:
        """
        Simulate perturbation of a gene.
        
        Args:
            tf_expression: Baseline TF expression (batch, n_tfs)
            perturbed_gene_idx: Gene to perturb
            perturbation_direction: -1 for KO, +1 for OE
            
        Returns:
            Predicted expression change (batch, n_genes)
        """
        batch_size = tf_expression.shape[0]
        
        # Get perturbation effect
        gene_idx = torch.tensor([perturbed_gene_idx] * batch_size, 
                               device=tf_expression.device)
        direction = torch.tensor([perturbation_direction] * batch_size,
                                device=tf_expression.device)
        
        effects = self.perturbation_sim(gene_idx, direction)
        
        return effects
        
    def compute_grn_edges(
        self,
        top_k: int = 100000
    ) -> List[Tuple[int, int, float]]:
        """
        Compute GRN edges with weights.
        
        Uses learned embeddings to score all potential TF->Gene edges.
        
        Args:
            top_k: Number of top edges to return
            
        Returns:
            List of (source_idx, target_idx, weight) tuples
        """
        device = self.gene_emb_for_edges.weight.device
        
        edges = []
        gene_embs = self.gene_emb_for_edges.weight  # (n_genes, hidden)
        
        # Score all TF -> Gene edges
        for tf_idx in range(min(self.n_tfs, self.n_genes)):
            tf_emb = gene_embs[tf_idx].unsqueeze(0)  # (1, hidden)
            
            for gene_idx in range(self.n_genes):
                if tf_idx == gene_idx:
                    continue
                    
                gene_emb = gene_embs[gene_idx].unsqueeze(0)
                combined = torch.cat([tf_emb, gene_emb], dim=-1)
                
                score = self.edge_scorer(combined).item()
                
                if score > 0.1:  # Threshold
                    edges.append((tf_idx, gene_idx, score))
                    
        # Sort by score and take top_k
        edges.sort(key=lambda x: x[2], reverse=True)
        
        return edges[:top_k]
        
    def forward(
        self,
        expression: torch.Tensor,
        tf_indices: Optional[torch.Tensor] = None,
        return_grn: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            expression: Gene expression (batch, n_genes)
            tf_indices: Optional indices of TF genes
            return_grn: Whether to return GRN edges
            
        Returns:
            Dict with predictions
        """
        # Extract TF expression
        if tf_indices is not None:
            tf_expression = expression[:, tf_indices]
        else:
            # Use first n_tfs genes as TFs
            tf_expression = expression[:, :self.n_tfs]
            
        # Predict expression
        pred_expression = self.predict_expression(tf_expression)
        
        # Get TF influence
        influenced, influence_matrix = self.tf_influence(
            tf_expression, return_influence_matrix=True
        )
        
        outputs = {
            'predicted_expression': pred_expression,
            'tf_influenced': influenced,
            'influence_matrix': influence_matrix
        }
        
        if return_grn:
            outputs['grn_edges'] = self.compute_grn_edges()
            
        return outputs


class PerturbationPredictor(nn.Module):
    """
    Unified perturbation effect predictor.
    
    Wraps LINGER-style GRN with additional features:
    - Expression reconstruction
    - Heterogeneity prediction
    - Multi-gene perturbation support
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs
        hidden_dim: Hidden dimension
        n_classes: Number of heterogeneity classes (optional)
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int = 0,
        hidden_dim: int = 256,
        n_classes: int = 0
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs if n_tfs > 0 else min(500, n_genes // 10)
        
        # Core GRN module
        self.grn = LINGERStyleGRN(
            n_genes=n_genes,
            n_tfs=self.n_tfs,
            hidden_dim=hidden_dim,
            use_per_gene_models=False
        )
        
        # Expression encoder
        self.expression_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Expression decoder
        self.expression_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes)
        )
        
        # Heterogeneity classifier (optional)
        if n_classes > 0:
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.classifier = None
            
    def forward(
        self,
        expression: torch.Tensor,
        perturbed_genes: Optional[torch.Tensor] = None,
        perturbation_directions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            expression: Gene expression (batch, n_genes)
            perturbed_genes: Optional indices of perturbed genes (batch,)
            perturbation_directions: Optional perturbation directions (batch,)
            
        Returns:
            Dict with predictions
        """
        # Encode expression
        encoded = self.expression_encoder(expression)
        
        # Reconstruct
        reconstructed = self.expression_decoder(encoded)
        
        outputs = {
            'encoded': encoded,
            'reconstructed': reconstructed
        }
        
        # GRN predictions
        grn_outputs = self.grn(expression)
        outputs.update(grn_outputs)
        
        # Perturbation effect prediction
        if perturbed_genes is not None:
            batch_size = expression.shape[0]
            effects = []
            
            for i in range(batch_size):
                gene_idx = perturbed_genes[i].item()
                direction = perturbation_directions[i].item() if perturbation_directions is not None else -1.0
                
                effect = self.grn.simulate_perturbation(
                    expression[i:i+1, :self.n_tfs],
                    gene_idx,
                    direction
                )
                effects.append(effect)
                
            outputs['perturbation_effects'] = torch.cat(effects, dim=0)
            
        # Classification
        if self.classifier is not None:
            outputs['class_logits'] = self.classifier(encoded)
            
        return outputs


def create_perturbation_predictor(
    n_genes: int,
    n_tfs: int = 0,
    hidden_dim: int = 256,
    n_classes: int = 0
) -> PerturbationPredictor:
    """
    Factory function to create perturbation predictor.
    
    Args:
        n_genes: Number of genes
        n_tfs: Number of TFs (0 = auto)
        hidden_dim: Hidden dimension
        n_classes: Number of heterogeneity classes
        
    Returns:
        PerturbationPredictor model
    """
    return PerturbationPredictor(
        n_genes=n_genes,
        n_tfs=n_tfs,
        hidden_dim=hidden_dim,
        n_classes=n_classes
    )


# Architecture Deep Dive: SingleCell-GRN v6

## Model Version: IntegratedGRNModel (train_v6.py)

This document describes the **sophisticated integrated model** architecture that combines:
- `CellStateEncoder` with transformers (from `models/cell_encoder.py`)
- `TFRegulatoryEncoder` with cascade GCN (from `models/tf_encoder.py`)
- Multi-network fusion architecture
- Context gating for cell-state-dependent TF effects
- Cofactor gate for PPI-modulated TF activity

---

## Table of Contents
1. [Conceptual Framework](#1-conceptual-framework)
2. [Complete Data Flow](#2-complete-data-flow)
3. [Data Input Specifications](#3-data-input-specifications)
4. [Model Architecture](#4-model-architecture)
5. [Forward Pass Details](#5-forward-pass-details)
6. [Loss Computation](#6-loss-computation)
7. [Gene-Specific Fine-Tuning](#7-gene-specific-fine-tuning)
8. [Output Interpretation](#8-output-interpretation)

---

## 1. Conceptual Framework

### The Regulatory Network Puzzle

The model integrates **four types of prior knowledge** to learn gene regulatory networks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRIOR KNOWLEDGE (Network Structure)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │    JASPAR        │  │   ChIP-seq       │  │    Rosetta PPI           │   │
│  │  DNA Motifs      │  │   TF Binding     │  │  Physical Interactions   │   │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────────┤   │
│  │                  │  │                  │  │                          │   │
│  │  TF Specificity  │  │  TF → Gene       │  │  Cofactor → TF           │   │
│  │  How FOCUSED a   │  │  Which TFs bind  │  │  Which proteins can      │   │
│  │  TF is on its    │  │  to which genes  │  │  ACTIVATE or INHIBIT     │   │
│  │  target genes    │  │                  │  │  TFs via physical        │   │
│  │                  │  │  TF → TF         │  │  contact                 │   │
│  │  High IC = More  │  │  TF cascade      │  │                          │   │
│  │  unique motif    │  │  hierarchy       │  │  (direction unknown)     │   │
│  │                  │  │                  │  │                          │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘   │
│           │                     │                         │                  │
│           ▼                     ▼                         ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    NETWORK PRIOR MATRIX                               │   │
│  │                                                                       │   │
│  │    tf_specificity_weights: (n_tfs,) - modulates effect magnitude     │   │
│  │    chip_mask: (n_genes, n_tfs) - allows/blocks TF→Gene effects       │   │
│  │    tf_tf_adjacency: (n_tfs, n_tfs) - TF cascade propagation          │   │
│  │    cof_to_tf_adj: (n_cof, n_tfs) - cofactor effects on TFs           │   │
│  │                                                                       │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING DATA (Observed Outcomes)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     SINGLE-CELL RNA-seq                               │   │
│  │                                                                       │   │
│  │    Each CELL TYPE = A different "state" of the regulatory network    │   │
│  │                                                                       │   │
│  │    Virtual CRISPRa/i: When TF activity differs between cell types,   │   │
│  │    we observe the EFFECT of that TF on gene expression                │   │
│  │                                                                       │   │
│  │    ┌─────────────┐  ┌─────────────┐      ┌─────────────┐              │   │
│  │    │ Cell Type A │  │ Cell Type B │ ...  │ Cell Type N │              │   │
│  │    │ TF1: HIGH   │  │ TF1: LOW    │      │ TF1: MED    │              │   │
│  │    │ TF2: LOW    │  │ TF2: HIGH   │      │ TF2: HIGH   │              │   │
│  │    │ Gene1: 5.2  │  │ Gene1: 2.1  │      │ Gene1: 3.8  │              │   │
│  │    │ Gene2: 1.0  │  │ Gene2: 8.3  │      │ Gene2: 4.2  │              │   │
│  │    └─────────────┘  └─────────────┘      └─────────────┘              │   │
│  │                                                                       │   │
│  │    If TF1 HIGH in A → Gene1 HIGH, TF1 LOW in B → Gene1 LOW           │   │
│  │    Then TF1 likely ACTIVATES Gene1!                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Training Strategy: Multi-TF Virtual CRISPRa/i

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-TF VIRTUAL PERTURBATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For each pair of cell types (Source, Target):                              │
│                                                                              │
│   1. Compute TF activity CHANGE for ALL TFs together:                        │
│      Δ_TF_activity = TF_activity[Target] - TF_activity[Source]              │
│      Shape: (n_tfs,) = (1480,)                                              │
│                                                                              │
│   2. The ACTUAL gene expression change is:                                   │
│      Δ_actual = Expression[Target] - Expression[Source]                      │
│      Shape: (n_genes,) = (26660,)                                           │
│                                                                              │
│   3. Model PREDICTS gene expression change from ALL TF changes:              │
│      Δ_predicted = Δ_TF_activity @ GRN_matrix^T                             │
│                                                                              │
│   4. LOSS = How well does Δ_predicted match Δ_actual?                        │
│                                                                              │
│   Key insight: Gene expression changes are the COMBINED effect of           │
│   multiple TFs changing simultaneously, not isolated single TFs.             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT FILES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  data/sclc/                        data/tf_binding/                         │
│  └── preprocessed.h5ad             ├── binding_matrix.npy (n_genes × n_tfs) │
│      (AnnData: cells × genes)      ├── tf_tf_adjacency.npy (n_tfs × n_tfs)  │
│                                    └── mappings.pkl                          │
│                                                                              │
│  data/tf_specificity/              data/ppi/                                │
│  ├── tf_specificity_weights.npy    ├── cof_to_tf_adj.npy (n_cof × n_tfs)   │
│  └── tf_specificity_weights_ppi.npy└── tf_tf_ppi.npy (n_tfs × n_tfs)       │
│                                    └── cofactor_gene_indices.npy            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING (in train_v6.py)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Load h5ad → Aggregate to pseudo-bulk → expression_means (n_ct, n_genes) │
│  2. Align binding matrix to expression genes                                │
│  3. Compute TF activity = expression @ binding / n_targets (LINGER)         │
│  4. Create cell-type pairs with HOLDOUT validation split                    │
│  5. Load JASPAR specificity weights and PPI matrices                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IntegratedGRNModel Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component 1: SingleCellDataEncoder (models/cell_encoder.py)        │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  • Input: Gene expression (batch, n_genes)                          │    │
│  │  • Transformer layers for cell state encoding                       │    │
│  │  • TF activity inference from expression                            │    │
│  │  • Output: cell_embedding (batch, hidden), tf_activity (batch, n_tfs)│   │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component 2: TFRegulatoryEncoder (models/tf_encoder.py)            │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  • TF-Gene binding encoder                                          │    │
│  │  • TFCascadeGCN: Graph convolution on TF-TF network                 │    │
│  │  • TF importance computation                                        │    │
│  │  • Output: tf_features (n_tfs, hidden), tf_importance (n_tfs,)      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component 3: Direct Effects Matrix                                  │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  • Shape: (n_genes, n_tfs)                                          │    │
│  │  • Initialized from ChIP binding * 0.01                             │    │
│  │  • Learnable (sparse, biologically constrained)                     │    │
│  │  • Multiplied by: tf_specificity × chip_mask × tf_importance        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component 4: Context Gating (FiLM-style modulation)                 │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  • Encodes source cell state                                        │    │
│  │  • Gamma (multiplicative gate) + Beta (additive shift)              │    │
│  │  • Allows cell-state-dependent TF effects                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component 5: Cofactor Gate                                          │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  • Uses Rosetta PPI: cof_to_tf_adj (n_cof, n_tfs)                   │    │
│  │  • Cofactor expression modulates TF potency                         │    │
│  │  • Gate: 1 + 0.2 * tanh(cofactor_influence)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT AFTER TRAINING                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GRN Matrix: (n_genes, n_tfs) = (26660, 1480)                               │
│                                                                              │
│  grn_matrix[gene, tf] > 0  →  TF ACTIVATES gene                            │
│  grn_matrix[gene, tf] < 0  →  TF REPRESSES gene                            │
│  grn_matrix[gene, tf] ≈ 0  →  No effect                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Input Specifications

### Single-Cell RNA-seq (h5ad)
```python
# Expected AnnData structure:
adata.X                  # (n_cells, n_genes) - log-normalized expression
adata.obs['celltype']    # Cell type annotations
adata.var_names          # Gene names

# Or pre-aggregated pseudo-bulk:
adata.X                  # (n_celltypes, n_genes)
adata.obs.index          # Cell type names
```

### TF Binding Data
```python
# data/tf_binding/binding_matrix.npy
# Shape: (n_genes, n_tfs)
# Values: Binary (1 = TF binds gene promoter/enhancer)

# data/tf_binding/tf_tf_adjacency.npy  
# Shape: (n_tfs, n_tfs)
# Values: Binary (1 = TF_i regulates TF_j gene)

# data/tf_binding/mappings.pkl
# Contains: gene_names, tf_names, tf_indices
```

### JASPAR Specificity
```python
# data/tf_specificity/tf_specificity_weights.npy
# Shape: (n_tfs,)
# Range: ~[0.7, 1.3]
# High = specific motif (strong effect), Low = promiscuous (diffuse effect)
```

### PPI Data
```python
# data/ppi/cof_to_tf_adj.npy
# Shape: (n_cofactors, n_tfs)
# Row-normalized adjacency matrix

# data/ppi/tf_tf_ppi.npy
# Shape: (n_tfs, n_tfs)
# TF-TF physical interactions (blended into cascade)

# data/ppi/cofactor_gene_indices.npy
# Shape: (n_cofactors,)
# Gene indices of cofactors in expression matrix
```

---

## 4. Model Architecture

### IntegratedGRNModel (train_v6.py)

```python
class IntegratedGRNModel(nn.Module):
    """
    Integrated GRN model using sophisticated encoder components.
    
    Architecture:
    1. SingleCellDataEncoder - Transformer-based cell state encoding
    2. TFRegulatoryEncoder - GCN-based TF cascade encoding
    3. Direct effects matrix - ChIP-initialized sparse effects
    4. Context gating - Cell-state-dependent modulation
    5. Cofactor gate - PPI-based TF potency modulation
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        hidden_dim: int = 512,
        num_gcn_layers: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_context_gating: bool = True,
        n_batches: int = 1
    ):
        # Component 1: Cell State Encoder
        self.cell_encoder = SingleCellDataEncoder(...)
        
        # Component 2: TF Regulatory Encoder
        self.tf_encoder = TFRegulatoryEncoder(...)
        
        # Component 3: Delta Expression Decoder
        self.delta_decoder = nn.Sequential(
            nn.Linear(n_tfs + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_genes)
        )
        
        # Component 4: Direct Effects (ChIP-initialized)
        self.direct_effects = nn.Parameter(torch.zeros(n_genes, n_tfs))
        
        # Component 5: Context Gating (optional)
        if use_context_gating:
            self.context_encoder = nn.Sequential(...)
            self.context_gate = nn.Sequential(..., nn.Sigmoid())
            self.context_shift = nn.Sequential(..., nn.Tanh())
```

### GRN Matrix Computation

```python
def get_grn_matrix(self, epoch: int = -1) -> torch.Tensor:
    """Compute the full TF→Gene effect matrix."""
    
    # 1. Get TF regulatory features (with cascade GCN)
    tf_outputs = self.tf_encoder(self.chip_mask, self.tf_tf_adjacency)
    tf_importance = tf_outputs.get('tf_importance')  # (n_tfs,)
    
    # 2. Base GRN = direct_effects * tf_specificity
    grn_matrix = self.direct_effects * self.tf_specificity_weights
    
    # 3. Apply ChIP mask (with learnable leakage)
    grn_matrix = grn_matrix * (self.chip_mask + self.chip_leakage)
    
    # 4. Weight by TF importance from cascade
    if tf_importance is not None:
        grn_matrix = grn_matrix * tf_importance.unsqueeze(0)
    
    return grn_matrix  # (n_genes, n_tfs)
```

---

## 5. Forward Pass Details

### Multi-TF Prediction (Primary Mode)

```python
def predict_delta_multi_tf(
    self,
    tf_activity_change: torch.Tensor,  # (batch, n_tfs)
    source_expression: torch.Tensor     # (batch, n_genes)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict expression CHANGE from ALL TF activity changes.
    
    Returns:
        predicted_delta: (batch, n_genes)
        grn_matrix: (n_genes, n_tfs)
    """
    
    # Step 1: Get GRN matrix
    grn_matrix = self.get_grn_matrix()  # (n_genes, n_tfs)
    
    # Step 2: Basic delta = TF_change @ GRN
    delta_basic = tf_activity_change @ grn_matrix.T  # (batch, n_genes)
    
    # Step 3: Context gating (cell-state-dependent)
    if self.use_context_gating and source_expression is not None:
        cell_outputs = self.cell_encoder(source_expression)
        cell_emb = cell_outputs['cell_embedding']
        
        combined = torch.cat([tf_activity_change, cell_emb], dim=-1)
        delta_neural = self.delta_decoder(combined)
        
        context = self.context_encoder(source_expression)
        gamma = self.context_gate(context)  # (batch, n_genes) in [0,1]
        beta = self.context_shift(context) * 0.1
        
        delta = gamma * (delta_basic + 0.1 * delta_neural) + beta
    else:
        delta = delta_basic
    
    # Step 4: Cofactor gate (PPI-based modulation)
    if self.cof_to_tf_adj is not None and source_expression is not None:
        cofactor_expr = source_expression[:, self.cofactor_gene_indices]
        cof_influence = cofactor_expr @ self.cof_to_tf_adj  # (batch, n_tfs)
        cof_gate = 1.0 + 0.2 * torch.tanh(cof_influence)
        
        tf_weighted = tf_activity_change * cof_gate
        delta_cof = tf_weighted @ grn_matrix.T
        delta = 0.5 * delta + 0.5 * delta_cof
    
    return delta, grn_matrix
```

### Effect Magnitude Breakdown

```
Final TF→Gene effect = base_effect × JASPAR_specificity × ChIP_mask 
                       × TF_importance × context_gate × cofactor_gate
                       
Where:
- base_effect:      Learned from training
- JASPAR_specificity: [0.7, 1.3] based on motif uniqueness
- ChIP_mask:        Binary + learnable leakage [0.005, 0.2]
- TF_importance:    From cascade GCN
- context_gate:     [0, 1] based on source cell state
- cofactor_gate:    [0.8, 1.2] based on cofactor expression
```

---

## 6. Loss Computation

### DeltaFocusedLoss

```python
class DeltaFocusedLoss(nn.Module):
    """
    Loss function focused on DELTA prediction.
    
    Total = MSE + ranking_weight * ranking + direction_weight * direction
            + chip_reg_weight * chip_regularization
    """
    
    def forward(self, pred_delta, actual_delta, gene_effects=None, chip_mask=None):
        # Primary: MSE on delta
        mse_loss = F.mse_loss(pred_delta, actual_delta)
        
        # Ranking: preserve relative gene ordering
        ranking_loss = self._ranking_loss(pred_delta, actual_delta)
        
        # Direction: get up/down right for significant changes
        direction_loss = self._direction_loss(pred_delta, actual_delta)
        
        # ChIP: penalize effects without binding evidence
        chip_loss = self._chip_regularization(gene_effects, chip_mask)
        
        total = mse_loss + 0.3 * ranking_loss + 0.2 * direction_loss + 0.1 * chip_loss
        
        return {'total': total, 'mse': mse_loss, ...}
```

---

## 7. Gene-Specific Fine-Tuning

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    IntegratedGRNModel (FROZEN)                              │
│                    All parameters frozen, used only for base prediction    │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼ base_prediction for target gene
┌────────────────────────────────────────────────────────────────────────────┐
│              GeneSpecificRefinementHead (TRAINABLE)                         │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                                                                      │ │
│   │   Input: tf_activity_change (batch, n_tfs)                          │ │
│   │           │                                                          │ │
│   │           ▼                                                          │ │
│   │   TF Importance Weights (learnable, softmax-normalized)             │ │
│   │           │                                                          │ │
│   │           ▼                                                          │ │
│   │   Refinement MLP:                                                    │ │
│   │   Linear(n_tfs, 256) → LayerNorm → GELU → Dropout                   │ │
│   │   Linear(256, 128) → LayerNorm → GELU → Dropout                     │ │
│   │   Linear(128, 1)                                                     │ │
│   │           │                                                          │ │
│   │           ▼                                                          │ │
│   │   refinement_prediction (batch,)                                     │ │
│   │                                                                      │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   Output = blend * base_prediction + (1 - blend) * refinement              │
│            └── learnable parameter, initialized to 0.5                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Loss Function

```python
# MSE between predicted and actual target gene delta
mse_loss = F.mse_loss(refined_prediction, actual_target_delta)

# Correlation loss: maximize Pearson r
corr_loss = 1.0 - pearson_correlation(refined_prediction, actual_target_delta)

# Sparsity regularization on TF importance
importance_reg = tf_importance.abs().mean() * 0.01

total_loss = mse_loss + 0.5 * corr_loss + importance_reg
```

### Cell-Type Holdout Validation

```python
def create_celltype_holdout_split(n_celltypes, val_fraction=0.2):
    """
    CRITICAL: Holdout entire cell types, not just random pairs.
    
    This prevents data leakage where the model memorizes cell-type patterns.
    """
    n_val = max(1, int(n_celltypes * val_fraction))
    
    val_celltypes = set(random.sample(range(n_celltypes), n_val))
    train_celltypes = set(range(n_celltypes)) - val_celltypes
    
    # Training: only pairs between training cell types
    train_pairs = [(i, j) for i in train_celltypes 
                   for j in train_celltypes if i != j]
    
    # Validation: pairs involving any held-out cell type
    val_pairs = [(i, j) for i in range(n_celltypes) 
                 for j in range(n_celltypes)
                 if i != j and (i in val_celltypes or j in val_celltypes)]
    
    return train_pairs, val_pairs
```

---

## 8. Output Interpretation

### GRN Matrix

```python
# Load trained GRN
grn = np.load('results/sclc_grn_v6/grn_matrix.npy')
# Shape: (n_genes, n_tfs) = (26660, 1480)

# Interpretation:
# grn[gene, tf] > 0  →  TF is an ACTIVATOR of gene
# grn[gene, tf] < 0  →  TF is a REPRESSOR of gene
# grn[gene, tf] ≈ 0  →  No significant effect

# Example: Find regulators of SSTR2
gene_idx = gene_names.index('SSTR2')
effects = grn[gene_idx, :]

top_activators = [(tf_names[i], effects[i]) 
                  for i in np.argsort(effects)[::-1][:10]]
top_repressors = [(tf_names[i], effects[i]) 
                  for i in np.argsort(effects)[:10]]
```

### TF Importance (Gene-Specific)

```python
# Load from fine-tuning results
importance_df = pd.read_csv('results/sclc_SSTR2/tf_importance_SSTR2.csv')

# Top TFs that predict SSTR2 expression changes
print(importance_df.head(10))
# TF_NAME     IMPORTANCE
# NEUROD1     0.0234
# ASCL1       0.0189
# ...
```

### Prediction Example

```python
# Predict expression change for a new cell state
model.eval()
with torch.no_grad():
    tf_activity_change = torch.FloatTensor(target_tf_activity - source_tf_activity)
    source_expr = torch.FloatTensor(source_expression)
    
    predicted_delta, grn = model.predict_delta_multi_tf(
        tf_activity_change.unsqueeze(0),
        source_expr.unsqueeze(0)
    )
    
    predicted_target_expression = source_expression + predicted_delta.squeeze().numpy()
```

---

## References

1. **LINGER**: Yuan & Duren, "Inferring gene regulatory networks from single-cell multiome data using atlas-scale external data", Nature Biotechnology (2024)

2. **ChIP-Atlas**: Oki et al., "ChIP-Atlas: a data-mining suite powered by full integration of public ChIP-seq data", EMBO Reports (2018)

3. **DGL**: Wang et al., "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks" (2019)

4. **JASPAR**: Castro-Mondragon et al., "JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles", Nucleic Acids Research (2022)

5. **Rosetta PPI**: Humphreys et al., "Computed structures of core eukaryotic protein complexes", Science (2021)

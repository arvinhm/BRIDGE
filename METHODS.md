# Methods: Virtual CRISPRi/CRISPRa for Gene Regulatory Network Inference

## Table of Contents
1. [Overview](#1-overview)
2. [Data Sources](#2-data-sources)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [TF Activity Computation](#4-tf-activity-computation)
5. [Virtual Perturbation Generation](#5-virtual-perturbation-generation)
6. [Model Architecture](#6-model-architecture)
7. [Training Procedure](#7-training-procedure)
8. [Loss Function](#8-loss-function)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Gene-Specific Fine-Tuning](#10-gene-specific-fine-tuning)
11. [Analysis and Visualization](#11-analysis-and-visualization)
12. [Output Files](#12-output-files)

---

## 1. Overview

### 1.1 The Problem

Traditional Gene Regulatory Network (GRN) inference faces fundamental challenges:

| Challenge | Description |
|-----------|-------------|
| **Single-cell noise** | High dropout (~80-90%), technical variation, batch effects |
| **Limited perturbation data** | Real CRISPR/RNAi experiments are expensive and low-throughput |
| **TF cascades** | TF_A → TF_B → Gene_X creates indirect effects hard to disentangle |
| **Cell state heterogeneity** | Different cell types have different regulatory programs |

### 1.2 Our Solution: Virtual CRISPRi/CRISPRa

**Core insight**: Cell-type differences are natural perturbation experiments.

```
If Cell Type A has TF_X ACTIVE   and Cell Type B has TF_X INACTIVE:
   → The difference in expression (A → B) reveals TF_X's effects
   → This is equivalent to "virtually knocking out TF_X in A"
   → We get FREE training data from observational scRNA-seq!
```

### 1.3 Key Innovations

| Traditional Approach | Our Approach |
|---------------------|--------------|
| Need real CRISPR experiments | Use cell-type differences as virtual perturbations |
| Limited to tested TFs | All ~1,500 TFs with differential activity |
| Expensive, low throughput | Thousands of virtual perturbation samples |
| Single perturbation at a time | Multi-TF: all TFs change together (biologically realistic) |

---

## 2. Data Sources

### 2.1 Single-Cell RNA-seq Data

**Input file**: `data/sclc/preprocessed.h5ad`

**Format**: AnnData object containing:
- `adata.X`: Log-normalized expression matrix (cells × genes)
- `adata.obs['celltype_key']`: Cell type annotations
- `adata.var_names`: Gene names

**Preprocessing** (via `aggregate_pseudobulk.py` or built-in):
1. Quality control filtering
2. Library size normalization
3. Log1p transformation
4. Aggregation to pseudo-bulk (per cell type)

### 2.2 ChIP-seq TF Binding Data

**Source**: ChIP-Atlas (https://chip-atlas.org)

**Input files** (in `data/tf_binding/`):

| File | Shape | Description |
|------|-------|-------------|
| `binding_matrix.npy` | (n_genes × n_tfs) | Binary/continuous TF→Gene binding |
| `tf_tf_adjacency.npy` | (n_tfs × n_tfs) | TF→TF regulatory relationships |
| `mappings.pkl` | - | Gene/TF name mappings |

### 2.3 JASPAR DNA Motif Data

**Source**: JASPAR 2024 (https://jaspar.genereg.net/)

**Purpose**: Quantify TF DNA-binding specificity

**Processing** (via `analyze_jaspar_match.py`):
- Parse JASPAR position frequency matrices (PFMs)
- Compute **Information Content (IC)** for each motif
- Handle **heterodimers** (e.g., POU2F1::SOX2)
- Match to our TFs

**Output**: `data/tf_specificity/tf_specificity_weights.npy`
- High weight (~1.3): TF with unique, specific motif
- Low weight (~0.9): Promiscuous TF with diffuse binding

### 2.4 Rosetta PPI Protein Interaction Data

**Source**: humanPPI project (AlphaFold2/RoseTTAFold2 predictions)

**Purpose**: Connect cofactors to DNA-binding TFs via physical interaction

**Processing** (via `analyze_ppi_cofactors.py` and `preprocess_ppi_rosetta.py`):
1. Identify **DNA-binding TFs** (JASPAR-matched)
2. Identify **Cofactors** (non-DNA-binding TFs)
3. Build Cofactor → TF interaction matrix
4. Build TF → TF physical interaction matrix

**Output files** (in `data/ppi/`):

| File | Shape | Description |
|------|-------|-------------|
| `cof_to_tf_adj.npy` | (n_cof × n_tfs) | Row-normalized Cofactor→TF adjacency |
| `tf_tf_ppi.npy` | (n_tfs × n_tfs) | Row-normalized TF→TF PPI |
| `cofactor_gene_indices.npy` | (n_cof,) | Gene indices of cofactors |

**Key Insight**: Cofactors affect genes INDIRECTLY through their TF partners:
```
RCOR2 (cofactor) → KDM1A (cofactor) → REST (DNA-binding) → Target Gene
```

### 2.5 Data Integration Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HOW EACH DATA SOURCE IS USED                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA SOURCE          │ MODEL COMPONENT           │ PURPOSE                  │
│  ─────────────────────┼───────────────────────────┼────────────────────────  │
│  ChIP-seq (binding)   │ chip_mask (n_genes×n_tfs) │ Allows/blocks TF→Gene    │
│  ChIP-seq (TF-TF)     │ tf_tf_adjacency (n_tfs²)  │ TF cascade propagation   │
│  JASPAR motifs        │ tf_specificity_weights    │ Modulates TF effect size │
│  Rosetta PPI:         │                           │                          │
│    - Cofactor gate    │ cof_to_tf_adj             │ Dynamic per cell-type    │
│    - TF-TF blend      │ tf_tf_ppi (20%)           │ Into cascade adjacency   │
│  Single-cell expr     │ Training target           │ What to predict          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Preprocessing Pipeline

### 3.1 Pseudo-bulk Aggregation

We aggregate single cells into cell-type profiles to reduce noise:

```python
# For each cell type c:
expression_mean[c] = mean(expression[cells_of_type_c])

# Result: (n_celltypes × n_genes) matrix
```

**Rationale**:
- Single cells are too noisy for GRN inference
- Cell-type averages capture stable regulatory states
- Differences between cell types reveal TF effects

### 3.2 Gene-TF Binding Matrix Alignment

```python
# Case-insensitive matching to expression genes
aligned_binding = align_binding_to_expression(binding_matrix, binding_genes, expression_genes)

# Typical alignment: ~75% of genes matched
```

---

## 4. TF Activity Computation

### 4.1 LINGER-style TF Activity

From LINGER (Nature Biotechnology 2024):
> "TF activity is reflected in the expression of its target genes"

**Mathematical definition**:

For each cell type `c` and TF `t`:

```
                    Σ_g (ChIP_binding[g,t] × Expression[c,g])
TF_activity[c,t] = ────────────────────────────────────────────
                         Σ_g ChIP_binding[g,t]
```

**Implementation**:
```python
def compute_tf_activity_linger(expression, binding_matrix):
    weighted_sum = expression @ binding_matrix  # (n_ct, n_tfs)
    n_targets = binding_matrix.sum(axis=0, keepdims=True).clip(min=1)
    tf_activity = weighted_sum / n_targets
    return tf_activity  # (n_ct, n_tfs)
```

This gives us TF activity for each cell type based on how highly expressed a TF's target genes are.

---

## 5. Virtual Perturbation Generation

### 5.1 Multi-TF Mode (Recommended)

**Key insight**: When comparing two cell types, ALL TFs change together. The observed expression difference is the COMBINED effect of all TF activity changes.

```python
class MultiTFDataset(Dataset):
    """Each sample is a cell-type pair with ALL TF changes."""
    
    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]
        
        return {
            # ALL TF activity changes together
            'tf_activity_change': self.tf_activity[tgt_idx] - self.tf_activity[src_idx],
            # Total expression delta
            'delta_expression': self.expression[tgt_idx] - self.expression[src_idx],
            # Source expression (for context gating)
            'source_expression': self.expression[src_idx]
        }
```

### 5.2 Cell-Type Pair Holdout (CRITICAL)

**Problem**: If we split on random samples, validation samples share cell-type pairs with training → data leakage.

**Solution**: Hold out entire cell-type PAIRS for validation.

```python
all_pairs = [(i, j) for i in range(n_ct) for j in range(n_ct) if i != j]

n_val_pairs = int(len(all_pairs) * 0.15)  # 15% for validation
random.shuffle(all_pairs)

val_pairs = set(all_pairs[:n_val_pairs])      # NEVER seen in training
train_pairs = set(all_pairs[n_val_pairs:])    # Training only
```

---

## 6. Model Architecture

### 6.1 IntegratedGRNModel (train_v6.py)

The model integrates sophisticated encoder components:

```python
class IntegratedGRNModel(nn.Module):
    """
    Components:
    1. SingleCellDataEncoder - Transformer-based cell state encoding
    2. TFRegulatoryEncoder - GCN-based TF cascade encoding
    3. Direct effects matrix - ChIP-initialized, learnable
    4. Context gating - FiLM-style cell-state modulation
    5. Cofactor gate - PPI-based TF potency modulation
    """
```

### 6.2 Key Components

#### CellStateEncoder (models/cell_encoder.py)
```
Expression → Linear → Transformer Layers → cell_embedding + tf_activity
```

#### TFRegulatoryEncoder (models/tf_encoder.py)
```
ChIP binding → TF-Gene Encoder → TFCascadeGCN → tf_features + tf_importance
              ↑
              TF-TF adjacency (from ChIP + PPI)
```

#### Direct Effects Matrix
```python
self.direct_effects = nn.Parameter(torch.zeros(n_genes, n_tfs))
# Initialized from ChIP binding × 0.01
```

#### Context Gating (FiLM-style)
```python
context = self.context_encoder(source_expression)
gamma = self.context_gate(context)  # Multiplicative gate [0, 1]
beta = self.context_shift(context)  # Additive shift
gene_effects = gamma * gene_effects + beta
```

#### Cofactor Gate
```python
cofactor_expr = source_expression[:, cofactor_gene_indices]
cof_influence = cofactor_expr @ cof_to_tf_adj  # (batch, n_tfs)
cof_gate = 1.0 + 0.2 * tanh(cof_influence)     # [0.8, 1.2]
```

### 6.3 GRN Matrix Computation

```python
def get_grn_matrix(self):
    # Base GRN
    grn = self.direct_effects * self.tf_specificity_weights
    
    # ChIP mask with learnable leakage
    grn = grn * (self.chip_mask + self.chip_leakage)
    
    # TF importance from cascade GCN
    tf_outputs = self.tf_encoder(self.chip_mask, self.tf_tf_adjacency)
    if tf_outputs.get('tf_importance') is not None:
        grn = grn * tf_outputs['tf_importance']
    
    return grn  # (n_genes, n_tfs)
```

---

## 7. Training Procedure

### 7.1 Multi-TF Forward Pass

```python
def predict_delta_multi_tf(self, tf_activity_change, source_expression):
    """
    Input:
        tf_activity_change: (batch, n_tfs) - ALL TF activity changes
        source_expression: (batch, n_genes)
    
    Output:
        predicted_delta: (batch, n_genes)
    """
    # 1. Get GRN matrix
    grn_matrix = self.get_grn_matrix()
    
    # 2. Basic: delta = TF_change @ GRN
    delta = tf_activity_change @ grn_matrix.T
    
    # 3. Context gating (if enabled)
    if self.use_context_gating:
        context = self.context_encoder(source_expression)
        gamma = self.context_gate(context)
        beta = self.context_shift(context) * 0.1
        delta = gamma * delta + beta
    
    # 4. Cofactor gate (if PPI loaded)
    if self.cof_to_tf_adj is not None:
        cof_expr = source_expression[:, self.cofactor_gene_indices]
        cof_gate = 1 + 0.2 * tanh(cof_expr @ self.cof_to_tf_adj)
        tf_weighted = tf_activity_change * cof_gate
        delta_cof = tf_weighted @ grn_matrix.T
        delta = 0.5 * delta + 0.5 * delta_cof
    
    return delta
```

### 7.2 Training Loop

```python
for epoch in range(epochs):
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Predict delta
        pred_delta, grn = model.predict_delta_multi_tf(
            batch['tf_activity_change'],
            batch['source_expression']
        )
        
        # Compute loss
        losses = loss_fn(pred_delta, batch['delta_expression'], ...)
        
        losses['total'].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validate on held-out pairs
    val_metrics = validate(val_loader)
    
    # Save best model
    if val_metrics['delta_pcc'] > best:
        save_checkpoint('best_model.pt')
```

---

## 8. Loss Function

### 8.1 DeltaFocusedLoss

```
Total Loss = L_mse + 0.3×L_ranking + 0.2×L_direction + 0.1×L_chip
```

**MSE Loss (Primary)**:
```python
L_mse = MSE(predicted_delta, actual_delta)
```

**Ranking Loss**: Preserve relative gene ordering
```python
for gene pairs (i, j):
    target_sign = sign(actual_delta[i] - actual_delta[j])
    pred_diff = pred_delta[i] - pred_delta[j]
    L_ranking = relu(-target_sign * pred_diff + margin)
```

**Direction Loss**: Get up/down correct for significant changes
```python
significant = abs(actual_delta) > threshold
pred_probs = sigmoid(pred_delta[significant] * 3)
actual_binary = (actual_delta[significant] > 0).float()
L_direction = BCE(pred_probs, actual_binary)
```

**ChIP Regularization**: Penalize effects without binding evidence
```python
no_binding = (chip_mask < 0.5)
L_chip = mean(abs(grn_matrix[no_binding]))
```

---

## 9. Evaluation Metrics

### 9.1 Delta PCC (Primary Metric)

```python
delta_pcc = mean(pearson_correlation(pred_delta[i], actual_delta[i]) for i in batch)
```

| Delta PCC | Interpretation |
|-----------|----------------|
| < 0.1 | Not learning TF effects |
| 0.1-0.3 | Starting to learn |
| 0.3-0.5 | Good TF effect prediction |
| > 0.5 | Excellent |

### 9.2 Direction Accuracy

```python
significant = abs(actual_delta) > threshold
direction_acc = mean(sign(pred[significant]) == sign(actual[significant]))
```

| Dir Acc | Interpretation |
|---------|----------------|
| 0.5 | Random |
| 0.6-0.7 | Better than random |
| > 0.8 | Excellent |

---

## 10. Gene-Specific Fine-Tuning

### 10.1 Motivation

The base GRN model learns general TF→Gene relationships. For a specific gene of interest, we can improve predictions by training a dedicated refinement head.

### 10.2 Architecture

```python
class GeneSpecificRefinementHead(nn.Module):
    """
    Sits ON TOP of frozen IntegratedGRNModel.
    
    Input: tf_activity_change (n_tfs,)
    Output: refined delta prediction for target gene (scalar)
    """
    def __init__(self, n_tfs, hidden_dim=256):
        # Learnable TF importance for this gene
        self.tf_importance = nn.Parameter(torch.zeros(n_tfs))
        
        # Refinement MLP
        self.refinement = nn.Sequential(
            nn.Linear(n_tfs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable blend between base and refinement
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, tf_activity_change, base_prediction):
        importance = F.softmax(self.tf_importance, dim=0)
        weighted_tf = tf_activity_change * importance
        refinement = self.refinement(weighted_tf).squeeze(-1)
        
        blend = torch.sigmoid(self.blend_weight)
        return blend * base_prediction + (1 - blend) * refinement
```

### 10.3 Cell-Type Holdout Validation

**Critical**: We hold out entire CELL TYPES, not just random pairs.

```python
val_celltypes = random.sample(range(n_ct), int(n_ct * 0.2))
train_celltypes = set(range(n_ct)) - set(val_celltypes)

# Training pairs: between training cell types only
train_pairs = [(i, j) for i in train_celltypes 
               for j in train_celltypes if i != j]

# Validation pairs: involve any held-out cell type
val_pairs = [(i, j) for i, j in all_pairs 
             if i in val_celltypes or j in val_celltypes]
```

### 10.4 Loss Function

```python
# MSE + Correlation
mse_loss = F.mse_loss(predicted, actual)
corr_loss = 1.0 - pearson_correlation(predicted, actual)
importance_reg = tf_importance.abs().mean() * 0.01

total = mse_loss + 0.5 * corr_loss + importance_reg
```

---

## 11. Analysis and Visualization

### 11.1 analyze_target_full.py

Generates 3 publication-quality multi-panel figures:

**Figure 1: What regulates the target?**
- TF effect sizes (activators vs repressors)
- Effect rankings
- ChIP binding evidence

**Figure 2: Why these regulators?**
- TF-TF cascade effects
- Cofactor interactions
- JASPAR specificity

**Figure 3: Calibration & controllability**
- Predicted vs actual expression
- Per-cell-type performance
- Intervention predictions

### 11.2 Automatic Integration

Fine-tuning automatically runs analysis at the end:

```python
# At end of finetune_gene.py
if run_analysis:
    from analyze_target_full import GRNAnalyzer, create_figure1, create_figure2, create_figure3
    
    analyzer = GRNAnalyzer(model_dir, gene, singlecell_path, ...)
    
    create_figure1(analyzer, analysis_dir)
    create_figure2(analyzer, analysis_dir)
    create_figure3(analyzer, analysis_dir)
    save_csvs(analyzer, analysis_dir)
```

---

## 12. Output Files

### 12.1 Base Training Outputs

```
results/sclc_grn_v6/
├── best_model.pt           # Model checkpoint
│   ├── model_state_dict    # All model weights
│   ├── config              # Architecture config
│   ├── epoch               # Best epoch
│   └── delta_pcc           # Best validation metric
├── grn_matrix.npy          # (n_genes × n_tfs) effect matrix
├── gene_names.csv          # Gene names
├── tf_names.csv            # TF names
├── celltype_names.csv      # Cell types
├── tf_activity.npy         # (n_ct × n_tfs) TF activity
├── binding_matrix.npy      # Aligned ChIP binding
├── model_config.json       # Config
└── history.json            # Training history
```

### 12.2 Fine-Tuning Outputs

```
results/sclc_SSTR2/
├── best_head.pt            # Gene-specific head checkpoint
│   ├── head_state_dict
│   ├── target_gene
│   ├── best_val_r
│   ├── tf_importance       # Learned TF importance weights
│   └── blend_weight        # Learned blend parameter
├── grn_matrix.npy
├── tf_importance_SSTR2.csv # Ranked TF importance
├── config.json
├── gene_names.csv
├── tf_names.csv
├── celltype_names.csv
├── tf_activity.npy
├── binding_matrix.npy
└── analysis/               # Auto-generated figures
    ├── figure1_regulators.png/.pdf
    ├── figure2_mechanisms.png/.pdf
    ├── figure3_calibration.png/.pdf
    ├── regulators_SSTR2.csv
    ├── interventions_SSTR2.csv
    └── summary_SSTR2.csv
```

---

## References

1. **LINGER**: Yuan & Duren, "Inferring gene regulatory networks from single-cell multiome data using atlas-scale external data", Nature Biotechnology (2024)

2. **ChIP-Atlas**: Oki et al., "ChIP-Atlas: a data-mining suite powered by full integration of public ChIP-seq data", EMBO Reports (2018)

3. **JASPAR**: Castro-Mondragon et al., "JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles", Nucleic Acids Research (2022)

4. **Rosetta PPI**: Humphreys et al., "Computed structures of core eukaryotic protein complexes", Science (2021)

5. **DGL**: Wang et al., "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks" (2019)

6. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI (2018)

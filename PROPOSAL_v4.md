# 📋 PROPOSAL v4: Virtual CRISPRi/CRISPRa for GRN Learning

## Executive Summary

Train a model that learns **TF→Gene regulatory effects** by treating **cell-type differences as natural perturbation experiments**.

---

## Scientific Foundation

### The Core Insight

> "If cell type A has TF_X **active** and cell type B has TF_X **inactive**, 
> then Expression(A) → Expression(B) is the **effect of knocking out TF_X**"

This creates **virtual CRISPR experiments** from observational scRNA-seq data.

### References Incorporated

| Source | Concept Used | How Applied |
|--------|--------------|-------------|
| **LINGER** (Nat Biotech 2024) | TF activity = weighted mean of target expression | Define TF active/inactive |
| **DPFunc/Pro2Sig** | GCN + Transformer fusion, Focal Loss | TF→TF cascade modeling |
| **DeepCRE** | CNN for cis-regulatory elements | (Future: promoter sequences) |
| **TFNET Monster** | Multi-network GCN, residual connections | Gene embedding from networks |

---

## Phase 1: TF Activity Definition (LINGER Approach)

### Scientific Rationale

From LINGER (Nature Biotech 2024):
> "TF activity is reflected in the expression of its target genes"

### Mathematical Definition

```python
# For each cell type c and TF t:
TF_activity[c, t] = Σ(ChIP_binding[g, t] × Expression[c, g]) / Σ(ChIP_binding[g, t])
                    for all genes g
```

This is **ChIP-weighted mean expression of targets**.

### TF State Classification

```
For each cell type c and TF t:

1. Compute TF_activity[c, t] using LINGER formula
2. Compute global statistics:
   - mean_activity[t] = mean(TF_activity[:, t]) across all cell types
   - std_activity[t] = std(TF_activity[:, t])
3. Classify:
   - ACTIVE:   TF_activity[c, t] > mean + 0.5*std  →  state = +1
   - INACTIVE: TF_activity[c, t] < mean - 0.5*std  →  state = -1
   - VARIABLE: otherwise                           →  state = 0

Result: TF_state_matrix[n_celltypes, n_tfs] ∈ {-1, 0, +1}
```

---

## Phase 2: Virtual Perturbation Pairs

### Creating Training Data

For each pair of cell types (source → target):

```python
for source_ct in cell_types:
    for target_ct in cell_types:
        if source_ct == target_ct:
            continue
            
        # Find TFs that differ
        tf_diff = TF_state[source_ct] - TF_state[target_ct]
        
        # TFs that are MORE active in source → "virtual knockout" to get target
        ko_tfs = where(tf_diff > 0)  
        
        # TFs that are LESS active in source → "virtual overexpression" to get target
        oe_tfs = where(tf_diff < 0)
        
        # Expression change we need to predict
        delta_expression = Expression[target_ct] - Expression[source_ct]
        
        # Create training samples
        for tf in ko_tfs:
            samples.append({
                'source_expression': Expression[source_ct],
                'perturbed_tf': tf,
                'perturbation_type': 'knockout',
                'target_delta': delta_expression,  # What should change
                'target_expression': Expression[target_ct]
            })
```

### Data Augmentation

From 16 cell types → 16 × 15 = **240 cell-type pairs**
Per pair: ~50-200 TFs differ significantly
Total: **~10,000-50,000 virtual perturbation samples**

---

## Phase 3: Model Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      TF PERTURBATION PREDICTOR                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT:                                                                  │
│  ├── Source expression: (n_genes,)                                      │
│  ├── TF to perturb: int                                                 │
│  └── Perturbation type: {knockout=-1, none=0, overexpress=+1}          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TF EFFECT MATRIX                              │   │
│  │                    (n_genes × n_tfs)                             │   │
│  │                                                                   │   │
│  │  W[g, t] = how gene g responds to changes in TF t                │   │
│  │  • Positive: gene goes UP when TF is active                      │   │
│  │  • Negative: gene goes DOWN when TF is active (repressor)        │   │
│  │  • Regularized by ChIP binding: W ⊙ ChIP_mask                    │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              PERTURBATION EFFECT COMPUTATION                     │   │
│  │                                                                   │   │
│  │  If knockout TF_t:                                                │   │
│  │    ΔExpression = -W[:, t] × TF_activity[source, t]               │   │
│  │                                                                   │   │
│  │  If overexpress TF_t:                                             │   │
│  │    ΔExpression = +W[:, t] × (1 - TF_activity[source, t])         │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              EXPRESSION DECODER (from DPFunc)                    │   │
│  │                                                                   │   │
│  │  Predicted = Source + ΔExpression                                │   │
│  │  Refined = Decoder(Predicted, Context_embedding)                 │   │
│  │                                                                   │   │
│  │  Context = [source_embedding, tf_embedding, perturbation_type]   │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  OUTPUT:                                                                 │
│  ├── Predicted target expression: (n_genes,)                            │
│  └── Predicted delta: (n_genes,)                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components (from peer-reviewed code)

#### 1. TF Effect Matrix with ChIP Regularization

```python
class TFEffectMatrix(nn.Module):
    """
    Learnable TF→Gene effect weights, regularized by ChIP-seq binding.
    
    From LINGER concept: TFs only affect genes they bind to.
    """
    def __init__(self, n_genes, n_tfs):
        super().__init__()
        # Learnable weights: sign indicates activation/repression
        self.effect_weights = nn.Parameter(torch.zeros(n_genes, n_tfs))
        # ChIP binding mask (set externally)
        self.register_buffer('chip_mask', None)
        
    def forward(self, tf_idx, perturbation_strength):
        """Get effect of perturbing a specific TF."""
        weights = self.effect_weights[:, tf_idx]
        if self.chip_mask is not None:
            weights = weights * self.chip_mask[:, tf_idx]  # Only where binding exists
        return weights * perturbation_strength
```

#### 2. Expression Encoder (from DPFunc/Pro2Sig)

```python
class ExpressionEncoder(nn.Module):
    """
    Encode cell-type expression profile.
    Architecture from Pro2Sig: Linear → BatchNorm → ReLU → Dropout
    """
    def __init__(self, n_genes, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
```

#### 3. TF-TF Cascade GCN (from DPFunc)

```python
class TFCascadeGCN(nn.Module):
    """
    Model TF→TF regulatory cascades using GCN.
    Architecture from DPFunc: GraphConv with residual connections.
    
    If TF_A regulates TF_B, and TF_B regulates Gene_X,
    then knocking out TF_A affects Gene_X indirectly.
    """
    def __init__(self, n_tfs, hidden_dim, n_layers=2):
        super().__init__()
        # From DPFunc: GCN with batch norm and residual
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'bn': nn.BatchNorm1d(hidden_dim),
                'conv': GraphConv(hidden_dim, hidden_dim),
                'dropout': nn.Dropout(0.3)
            }))
```

---

## Phase 4: Loss Function

### Multi-Component Loss (from TFNET + DPFunc)

```python
class PerturbationLoss(nn.Module):
    """
    Combined loss for perturbation prediction.
    
    Components:
    1. MSE on predicted vs actual expression
    2. Ranking loss: preserve gene ordering (from our discussion)
    3. Direction loss: did we predict up/down correctly?
    4. ChIP regularization: penalize edges without binding
    """
    def forward(self, pred, target, effect_matrix, chip_mask):
        # 1. Expression prediction loss (weighted by gene variability)
        mse_loss = F.mse_loss(pred, target)
        
        # 2. Ranking loss (from our earlier work)
        ranking_loss = self.compute_ranking_loss(pred, target)
        
        # 3. Direction accuracy loss
        pred_direction = torch.sign(pred - pred.mean())
        target_direction = torch.sign(target - target.mean())
        direction_loss = F.mse_loss(pred_direction, target_direction)
        
        # 4. ChIP regularization: L1 penalty on weights where no binding
        no_binding_mask = (chip_mask == 0)
        chip_reg = effect_matrix.abs()[no_binding_mask].mean()
        
        return mse_loss + 0.3*ranking_loss + 0.2*direction_loss + 0.1*chip_reg
```

---

## Phase 5: Training Strategy

### Stage 1: Direct TF Effects (Epochs 1-50)

Learn TF→Gene effects from single-TF perturbations:
- Input: Source expression + single TF knockout/overexpression
- Target: Actual target cell type expression
- Focus: Learn TF effect matrix

### Stage 2: TF Cascade Effects (Epochs 50-100)

Enable TF→TF→Gene cascades:
- Use GCN to propagate TF perturbations through TF-TF network
- Input: Same as Stage 1
- Model: Now includes TF cascade GCN
- Focus: Learn indirect effects

### Stage 3: Multi-TF Combinations (Optional)

Learn combined effects:
- Perturb multiple TFs simultaneously
- Predict combined effect

---

## Validation Strategy

### Hold-Out Cell Types

```python
# 16 cell types total
train_celltypes = cell_types[:12]  # 12 for training pairs
val_celltypes = cell_types[12:14]  # 2 for validation
test_celltypes = cell_types[14:]   # 2 completely held out

# Test: Can we predict held-out cell type by perturbing train cell types?
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Expression PCC** | Correlation of predicted vs actual | > 0.7 |
| **Direction Accuracy** | Did gene go up/down correctly? | > 0.75 |
| **Ranking Spearman** | Is relative ordering preserved? | > 0.6 |
| **TF Effect Recovery** | Do learned effects match known biology? | Qualitative |

---

## Expected Outputs

1. **TF Effect Matrix** (n_genes × n_tfs)
   - Interpretable: W[gene, tf] = effect strength
   - Validated against ChIP-seq

2. **Perturbation Predictions**
   - "If you KO TF_X in cell type A, it becomes like cell type B"
   
3. **Regulatory Network**
   - TF→Gene edges with learned weights
   - TF→TF cascade paths

---

## Questions Before Implementation

1. **Start with direct effects only (Phase 1-2)?** Or include TF cascades from start?

2. **Number of top TFs per cell-type pair?** (e.g., top 50 most different TFs)

3. **Include TF's own expression as additional feature?**

---

## File Structure

```
SingleCell_GRN/
├── scripts/
│   ├── train_v4.py              # New training script
│   └── prepare_perturbations.py # Generate virtual perturbation pairs
├── models/
│   ├── tf_effect_model.py       # TF effect matrix + decoder
│   └── tf_cascade_gcn.py        # TF→TF cascade modeling
└── data/
    └── perturbation_pairs/      # Generated training data
```

---

**Shall I implement this approach?**


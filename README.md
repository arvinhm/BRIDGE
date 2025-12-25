# SingleCell-GRN: Nature-Level Gene Regulatory Network Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning framework for inferring gene regulatory networks (GRN) from single-cell RNA-seq data, integrating **ChIP-seq** (TF→Gene binding), **JASPAR** (TF DNA motif specificity), and **Rosetta PPI** (protein-protein interactions).

## Key Innovation: Virtual CRISPRa/CRISPRi

Instead of requiring expensive CRISPR experiments, we treat **cell-type differences as natural perturbation experiments**:

```
Cell Type A: TF_X ACTIVE   →  Gene_Y HIGH
Cell Type B: TF_X INACTIVE →  Gene_Y LOW
                    ↓
Conclusion: TF_X likely ACTIVATES Gene_Y!
```

This provides **thousands of virtual perturbation samples** from observational scRNA-seq data.

## Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Delta_PCC** | 0.984 | Correlation of predicted vs actual expression CHANGES |
| **Direction_Acc** | 99.4% | Accuracy of up/down regulation prediction |
| **Fine-tune r** | 0.984 | Gene-specific prediction correlation (SSTR2 example) |
| **Fine-tune R²** | 0.968 | Variance explained for target gene |

## Features 🆕

| Feature | Description |
|---------|-------------|
| **Figure 4: TF Generalizability** | Identifies TFs with consistent effects across ALL cell types |
| **Multi-Gene Mode** | Fine-tune for multiple genes → find **Master Regulators** |
| **Master Regulator Discovery** | Rank TFs by their effect on entire gene signatures |
| **UMAP Visualization** | Cell-type embeddings colored by TF effects |
| **🆕 Hypothesis Builder GUI** | Interactive pathway builder to test regulatory hypotheses |
| **🆕 Hypothesis Testing** | Validate your TF→Gene pathway hypotheses against the GRN |

## Quick Reference: Three Modes

| Mode | Command | Purpose | Training? |
|------|---------|---------|-----------|
| **Single Gene** | `--gene SSTR2` | Best predictions for one gene | ✅ Yes |
| **Multi-Gene** | `--gene genes.txt` | Find master regulators | ✅ Yes |
| **Hypothesis** | `--hypothesis path.txt` | Validate your pathway | ❌ No |

```bash
# 1️⃣ Single gene fine-tuning (TRAINS the model for your gene)
python scripts/finetune_gene.py --gene SSTR2 --model results/sclc/ ...

# 2️⃣ Multi-gene / Master regulators (TRAINS for multiple genes)
python scripts/finetune_gene.py --gene genes.txt --model results/sclc/ ...

# 3️⃣ Hypothesis testing (NO training - just validates against learned GRN)
python scripts/finetune_gene.py --hypothesis pathway.txt --model results/sclc/ ...

# 🎨 Launch interactive hypothesis builder GUI
python scripts/hypothesis_builder.py
```

---

## When to Use Each Mode

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DECISION TREE: Which mode should I use?                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  "I want to PREDICT gene expression"                                    │
│       │                                                                  │
│       ├── For ONE gene → --gene SSTR2 (Single Gene Fine-tune)           │
│       │                                                                  │
│       └── For MULTIPLE genes → --gene genes.txt (Master Regulators)     │
│                                                                          │
│  "I have a HYPOTHESIS to test"                                          │
│       │                                                                  │
│       └── --hypothesis pathway.txt (Validation only, NO training)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters:

| Goal | Mode | Reason |
|------|------|--------|
| Find TFs that regulate SSTR2 | `--gene SSTR2` | Fine-tune focuses model on your gene |
| Find pan-NE master regulators | `--gene NE_markers.txt` | Identifies TFs affecting ALL genes |
| Test "ASCL1 → SSTR2" | `--hypothesis` | Unbiased check against learned GRN |

---

## Hypothesis Pathway Builder 🧬

A beautiful interactive GUI to build and test regulatory hypotheses.

### Launching the GUI

```bash
# Option 1: Direct launch
python scripts/hypothesis_builder.py

# Option 2: With predefined structure
python scripts/hypothesis_builder.py --tfs 3 --genes 2

# Option 3: Via finetune script
python scripts/finetune_gene.py --hypothesis gui \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --output results/hypothesis/
```

### GUI Features

| Feature | How To | Description |
|---------|--------|-------------|
| **Add TF** | Click "➕ Add TF" or toolbar | Creates a blue square (■) |
| **Add Gene** | Click "➕ Add Gene" or toolbar | Creates a pink circle (●) |
| **Rename** | Double-click node | Enter real gene symbol |
| **Move** | Drag node | Reposition on canvas |
| **Activator** | Press `A`, click source, click target | Green arrow (→) |
| **Repressor** | Press `R`, click source, click target | Red bar (⊣) |
| **Delete** | Right-click → Delete | Remove node or connection |
| **Set Output** | Right-click → Set as Output | Mark as prediction target (gold) |
| **Save** | File → Save | JSON format for later editing |
| **Export** | Click "🧪 Test Hypothesis" | TXT format for testing |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Activator mode |
| `R` | Repressor mode |
| `Esc` | Select mode (exit connection mode) |
| `Delete` | Delete selected node |
| `Ctrl+S` | Save hypothesis |
| `Ctrl+O` | Open hypothesis |
| `Ctrl+N` | New hypothesis |

### Visual Legend

```
■ Blue Square = Transcription Factor (TF)
● Pink Circle = Gene
● Gold Circle = Output Gene (prediction target)

→ Green Arrow = Activator (TF increases gene expression)
⊣ Red Bar     = Repressor (TF decreases gene expression)
```

---

## Hypothesis File Format

After exporting from the GUI (or creating manually):

```bash
# hypothesis.txt
# Format: SOURCE -> TARGET (activator) or SOURCE -| TARGET (repressor)

# NODES (for reference)
# TF: ASCL1
# TF: REST
# TF: YAP1
# GENE: SSTR2 [OUTPUT]
# GENE: CHGB [OUTPUT]

# CONNECTIONS
ASCL1 -> SSTR2      # ASCL1 activates SSTR2
REST -| SSTR2       # REST represses SSTR2
YAP1 -> ASCL1       # YAP1 activates ASCL1
YAP1 -| REST        # YAP1 represses REST
ASCL1 -> YAP1       # Bidirectional loop

# OUTPUT GENES (what you're predicting)
OUTPUT: SSTR2
OUTPUT: CHGB
```

### Syntax Rules:
- `->` = Activator (TF increases target expression)
- `-|` = Repressor (TF decreases target expression)
- Lines starting with `#` are comments
- `OUTPUT:` declares prediction targets (optional)

---

## Testing Your Hypothesis

### Run the Test

```bash
python scripts/finetune_gene.py \
    --hypothesis hypothesis.txt \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/hypothesis_test/
```

**Note:** `--gene` is NOT required when `--hypothesis` is provided!

### Output Files

```
results/hypothesis_test/
├── hypothesis_results.csv    # Edge-by-edge analysis
└── hypothesis_test.png       # 4-panel visualization
```

### Interpreting Results

```
============================================================
HYPOTHESIS TEST RESULTS
============================================================

Overall: 5/7 edges SUPPORTED (71.4%)

Edge-by-Edge Analysis:
------------------------------------------------------------
  ASCL1 → SSTR2
    Effect: +0.1234 (percentile: 89.2%)
    ✓ SUPPORTED

  REST ⊣ SSTR2
    Effect: -0.0891 (percentile: 76.4%)
    ✓ SUPPORTED

  YAP1 → SSTR2
    Effect: -0.0567 (percentile: 55.1%)
    ✗ NOT SUPPORTED  ← Data suggests YAP1 REPRESSES, not activates!
```

### Result Interpretation

| Symbol | Meaning | What It Means |
|--------|---------|---------------|
| ✓ SUPPORTED | GRN agrees with your hypothesis | Data supports this relationship |
| ✗ NOT SUPPORTED | GRN contradicts your hypothesis | Data suggests opposite effect |
| ⚠ Could not find | Node not in GRN | Gene isn't a TF, or not in dataset |
| Percentile | Effect strength rank | Higher = stronger, more reliable |

### What "NOT SUPPORTED" Means

**Does NOT mean you're wrong!** It means:
1. In THIS dataset, the relationship appears different
2. Could be context-dependent (tissue/disease-specific)
3. Could be indirect effects (TF→X→Gene instead of TF→Gene)
4. Could reflect regulatory complexity (activator in some contexts, repressor in others)

### Confidence Levels

| Percentile | Confidence | Interpretation |
|------------|------------|----------------|
| > 95% | 🟢 High | Strong effect, very confident |
| 75-95% | 🟡 Medium | Moderate effect, reasonably confident |
| 50-75% | 🟠 Low | Weak effect, less confident |
| < 50% | 🔴 Very Low | Near-zero effect, likely noise |

---

## Complete Workflow Example

### Goal: Study SCLC Neuroendocrine Signaling

```bash
# Step 1: Train the base GRN model (once)
python scripts/train_v6.py \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/sclc/ \
    --epochs 50

# Step 2: Fine-tune for SSTR2 predictions
python scripts/finetune_gene.py \
    --gene SSTR2 \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/sclc_SSTR2/

# Step 3: Find master regulators for NE signature
echo -e "SSTR2\nASCL1\nNEUROD1\nCHGA\nSYP\nDLL3" > NE_genes.txt

python scripts/finetune_gene.py \
    --gene NE_genes.txt \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/sclc_NE_master/

# Step 4: Build and test a hypothesis
python scripts/hypothesis_builder.py  # Create in GUI, save as my_hypothesis.txt

python scripts/finetune_gene.py \
    --hypothesis my_hypothesis.txt \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/hypothesis_test/
```

---

## Complete Workflow

### Prerequisites

```bash
# Create environment
conda create -n grn python=3.10
conda activate grn

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scanpy anndata numpy pandas scipy matplotlib seaborn tqdm dgl
```

### Step 1: Data Preparation

#### 1.1 Single-Cell RNA-seq Data

Prepare an AnnData object with:
- `adata.X`: Log-normalized expression (cells × genes)
- `adata.obs['celltype']`: Cell type annotations
- `adata.var_names`: Gene symbols

```python
import scanpy as sc

# Load and preprocess
adata = sc.read_h5ad('raw_data.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Aggregate to pseudo-bulk (optional, recommended for noisy data)
# The training script can also do this automatically
adata_pseudobulk = aggregate_to_celltype(adata, 'celltype')
adata_pseudobulk.write('data/sclc/preprocessed.h5ad')
```

#### 1.2 TF Binding Data (ChIP-Atlas)

Your ChIP-Atlas data should be processed into:

```
data/tf_binding/
├── binding_matrix.npy      # (n_genes, n_tfs) - TF→Gene binding
├── tf_tf_adjacency.npy     # (n_tfs, n_tfs) - TF→TF regulation
└── mappings.pkl            # Gene/TF name mappings
```

**Source:** ChIP-Atlas (https://chip-atlas.org) with your `TF_pathways/` processing.

### Step 2: Preprocessing

```bash
cd SingleCell_GRN

# 2a. JASPAR Motif Specificity
# Computes how "specific" each TF's DNA motif is
python scripts/analyze_jaspar_match.py
# Output: data/tf_specificity/tf_specificity_weights.npy

# 2b. PPI Cofactor Analysis
# Identifies cofactor-TF relationships from Rosetta predictions
python scripts/analyze_ppi_cofactors.py
# Output: data/tf_specificity/tf_specificity_weights_ppi.npy

# 2c. PPI Matrix Generation
# Creates cofactor→TF and TF→TF PPI adjacency matrices
python scripts/preprocess_ppi_rosetta.py
# Output: data/ppi/cof_to_tf_adj.npy, data/ppi/tf_tf_ppi.npy
```

### Step 3: Input Validation (IMPORTANT!)

```bash
# Validates ALL input files before training
python scripts/validate_all_inputs.py
```

This checks:
- TF binding matrix dimensions
- PPI matrix alignment
- JASPAR weight ranges
- Single-cell data structure
- Known TF presence (NEUROD1, REST, etc.)

### Step 4: Training

```bash
python scripts/train_v6.py \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --celltype-key sclc_subcluster \
    --output results/sclc_grn_v6/ \
    --ppi data/ppi/ \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --val-every 5
```

#### Key Arguments

| Argument | Description |
|----------|-------------|
| `--singlecell` | Path to preprocessed.h5ad |
| `--tf-binding` | Directory with ChIP-Atlas data |
| `--celltype-key` | Column name in adata.obs for cell types |
| `--ppi` | Directory with Rosetta PPI matrices |
| `--epochs` | Number of training epochs (100-300 recommended) |
| `--val-every` | Validate every N epochs |

#### What to Watch

```
Epoch  10: loss=0.494 delta=0.938 dir_acc=0.983
  [VAL] loss=1.19 | Delta=0.939 (LEARNING!) Dir=0.989
  >> New best! Delta_PCC=0.9393
```

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Delta_PCC | > 0.35 | Model learning TF effects |
| Dir_Acc | > 65% | Correct up/down predictions |
| Val_Loss | ↓ | Watch for overfitting |

### Step 5: Gene-Specific Fine-Tuning

For genes of particular interest, fine-tune to maximize prediction accuracy:

#### 5A. Single Gene Mode

```bash
python scripts/finetune_gene.py \
    --gene SSTR2 \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/sclc_SSTR2/ \
    --epochs 100 \
    --batch-size 32 \
    --hidden-dim 256
```

**Key Features:**
- Uses **same architecture** as base model (IntegratedGRNModel)
- Adds a **gene-specific refinement head**
- **Freezes base model** - only trains refinement
- **Cell-type holdout** - prevents data leakage
- **Automatic analysis** - generates 4 figures at the end

#### 5B. Multi-Gene Mode (Master Regulator Discovery) 🆕

For discovering **master regulators** that affect multiple genes of interest:

```bash
# Create a text file with target genes (one per line)
cat > genes_of_interest.txt << EOF
SSTR2
ASCL1
NEUROD1
DLL3
CHGA
SYP
# Comments start with #
EOF

# Run multi-gene fine-tuning
python scripts/finetune_gene.py \
    --gene genes_of_interest.txt \
    --model results/sclc/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --tf-binding data/tf_binding/ \
    --ppi data/ppi/ \
    --output results/sclc_master_regulators/ \
    --epochs 100
```

**Multi-Gene Outputs:**

| File | Description |
|------|-------------|
| `master_regulators.csv` | Ranked TFs affecting **ALL** target genes |
| `tf_importance_<GENE>.csv` | Per-gene TF importance for each target |
| `figure5_master_regulators.png` | Master regulator overview (10 panels) |
| `figure6_per_gene_regulators.png` | Per-gene top TF regulators |
| `target_genes.csv` | List of target genes analyzed |

**Use Cases:**
- 🧬 Finding **pan-pathway regulators** (e.g., neuroendocrine master TFs)
- 🎯 Identifying **therapeutic targets** affecting multiple disease genes
- 📊 Comparing TF effects across a gene signature

### Step 6: Analysis & Interpretation

Analysis runs automatically after fine-tuning, or manually:

```bash
python scripts/analyze_target_full.py \
    --gene SSTR2 \
    --model results/sclc_grn_v6/ \
    --singlecell data/sclc/preprocessed.h5ad \
    --ppi data/ppi/ \
    --output analysis/SSTR2/
```

#### Generated Outputs

```
analysis/SSTR2/
├── figure1_regulators.png       # What regulates SSTR2?
├── figure2_mechanisms.png       # Why these regulators?
├── figure3_calibration.png      # Prediction validation
├── figure4_generalizability.png # TF generalizability across cell types 🆕
├── regulators_SSTR2.csv         # All 1480 TFs ranked
├── interventions_SSTR2.csv      # Recommended experiments
├── tf_generalizability_SSTR2.csv # TF consistency scores 🆕
└── summary_SSTR2.csv            # Key statistics
```

#### Figure 4: TF Generalizability Analysis 🆕

| Panel | Description |
|-------|-------------|
| **A-B** | UMAP/PCA of cell types colored by top activator/repressor effects |
| **C** | Bar chart: Generalizability scores for top 20 TFs |
| **D** | Scatter: Effect size vs Generalizability |
| **E** | Histogram: Consistency distribution across all TFs |
| **F** | Heatmap: Top TF effects across cell types (Z-scored) |
| **G** | Pie chart: Universal vs Context-specific TF classification |
| **H** | UMAP colored by target gene expression |
| **I** | Bar chart: Best therapeutic targets (effect × generalizability) |
| **J** | Summary statistics and recommendations |

**Why Generalizability Matters:**
- A TF with **high effect but low generalizability** may only work in specific cell types
- A TF with **high generalizability** works consistently across patient heterogeneity
- **Best therapeutic targets** = High effect × High generalizability!

---

## Data Integration Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ChIP-Atlas           JASPAR             Rosetta PPI                    │
│  TF→Gene binding      TF motif           Cofactor→TF                    │
│  TF→TF cascade        specificity        TF→TF physical                 │
│       │                   │                   │                         │
│       ▼                   ▼                   ▼                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │               IntegratedGRNModel (train_v6.py)                │       │
│  │                                                               │       │
│  │  • SingleCellDataEncoder (Transformer)                       │       │
│  │  • TFRegulatoryEncoder (Cascade GCN)                         │       │
│  │  • Context Gating (FiLM-style modulation)                    │       │
│  │  • Cofactor Gate (PPI-based TF potency)                      │       │
│  │  • Multi-TF Training (all TFs contribute)                    │       │
│  │                                                               │       │
│  └───────────────────────────┬──────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Single-Cell RNA-seq (Training Target)            │       │
│  │                                                               │       │
│  │  Cell-type differences = Virtual CRISPRa/i experiments       │       │
│  │  Δ(TF_activity) → Δ(Gene_expression)                         │       │
│  │                                                               │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Output Interpretation

### GRN Matrix

```python
import numpy as np
import pandas as pd

# Load
grn = np.load('results/sclc_grn_v6/grn_matrix.npy')
genes = pd.read_csv('results/sclc_grn_v6/gene_names.csv', header=None)[0].tolist()
tfs = pd.read_csv('results/sclc_grn_v6/tf_names.csv', header=None)[0].tolist()

# Find SSTR2 regulators
gene_idx = genes.index('SSTR2')
effects = grn[gene_idx, :]

# Top activators (positive effects)
top_act_idx = np.argsort(effects)[::-1][:10]
for i in top_act_idx:
    print(f"{tfs[i]}: +{effects[i]:.4f}")

# Top repressors (negative effects)
top_rep_idx = np.argsort(effects)[:10]
for i in top_rep_idx:
    print(f"{tfs[i]}: {effects[i]:.4f}")
```

### Interpretation

| Effect Sign | Interpretation | Experiment |
|-------------|----------------|------------|
| `grn[gene, tf] > 0` | TF **activates** gene | Overexpress TF → gene ↑ |
| `grn[gene, tf] < 0` | TF **represses** gene | Knockout TF → gene ↑ |
| `grn[gene, tf] ≈ 0` | No significant effect | Skip |

---

## Troubleshooting

### Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Delta_PCC stuck at 0 | Data loading issue | Check h5ad structure |
| Delta_PCC not improving | LR too low | Increase to 0.001 |
| GPU OOM | Batch too large | Reduce to 128 |
| NaN in training | LR too high | Reduce to 0.0001 |

### Fine-Tuning Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| val_r > 0.99 immediately | Data leakage | Check cell-type holdout |
| val_r < 0.1 | Gene not variable | Check expression variance |
| Empty dataloader | Batch > dataset | Reduce batch size |

### Common Errors

```bash
# FileNotFoundError: best_model.pt
# → Training didn't complete. Re-run train_v6.py

# KeyError: 'sclc_subcluster'
# → Use correct --celltype-key from your adata.obs

# RuntimeError: CUDA out of memory
# → Reduce --batch-size or use CPU with --device cpu
```

---

## Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for:
- Complete data flow diagrams
- Model component specifications
- GRN matrix computation formula
- Fine-tuning architecture

See [METHODS.md](METHODS.md) for:
- Mathematical formulations
- LINGER TF activity computation
- Loss function components
- Validation strategy

---

## Citation

```bibtex
@software{SingleCell-GRN,
  title = {SingleCell-GRN: Gene Regulatory Network Inference using Virtual Perturbation Learning},
  year = {2024},
  url = {https://github.com/xxx/SingleCell-GRN}
}
```

## References

1. **LINGER**: Yuan & Duren, "Inferring gene regulatory networks from single-cell multiome data", *Nature Biotechnology* (2024)
2. **ChIP-Atlas**: Oki et al., *EMBO Reports* (2018)
3. **JASPAR**: Castro-Mondragon et al., *NAR* (2022)
4. **Rosetta PPI**: Humphreys et al., *Science* (2021)
5. **DGL**: Wang et al., *Deep Graph Library* (2019)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

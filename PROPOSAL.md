# SingleCell-GRN: Integrated Gene Regulatory Network Inference for Single-Cell Data

## Executive Summary

This project integrates four state-of-the-art systems into a unified "Monster GRN Model" for predicting gene regulatory networks from single-cell RNA-seq data:

1. **LINGER** (Nature Biotechnology 2024): Neural network-based GRN inference with TF motif matching
2. **DeepCRE**: CNN-based cis-regulatory element prediction from DNA sequences
3. **Pro2Func/DPFunc**: Multi-modal deep learning with GCN, attention, and protein features
4. **TFNET Monster Model**: Gene expression change prediction with GenSAN, FunDNN, cross-perturbation

## Data Sources

### Input Data
1. **Single-Cell RNA-seq** (user provided):
   - Barcodes × Genes expression matrix
   - Metadata (batch, patient, cell type annotations)
   
2. **TF-Gene ChIP-Atlas Binding** (provided in TF_pathways):
   - `transcription_factor_gene_matrix.csv`: 12,554 genes × 1,390 TFs
   - Per-TF pathway analysis files
   - ChIP-seq achievement data

3. **Gene Networks** (from TFNET processed_data):
   - STRING protein-protein interactions
   - CORUM protein complexes
   - Pathway databases (KEGG, Reactome)
   - Disease associations (OMIM)

### Output
- **Gene Regulatory Network**: TF → Target gene edges with confidence scores
- **Perturbation Predictions**: Given a target gene, predict:
  - Which TFs/genes to manipulate
  - Direction (up/down regulate)
  - Confidence and pathway context

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SingleCell-GRN Monster Model                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │                        INPUT LAYER                                    │   ║
║  ├─────────────────────────┬──────────────────────┬────────────────────┤   ║
║  │   Single-Cell Matrix    │   TF Binding Matrix  │   Gene Networks    │   ║
║  │   (barcodes × genes)    │   (TF × genes)       │   (STRING, etc)    │   ║
║  └───────────┬─────────────┴──────────┬───────────┴────────┬───────────┘   ║
║              │                        │                    │                ║
║              ▼                        ▼                    ▼                ║
║  ┌───────────────────┐    ┌────────────────────┐    ┌───────────────────┐  ║
║  │  LINGER Cell      │    │  TF Regulatory     │    │  DPFunc Network   │  ║
║  │  State Encoder    │    │  Graph Encoder     │    │  GCN Encoder      │  ║
║  │  (Heterogeneity)  │    │  (ChIP-Atlas)      │    │  (Multi-network)  │  ║
║  └─────────┬─────────┘    └─────────┬──────────┘    └─────────┬─────────┘  ║
║            │                        │                         │             ║
║            └────────────────────────┼─────────────────────────┘             ║
║                                     ▼                                        ║
║            ┌────────────────────────────────────────────────────────────┐   ║
║            │               TRANSFORMER FUSION BLOCK                      │   ║
║            │   (Multi-head Cross-Attention between Cell, TF, Network)   │   ║
║            └────────────────────────────┬───────────────────────────────┘   ║
║                                         │                                    ║
║                                         ▼                                    ║
║            ┌────────────────────────────────────────────────────────────┐   ║
║            │                   GRN INFERENCE HEAD                        │   ║
║            │                                                             │   ║
║            │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   ║
║            │  │ TF Activity │  │  Edge Pred  │  │  Pathway    │         │   ║
║            │  │  Predictor  │  │  (TF→Gene)  │  │  Enrichment │         │   ║
║            │  └─────────────┘  └─────────────┘  └─────────────┘         │   ║
║            └────────────────────────────┬───────────────────────────────┘   ║
║                                         │                                    ║
║                                         ▼                                    ║
║            ┌────────────────────────────────────────────────────────────┐   ║
║            │              PERTURBATION SIMULATOR                         │   ║
║            │   (LINGER-style: Predict expression after TF KO/OE)        │   ║
║            └────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Component Details

### 1. Cell State Encoder (from LINGER)
- Input: Single-cell expression matrix (barcodes × genes)
- Architecture: Per-gene neural networks with TF/RE inputs
- Output: Cell embeddings capturing heterogeneity

```python
class CellStateEncoder(nn.Module):
    """LINGER-inspired cell state encoder"""
    def __init__(self, n_genes, n_tfs, hidden_dim=512):
        # Per-gene model: predicts gene expression from TF activities
        # Uses Shapley values for feature importance (GRN inference)
```

### 2. TF Regulatory Graph Encoder (NEW - uses ChIP-Atlas)
- Input: TF × Gene binding matrix from ChIP-Atlas
- Encodes TF-TF cascades: TF1 → TF2 → TF3 → Gene
- Uses message passing to propagate regulatory signals

```python
class TFRegulatoryEncoder(nn.Module):
    """Encodes TF regulatory network from ChIP-Atlas data"""
    def __init__(self, n_tfs, n_genes, hidden_dim=512):
        # TF embedding layer
        # GCN layers for TF-TF and TF-Gene propagation
        # Attention for regulatory cascade weighting
```

### 3. Multi-Network GCN Encoder (from Pro2Func/DPFunc)
- Processes 7 gene networks in parallel
- Uses DGL GraphConv with residual connections
- Transformer fusion across networks

```python
class MultiNetworkEncoder(nn.Module):
    """DPFunc-style multi-network encoder"""
    # STRING, CORUM, Pathway, Disease, Drug, Chrom, Protein
    # Parallel GCNs → Cross-network attention → Fused embeddings
```

### 4. Transformer Fusion Block (from Pro2Func)
- Multi-head cross-attention between:
  - Cell state embeddings
  - TF regulatory embeddings  
  - Network-derived gene embeddings
- Produces unified gene representations

### 5. GRN Inference Head
- **TF Activity**: Predict TF activity scores per cell
- **Edge Prediction**: Predict TF → Gene regulatory edges
- **Pathway Enrichment**: Map to KEGG/Reactome pathways

### 6. Perturbation Simulator (from LINGER)
- Given a TF knockout/overexpression:
- Predict changes in all target gene expressions
- Simulate cell state transitions

## Training Strategy

### Phase 1: Pre-training on ChIP-Atlas (Self-supervised)
- Mask random TF-Gene edges
- Predict masked edges from network structure
- Learn TF regulatory grammar

### Phase 2: Single-Cell Fine-tuning (Supervised)
- Input: Single-cell data + batch/patient labels
- Task 1: Predict batch/patient from expression (heterogeneity)
- Task 2: Predict TF activity from expression
- Task 3: Reconstruct expression from TF activity (GRN)

### Phase 3: Perturbation Training (if data available)
- Use actual perturbation data (Perturb-seq, CRISPR screens)
- Train perturbation simulator

## Query Interface

```python
# Example usage
model = SingleCellGRNModel.load('trained_model.pt')

# Load single-cell data
adata = sc.read_h5ad('cancer_cells.h5ad')

# Infer GRN for cell population
grn = model.infer_grn(adata, cell_type='cancer')

# Query: How to upregulate SSTR2?
strategies = model.find_regulators(
    target_gene='SSTR2',
    direction='increase',
    top_k=20
)

# Output:
# 1. Upregulate NOTCH1 (confidence: 0.92, direct: True)
# 2. Downregulate REST (confidence: 0.87, via pathway: REST → NRSE → SSTR2)
# 3. Pathway: Decrease Notch signaling (affects: SSTR2, SST, GHRL)

# Simulate perturbation
predicted_expression = model.simulate_perturbation(
    adata,
    perturbations={'REST': 'KO', 'NOTCH1': 'OE'}
)
```

## File Structure

```
SingleCell_GRN/
├── README.md
├── PROPOSAL.md (this file)
├── requirements.txt
│
├── data/
│   ├── tf_binding/              # ChIP-Atlas TF-gene matrices
│   │   ├── transcription_factor_gene_matrix.csv
│   │   └── pathway_annotations/
│   ├── gene_networks/           # Copied from TFNET
│   │   ├── STRING.txt
│   │   ├── CORUM.txt
│   │   └── ...
│   └── processed/               # Processed embeddings
│
├── models/
│   ├── __init__.py              # Module exports
│   ├── cell_encoder.py          # LINGER-based cell state encoder
│   ├── tf_encoder.py            # ChIP-Atlas TF regulatory encoder
│   ├── multi_network.py         # DPFunc multi-network GCN
│   ├── sequence_encoder.py      # DeepCRE-style DNA sequence encoder
│   └── monster_grn.py           # Complete integrated model (includes fusion, heads)
│
├── modules/                      # Reusable neural network components
│   ├── __init__.py
│   └── perturbation.py          # LINGER-style perturbation analysis
│
├── scripts/
│   ├── preprocess_tf_matrix.py  # Process ChIP-Atlas TF binding data
│   ├── preprocess_singlecell.py # Preprocess single-cell RNA-seq
│   ├── train.py                 # Main training script
│   ├── query_target.py          # Query for gene regulators
│   ├── visualize_grn.py         # Generate visualizations
│   └── run_pipeline.py          # Complete end-to-end pipeline
│
└── configs/
    └── default.yaml             # Default configuration
```

## Key Innovations

1. **Unified TF Regulatory Graph**: First to integrate ChIP-Atlas binding data with single-cell GRN inference

2. **Multi-level Attention**: Cell × TF × Network cross-attention for comprehensive feature fusion

3. **Hierarchical TF Cascades**: Model TF1 → TF2 → TF3 → Gene regulatory cascades

4. **Perturbation-aware Training**: Learn from actual perturbation effects

5. **Pathway Context**: Map predictions to biological pathways for interpretability

## Expected Performance

Based on component benchmarks:
- LINGER: AUC 0.75-0.85 for GRN edges
- Pro2Func: F-max 0.77 for function prediction
- TFNET Monster: PCC 0.85+ for expression prediction

Target: **GRN edge AUC > 0.80, TF activity correlation > 0.75**

## Timeline

1. **Week 1**: Project setup, data preprocessing, architecture implementation
2. **Week 2**: Pre-training on ChIP-Atlas
3. **Week 3**: Single-cell fine-tuning
4. **Week 4**: Evaluation, optimization, documentation

## References

1. Yuan & Duren. "Inferring gene regulatory networks from single-cell multiome data using atlas-scale external data." Nature Biotechnology (2024)
2. DeepCRE: Deep learning the cis-regulatory code for gene expression
3. DPFunc/Pro2Func: Multi-modal deep learning for protein function prediction
4. TFNET: Transcription factor network for gene expression prediction


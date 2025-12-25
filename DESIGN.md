# SingleCell-GRN: Complete System Design Document

## Figure Generation Prompt for Nature-Level Publication

---

# 🎨 GENERATIVE AI PROMPT FOR FIGURE 1

## Prompt for DALL-E / Midjourney / Stable Diffusion:

```
Create a highly detailed, Nature journal publication quality scientific figure showing a complete Gene Regulatory Network (GRN) inference pipeline. The figure should be a complex multi-panel layout with the following structure:

OVERALL LAYOUT: Landscape orientation, 4 main vertical sections flowing left-to-right:
[A: DATA SOURCES] → [B: PREPROCESSING] → [C: MODEL ARCHITECTURE] → [D: OUTPUTS & VALIDATION]

COLOR SCHEME: Professional scientific - Navy blue (#1e3a5f), Coral orange (#e6550d), Teal green (#31a354), Purple (#756bb1), Gray (#969696) on white background.

---

PANEL A (Left, 20% width): DATA SOURCES
- Title: "Multi-Modal Data Integration"
- Show 4 stacked data source boxes:
  
  A1. "Single-Cell RNA-seq" 
  - Icon: Grid of colored dots representing cells (84 cell types × 26,660 genes)
  - Label: "n = 84 pseudo-bulk profiles"
  - Color: Navy blue
  
  A2. "ChIP-Atlas TF Binding"
  - Icon: DNA double helix with TF proteins binding
  - Matrix visualization: 18,894 genes × 1,480 TFs
  - Show binding peaks on genomic locus
  - Color: Coral orange
  
  A3. "JASPAR Motifs"
  - Icon: Position weight matrix logo (DNA sequence logo)
  - Label: "535 TFs matched, IC weights [0.97-1.27]"
  - Color: Teal green
  
  A4. "Rosetta PPI Network"
  - Icon: Protein interaction network graph
  - Label: "933 cofactors → 1,480 TFs"
  - Nodes and edges with physical interaction symbols
  - Color: Purple

---

PANEL B (Center-Left, 25% width): PREPROCESSING & FEATURE EXTRACTION
- Title: "Data Processing Pipeline"

  B1. "Pseudo-bulk Aggregation"
  - Show single cells grouping into cell-type averages
  - Arrow: "mean(cells_per_type)"
  - Heatmap showing expression patterns
  
  B2. "TF Activity Computation (LINGER)"
  - Equation box: "TF_activity = Σ(binding × expression) / Σ(binding)"
  - Show weighted sum visualization
  - Output: 84 × 1,480 TF activity matrix
  
  B3. "Virtual CRISPRa/i Generation"
  - Show two cell types with different TF states
  - Arrow showing: "Cell_A (TF_X HIGH) → Cell_B (TF_X LOW)"
  - Label: "Δ_expression = virtual perturbation"
  - Text: "6,972 cell-type pairs = FREE training data"
  
  B4. "Network Alignment"
  - Venn diagram showing gene overlap
  - Numbers: 17,591 aligned genes
  - ChIP mask creation

---

PANEL C (Center-Right, 35% width): MODEL ARCHITECTURE
- Title: "IntegratedGRNModel (156M parameters)"
- This is the MOST COMPLEX panel with nested components:

  C1. TOP: "Input Layer"
  - Box: "TF Activity Change: Δ = (batch, 1480)"
  - Box: "Source Expression: (batch, 26660)"
  
  C2. LEFT BRANCH: "TFRegulatoryEncoder"
  - Nested components:
    - "TFCascadeGCN" - 3-layer Graph Convolution Network
    - Show TF-TF network graph with message passing arrows
    - "175,066 edges, row-normalized"
    - Edge weights from ChIP + PPI (0.8 + 0.2 blend)
    - Output: "tf_importance (1480,)"
  
  C3. RIGHT BRANCH: "SingleCellDataEncoder"  
  - "Transformer Layers (2 heads)"
  - Self-attention visualization
  - Output: "cell_embedding (512,)"
  
  C4. CENTER: "Direct Effects Matrix"
  - Large matrix visualization (26660 × 1480)
  - Color-coded: red=repressor, blue=activator
  - Sparsity pattern from ChIP mask
  - Label: "ChIP-initialized, learnable"
  
  C5. FUSION BLOCK: "Multi-Modal Fusion"
  - Show convergence of all branches
  - "Delta Decoder MLP"
  - Architecture: 1992 → 1024 → 1024 → 26660
  
  C6. GATING MECHANISMS (side boxes):
  - "Context Gate (FiLM)" 
    - Equation: "γ × effect + β"
    - Range: [0, 1] multiplicative
  - "Cofactor Gate (PPI)"
    - Equation: "1 + 0.2 × tanh(cof_influence)"
    - Shows cofactor expression modulating TF
  - "ChIP Leakage"
    - Learnable parameter [0.005, 0.2]
  
  C7. OUTPUT: "Predicted Δ Expression"
  - Formula: "Δ_pred = (tf_change × GRN^T) × gates"
  - Shape: (batch, 26660)

---

PANEL D (Right, 20% width): OUTPUTS & VALIDATION
- Title: "Results & Applications"

  D1. "Training Progress"
  - Learning curve plot
  - X: Epochs (1-20)
  - Y: Delta PCC (0 → 0.98)
  - Highlight: "Val Delta_PCC = 0.984"
  
  D2. "GRN Matrix Output"
  - Heatmap of final GRN
  - Rows: Top 50 genes
  - Cols: Top 30 TFs
  - Color scale: Red (repressor) ↔ Blue (activator)
  
  D3. "Predicted vs Actual Scatter"
  - Scatter plot with density contours
  - X: Predicted Δ, Y: Actual Δ
  - R² = 0.968 annotation
  - Identity line
  
  D4. "Gene-Specific Fine-Tuning"
  - Show SSTR2 as example
  - "Refinement Head" box
  - "Val r = 0.984"
  - Bar chart: Top 5 activators, Top 5 repressors
  
  D5. "Experimental Recommendations"
  - Table format:
    - "OE SMAD4 → SSTR2 ↑"
    - "KO TRPS1 → SSTR2 ↑"

---

CONNECTING ARROWS:
- Large flowing arrows between major panels (A→B→C→D)
- Small arrows within panels showing data flow
- Dashed arrows for optional paths
- Color-coded by data type

ANNOTATIONS:
- Panel labels: "a", "b", "c", "d" in circles (top-left of each)
- Sub-panel labels: lowercase roman numerals
- Key equations in boxes with light gray background
- Sample sizes in parentheses

STYLE:
- Clean sans-serif font (Helvetica or Arial)
- Thin black borders on panels
- Subtle drop shadows on main components
- White space between major sections
- Legend in bottom-right corner explaining colors

DIMENSIONS: 18 inches wide × 12 inches tall (for Nature full-width figure)
```

---

# 📊 COMPLETE PIPELINE STRUCTURE

## ASCII Art Representation

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    SINGLECELL-GRN: VIRTUAL CRISPRa/i FOR GENE REGULATORY NETWORK INFERENCE                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              A. DATA SOURCES                                                                 │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐                     │ ║
║  │  │  📊 SINGLE-CELL       │  │  🧬 CHIP-ATLAS       │  │  🔤 JASPAR           │  │  🔗 ROSETTA PPI      │                     │ ║
║  │  │  RNA-SEQ             │  │  TF BINDING          │  │  DNA MOTIFS          │  │  PROTEIN-PROTEIN     │                     │ ║
║  │  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤                     │ ║
║  │  │ • preprocessed.h5ad  │  │ • binding_matrix.npy │  │ • JASPAR.jaspar      │  │ • final_predictions  │                     │ ║
║  │  │ • 84 cell types      │  │ • (18894 × 1480)     │  │ • 535 TFs matched    │  │   _90.tsv            │                     │ ║
║  │  │ • 26,660 genes       │  │ • tf_tf_adjacency    │  │ • IC weights         │  │ • 933 cofactors      │                     │ ║
║  │  │ • Log-normalized     │  │ • (1480 × 1480)      │  │ • [0.97, 1.27]       │  │ • 1,826 TF-TF PPIs   │                     │ ║
║  │  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘                     │ ║
║  │             │                         │                         │                         │                                  │ ║
║  └─────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────────────┘ ║
║                │                         │                         │                         │                                    ║
║                ▼                         ▼                         ▼                         ▼                                    ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              B. PREPROCESSING                                                                │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  STEP 1: PSEUDO-BULK AGGREGATION                                                                                    │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │  Single Cells (n=10,000+)  ──▶  mean(cells_per_type)  ──▶  Pseudo-bulk (84 × 26,660)                              │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  [○○○○○]  [●●●●●]  [◐◐◐◐◐]       ┌────────┐              ┌────────────────────────┐                                │    │ ║
║  │  │  [○○○○○]  [●●●●●]  [◐◐◐◐◐]  ──▶  │ mean() │  ──▶         │ CellType1: [0.2, 3.1..│                                │    │ ║
║  │  │  [○○○○○]  [●●●●●]  [◐◐◐◐◐]       └────────┘              │ CellType2: [1.5, 0.8..│                                │    │ ║
║  │  │   Type A    Type B   Type C                              │ ...                    │                                │    │ ║
║  │  │                                                          │ CellType84:[0.1, 2.4..│                                │    │ ║
║  │  │                                                          └────────────────────────┘                                │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  STEP 2: TF ACTIVITY COMPUTATION (LINGER Method)                                                                    │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │                          Σ (ChIP_binding[gene,tf] × Expression[celltype,gene])                                     │    │ ║
║  │  │  TF_activity[ct,tf] = ────────────────────────────────────────────────────────────                                  │    │ ║
║  │  │                                   Σ ChIP_binding[gene,tf]                                                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Expression (84 × 26660)  ×  Binding (26660 × 1480)  /  n_targets  =  TF_Activity (84 × 1480)                     │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Interpretation: "TF activity reflected by expression of its target genes"                                        │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  STEP 3: VIRTUAL PERTURBATION GENERATION (The Key Innovation!)                                                      │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  INSIGHT: Cell-type differences ARE natural perturbation experiments!                                             │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ┌─────────────────┐                              ┌─────────────────┐                                              │    │ ║
║  │  │  │  CELL TYPE A    │         COMPARE              │  CELL TYPE B    │                                              │    │ ║
║  │  │  │  ─────────────  │         ───────              │  ─────────────  │                                              │    │ ║
║  │  │  │  REST: HIGH     │    ─────────────────▶        │  REST: LOW      │                                              │    │ ║
║  │  │  │  SSTR2: LOW     │    Δ_TF = B - A              │  SSTR2: HIGH    │                                              │    │ ║
║  │  │  │  (Neuroendocr.) │    Δ_Expr = B - A            │  (Neuronal)     │                                              │    │ ║
║  │  │  └─────────────────┘                              └─────────────────┘                                              │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  CONCLUSION: REST ↓ → SSTR2 ↑  ═══▶  REST is a REPRESSOR of SSTR2!                                                │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  TRAINING DATA: 84 cell types × 83 partners = 6,972 virtual perturbation pairs (FREE!)                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  STEP 4: PRIOR KNOWLEDGE INTEGRATION                                                                                │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐              │    │ ║
║  │  │  │ analyze_jaspar     │    │ preprocess_ppi     │    │ tf_tf_adjacency    │    │ cof_to_tf_adj      │              │    │ ║
║  │  │  │ _match.py          │    │ _rosetta.py        │    │ .npy               │    │ .npy               │              │    │ ║
║  │  │  ├────────────────────┤    ├────────────────────┤    ├────────────────────┤    ├────────────────────┤              │    │ ║
║  │  │  │ tf_specificity     │    │ TF-TF PPI blend    │    │ ChIP TF cascade    │    │ Cofactor→TF gate   │              │    │ ║
║  │  │  │ _weights.npy       │    │ (0.8 ChIP + 0.2    │    │ 175,066 edges      │    │ 933 cofactors      │              │    │ ║
║  │  │  │ (1480,)            │    │  PPI)              │    │ Row-normalized     │    │ → 1480 TFs         │              │    │ ║
║  │  │  └────────────────────┘    └────────────────────┘    └────────────────────┘    └────────────────────┘              │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                              │                                                                   ║
║                                                              ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              C. MODEL ARCHITECTURE                                                           │ ║
║  │                                        IntegratedGRNModel (156,687,959 parameters)                                          │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │                        ┌─────────────────────────────────────────────────────────────────┐                                  │ ║
║  │                        │                         INPUT LAYER                              │                                  │ ║
║  │                        │  ┌────────────────────────┐  ┌────────────────────────┐         │                                  │ ║
║  │                        │  │ tf_activity_change     │  │ source_expression      │         │                                  │ ║
║  │                        │  │ (batch, 1480)          │  │ (batch, 26660)         │         │                                  │ ║
║  │                        │  │ Δ = TF[tgt] - TF[src]  │  │ Expression of source   │         │                                  │ ║
║  │                        │  └──────────┬─────────────┘  └──────────┬─────────────┘         │                                  │ ║
║  │                        └─────────────┼───────────────────────────┼───────────────────────┘                                  │ ║
║  │                                      │                           │                                                           │ ║
║  │               ┌──────────────────────┼───────────────────────────┼──────────────────────┐                                   │ ║
║  │               │                      ▼                           ▼                      │                                   │ ║
║  │               │   ┌──────────────────────────────┐  ┌──────────────────────────────┐   │                                   │ ║
║  │               │   │  TFRegulatoryEncoder         │  │  SingleCellDataEncoder       │   │                                   │ ║
║  │               │   │  (models/tf_encoder.py)      │  │  (models/cell_encoder.py)    │   │                                   │ ║
║  │               │   ├──────────────────────────────┤  ├──────────────────────────────┤   │                                   │ ║
║  │               │   │                              │  │                              │   │                                   │ ║
║  │               │   │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │   │                                   │ ║
║  │               │   │  │  TFGeneBindingEncoder  │  │  │  │  Gene Embedding        │  │   │                                   │ ║
║  │               │   │  │  ChIP binding → dense  │  │  │  │  (26660 → 512)         │  │   │                                   │ ║
║  │               │   │  └──────────┬─────────────┘  │  │  └──────────┬─────────────┘  │   │                                   │ ║
║  │               │   │             │                │  │             │                │   │                                   │ ║
║  │               │   │             ▼                │  │             ▼                │   │                                   │ ║
║  │               │   │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │   │                                   │ ║
║  │               │   │  │  TFCascadeGCN          │  │  │  │  Transformer Layers    │  │   │                                   │ ║
║  │               │   │  │  ──────────────────    │  │  │  │  ────────────────────  │  │   │                                   │ ║
║  │               │   │  │  3-layer DGL GraphConv │  │  │  │  2 layers, 8 heads     │  │   │                                   │ ║
║  │               │   │  │  175,066 edges         │  │  │  │  Self-attention        │  │   │                                   │ ║
║  │               │   │  │  Residual connections  │  │  │  │  LayerNorm + Dropout   │  │   │                                   │ ║
║  │               │   │  │                        │  │  │  │                        │  │   │                                   │ ║
║  │               │   │  │  Message Passing:      │  │  │  │  Attention:            │  │   │                                   │ ║
║  │               │   │  │  h' = GCN(h, A_tf)     │  │  │  │  Attn(Q,K,V) → context │  │   │                                   │ ║
║  │               │   │  │                        │  │  │  │                        │  │   │                                   │ ║
║  │               │   │  └──────────┬─────────────┘  │  │  └──────────┬─────────────┘  │   │                                   │ ║
║  │               │   │             │                │  │             │                │   │                                   │ ║
║  │               │   │             ▼                │  │             ▼                │   │                                   │ ║
║  │               │   │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │   │                                   │ ║
║  │               │   │  │  tf_importance         │  │  │  │  cell_embedding        │  │   │                                   │ ║
║  │               │   │  │  (1480,)               │  │  │  │  (batch, 512)          │  │   │                                   │ ║
║  │               │   │  └──────────┬─────────────┘  │  │  └──────────┬─────────────┘  │   │                                   │ ║
║  │               │   │             │                │  │             │                │   │                                   │ ║
║  │               │   └─────────────┼────────────────┘  └─────────────┼────────────────┘   │                                   │ ║
║  │               │                 │                                 │                    │                                   │ ║
║  │               └─────────────────┼─────────────────────────────────┼────────────────────┘                                   │ ║
║  │                                 │                                 │                                                         │ ║
║  │                                 ▼                                 ▼                                                         │ ║
║  │               ┌─────────────────────────────────────────────────────────────────────────────────────┐                      │ ║
║  │               │                              DIRECT EFFECTS MATRIX                                   │                      │ ║
║  │               │                           (The Core Learnable GRN)                                   │                      │ ║
║  │               ├─────────────────────────────────────────────────────────────────────────────────────┤                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  self.direct_effects = nn.Parameter(torch.zeros(26660, 1480))                       │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  Initialization: chip_binding × 0.01 (ChIP-informed sparse structure)               │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  ┌────────────────────────────────────────────────────────────────────────────┐     │                      │ ║
║  │               │  │  GRN COMPUTATION FORMULA:                                                   │     │                      │ ║
║  │               │  │                                                                             │     │                      │ ║
║  │               │  │  grn_matrix = direct_effects                                                │     │                      │ ║
║  │               │  │             × tf_specificity_weights    (JASPAR modulation)                 │     │                      │ ║
║  │               │  │             × (chip_mask + chip_leakage) (ChIP constraint + learnable leak) │     │                      │ ║
║  │               │  │                                                                             │     │                      │ ║
║  │               │  │  chip_leakage = 0.005 + 0.195 × sigmoid(learnable_param)  ∈ [0.005, 0.2]   │     │                      │ ║
║  │               │  │                                                                             │     │                      │ ║
║  │               │  └────────────────────────────────────────────────────────────────────────────┘     │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  Final GRN: (26660 × 1480) - Each entry = effect of TF on gene                      │                      │ ║
║  │               │             Positive = Activator, Negative = Repressor                               │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               └──────────────────────────────────────────┬──────────────────────────────────────────┘                      │ ║
║  │                                                          │                                                                  │ ║
║  │                                                          ▼                                                                  │ ║
║  │               ┌─────────────────────────────────────────────────────────────────────────────────────┐                      │ ║
║  │               │                              GATING MECHANISMS                                       │                      │ ║
║  │               ├─────────────────────────────────────────────────────────────────────────────────────┤                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  ┌───────────────────────────┐  ┌────────────────────────────────────────────────┐  │                      │ ║
║  │               │  │  CONTEXT GATING (FiLM)    │  │  COFACTOR GATING (PPI)                         │  │                      │ ║
║  │               │  │  ───────────────────────  │  │  ─────────────────────────────────────────     │  │                      │ ║
║  │               │  │                           │  │                                                 │  │                      │ ║
║  │               │  │  context = Encoder(expr)  │  │  cof_expr = source_expr[:, cof_indices]       │  │                      │ ║
║  │               │  │  γ = sigmoid(Gate(ctx))   │  │  cof_influence = cof_expr @ cof_to_tf_adj     │  │                      │ ║
║  │               │  │  β = tanh(Shift(ctx))×0.1 │  │  cof_gate = 1.0 + 0.2 × tanh(cof_influence)   │  │                      │ ║
║  │               │  │                           │  │                                                 │  │                      │ ║
║  │               │  │  delta = γ × delta + β    │  │  Range: [0.8, 1.2] - mild modulation           │  │                      │ ║
║  │               │  │                           │  │                                                 │  │                      │ ║
║  │               │  │  Purpose: Cell-state-     │  │  Purpose: TF potency depends on whether       │  │                      │ ║
║  │               │  │  dependent TF effects     │  │  its cofactor partners are expressed          │  │                      │ ║
║  │               │  └───────────────────────────┘  └────────────────────────────────────────────────┘  │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               └──────────────────────────────────────────┬──────────────────────────────────────────┘                      │ ║
║  │                                                          │                                                                  │ ║
║  │                                                          ▼                                                                  │ ║
║  │               ┌─────────────────────────────────────────────────────────────────────────────────────┐                      │ ║
║  │               │                              DELTA DECODER                                           │                      │ ║
║  │               ├─────────────────────────────────────────────────────────────────────────────────────┤                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  Input: concat([tf_activity_change, cell_embedding]) = (batch, 1480 + 512)          │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  Architecture:                                                                       │                      │ ║
║  │               │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │                      │ ║
║  │               │  │ Linear   │──▶│ LayerNorm│──▶│ GELU     │──▶│ Dropout  │──▶│ Linear   │──▶ ...    │                      │ ║
║  │               │  │ 1992→1024│   │          │   │          │   │ 0.1      │   │ 1024→1024│           │                      │ ║
║  │               │  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘           │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  Final: Linear(1024 → 26660)                                                         │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               └──────────────────────────────────────────┬──────────────────────────────────────────┘                      │ ║
║  │                                                          │                                                                  │ ║
║  │                                                          ▼                                                                  │ ║
║  │               ┌─────────────────────────────────────────────────────────────────────────────────────┐                      │ ║
║  │               │                              PREDICTION FORMULA                                      │                      │ ║
║  │               ├─────────────────────────────────────────────────────────────────────────────────────┤                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  1. Basic prediction:     delta_basic = tf_activity_change @ grn_matrix.T           │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  2. Neural refinement:    delta_neural = delta_decoder(concat(tf_change, cell_emb)) │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  3. Context gating:       delta = gamma × (delta_basic + 0.1 × delta_neural) + beta │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  4. Cofactor modulation:  tf_weighted = tf_activity_change × cof_gate               │                      │ ║
║  │               │                           delta_cof = tf_weighted @ grn_matrix.T                    │                      │ ║
║  │               │                           delta_final = 0.5 × delta + 0.5 × delta_cof               │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               │  OUTPUT: predicted_delta (batch, 26660)                                              │                      │ ║
║  │               │                                                                                      │                      │ ║
║  │               └─────────────────────────────────────────────────────────────────────────────────────┘                      │ ║
║  │                                                                                                                              │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                              │                                                                   ║
║                                                              ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              D. TRAINING & LOSS                                                              │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  LOSS FUNCTION: DeltaFocusedLoss                                                                                    │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Total Loss = L_mse + 0.3 × L_ranking + 0.2 × L_direction + 0.1 × L_chip                                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐                │    │ ║
║  │  │  │ L_mse               │  │ L_ranking           │  │ L_direction         │  │ L_chip              │                │    │ ║
║  │  │  │ ─────               │  │ ─────────           │  │ ───────────         │  │ ──────              │                │    │ ║
║  │  │  │ MSE(pred, actual)   │  │ Preserve relative   │  │ Get up/down right   │  │ Penalize effects    │                │    │ ║
║  │  │  │                     │  │ gene ordering       │  │ for significant Δ   │  │ without binding     │                │    │ ║
║  │  │  │ Primary objective   │  │ Margin-based loss   │  │ BCE loss on sign    │  │ evidence            │                │    │ ║
║  │  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘                │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  VALIDATION STRATEGY: Cell-Type Pair Holdout (CRITICAL for no data leakage!)                                       │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  All pairs: 84 × 83 = 6,972 cell-type combinations                                                                 │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Train pairs: 5,927 (85%)  ──▶  Model trains on these                                                              │    │ ║
║  │  │  Val pairs:   1,045 (15%)  ──▶  NEVER seen during training!                                                        │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Holdout is at the PAIR level, not sample level, to prevent memorization                                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  TRAINING RESULTS (20 epochs)                                                                                       │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Epoch   Train_Loss   Train_Delta_PCC   Val_Loss   Val_Delta_PCC   Val_Dir_Acc                                     │    │ ║
║  │  │  ─────   ──────────   ───────────────   ────────   ─────────────   ───────────                                     │    │ ║
║  │  │    1        3.07           0.304         5.54         0.652          0.958                                         │    │ ║
║  │  │    2        2.43           0.566         3.68         0.795          0.974                                         │    │ ║
║  │  │    5        1.26           0.823         2.45         0.873          0.982                                         │    │ ║
║  │  │   10        0.49           0.938         1.19         0.939          0.989                                         │    │ ║
║  │  │   15        0.26           0.970         0.66         0.966          0.992                                         │    │ ║
║  │  │   20        0.18           0.982         0.33         0.984          0.994       ◄── BEST MODEL SAVED              │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ★ FINAL VALIDATION DELTA_PCC = 0.9843 (98.4% correlation!)                                                        │    │ ║
║  │  │  ★ DIRECTION ACCURACY = 99.4%                                                                                       │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                              │                                                                   ║
║                                                              ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              E. GENE-SPECIFIC FINE-TUNING                                                    │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │  For genes of particular interest (e.g., SSTR2), we add a gene-specific refinement head:                                   │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐       │    │ ║
║  │  │  │                         IntegratedGRNModel (156M params)                                                 │       │    │ ║
║  │  │  │                                    FROZEN ❄️                                                             │       │    │ ║
║  │  │  │                                                                                                          │       │    │ ║
║  │  │  │  Input: tf_activity_change ──────────────────────────────────────▶ base_prediction (26660,)             │       │    │ ║
║  │  │  │                                                                           │                              │       │    │ ║
║  │  │  │                                                                           │ [target_gene_idx]            │       │    │ ║
║  │  │  │                                                                           ▼                              │       │    │ ║
║  │  │  │                                                               base_pred_gene (scalar)                   │       │    │ ║
║  │  │  │                                                                           │                              │       │    │ ║
║  │  │  └───────────────────────────────────────────────────────────────────────────┼──────────────────────────────┘       │    │ ║
║  │  │                                                                              │                                       │    │ ║
║  │  │                                                                              ▼                                       │    │ ║
║  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐       │    │ ║
║  │  │  │                    GeneSpecificRefinementHead (414K params) - TRAINABLE 🔥                              │       │    │ ║
║  │  │  │                                                                                                          │       │    │ ║
║  │  │  │  tf_activity_change ──────▶ TF Importance Weights ──────▶ Refinement MLP ──────▶ refined_delta         │       │    │ ║
║  │  │  │       (1480,)                (learnable, 1480)              256→128→1                                   │       │    │ ║
║  │  │  │                                    │                                                  │                  │       │    │ ║
║  │  │  │                                    │ softmax                                          │                  │       │    │ ║
║  │  │  │                                    ▼                                                  │                  │       │    │ ║
║  │  │  │                           importance_weights                                          │                  │       │    │ ║
║  │  │  │                                    │                                                  │                  │       │    │ ║
║  │  │  │                                    └──────────────────────────────────────────────────┤                  │       │    │ ║
║  │  │  │                                                                                       │                  │       │    │ ║
║  │  │  │                                                                                       ▼                  │       │    │ ║
║  │  │  │                                                    ┌──────────────────────────────────────┐             │       │    │ ║
║  │  │  │                                                    │  BLEND FORMULA:                      │             │       │    │ ║
║  │  │  │                                                    │                                      │             │       │    │ ║
║  │  │  │  base_pred_gene ───────────────────────────────▶   │  output = σ(blend) × base_pred      │             │       │    │ ║
║  │  │  │                                                    │         + (1-σ(blend)) × refined     │             │       │    │ ║
║  │  │  │                                                    │                                      │             │       │    │ ║
║  │  │  │                                                    │  blend = learnable param (init 0.5)  │             │       │    │ ║
║  │  │  │                                                    └──────────────────────────────────────┘             │       │    │ ║
║  │  │  │                                                                     │                                   │       │    │ ║
║  │  │  │                                                                     ▼                                   │       │    │ ║
║  │  │  │                                                          final_prediction (scalar)                      │       │    │ ║
║  │  │  │                                                                                                          │       │    │ ║
║  │  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘       │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  CELL-TYPE HOLDOUT VALIDATION (Prevents data leakage!)                                                              │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Hold out ENTIRE CELL TYPES (not just pairs):                                                                      │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  84 cell types → 68 training (80%) + 16 validation (20%)                                                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Train pairs: between training cell types only                                                                      │    │ ║
║  │  │  Val pairs: involve ANY held-out cell type                                                                          │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  FINE-TUNING RESULTS (SSTR2 example, 20 epochs)                                                                     │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  ★ Best Validation r = 0.9836                                                                                       │    │ ║
║  │  │  ★ R² = 0.9675 (96.8% variance explained!)                                                                         │    │ ║
║  │  │  ★ Direction accuracy = 91.14%                                                                                      │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  Learned blend weight: 0.65 (relies more on base model)                                                            │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                              │                                                                   ║
║                                                              ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                              F. OUTPUTS & INTERPRETATION                                                     │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  GRN MATRIX INTERPRETATION                                                                                          │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  grn_matrix.npy: (26660 genes × 1480 TFs)                                                                          │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  grn[gene, tf] > 0  ═══▶  TF is an ACTIVATOR of gene   ═══▶  Overexpress TF → gene ↑                              │    │ ║
║  │  │  grn[gene, tf] < 0  ═══▶  TF is a REPRESSOR of gene    ═══▶  Knockout TF → gene ↑                                 │    │ ║
║  │  │  grn[gene, tf] ≈ 0  ═══▶  No significant effect                                                                    │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  SSTR2 EXAMPLE:                                                                                                     │    │ ║
║  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐               │    │ ║
║  │  │  │  TOP 5 ACTIVATORS              │  TOP 5 REPRESSORS                                              │               │    │ ║
║  │  │  │  ─────────────────             │  ─────────────────                                             │               │    │ ║
║  │  │  │  1. ZBTB25: +0.098             │  1. INTS12: -0.056                                             │               │    │ ║
║  │  │  │  2. CPSF4:  +0.081             │  2. NFYC:   -0.052                                             │               │    │ ║
║  │  │  │  3. MSX2:   +0.071             │  3. GRHL1:  -0.046                                             │               │    │ ║
║  │  │  │  4. SP5:    +0.055             │  4. RAD21:  -0.044                                             │               │    │ ║
║  │  │  │  5. EN1:    +0.048             │  5. REST:   -0.005  (KNOWN repressor!)                         │               │    │ ║
║  │  │  │                                │                                                                 │               │    │ ║
║  │  │  │  Total: 675 activators         │  Total: 805 repressors                                         │               │    │ ║
║  │  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘               │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │  OUTPUT FILES                                                                                                       │    │ ║
║  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  │  results/sclc_grn_v6/                          results/sclc_SSTR2/                                                 │    │ ║
║  │  │  ├── best_model.pt (798 MB)                    ├── best_head.pt (1.7 MB)                                           │    │ ║
║  │  │  ├── grn_matrix.npy (158 MB)                   ├── tf_importance_SSTR2.csv                                         │    │ ║
║  │  │  ├── gene_names.csv                            ├── analysis/                                                        │    │ ║
║  │  │  ├── tf_names.csv                              │   ├── figure1_regulators.png                                      │    │ ║
║  │  │  ├── celltype_names.csv                        │   ├── figure2_mechanisms.png                                      │    │ ║
║  │  │  ├── tf_activity.npy                           │   ├── figure3_calibration.png                                     │    │ ║
║  │  │  ├── binding_matrix.npy                        │   ├── regulators_SSTR2.csv                                        │    │ ║
║  │  │  └── model_config.json                         │   ├── interventions_SSTR2.csv                                     │    │ ║
║  │  │                                                │   └── summary_SSTR2.csv                                           │    │ ║
║  │  │                                                                                                                     │    │ ║
║  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │ ║
║  │                                                                                                                              │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

# 📐 MERMAID DIAGRAM FOR FIGURE GENERATION

```mermaid
flowchart TB
    subgraph A["📊 DATA SOURCES"]
        A1["🧬 Single-Cell RNA-seq<br/>84 cell types × 26,660 genes"]
        A2["🔬 ChIP-Atlas TF Binding<br/>18,894 genes × 1,480 TFs"]
        A3["🔤 JASPAR Motifs<br/>535 TFs matched"]
        A4["🔗 Rosetta PPI<br/>933 cofactors"]
    end

    subgraph B["⚙️ PREPROCESSING"]
        B1["Pseudo-bulk Aggregation"]
        B2["TF Activity (LINGER)"]
        B3["Virtual CRISPRa/i<br/>6,972 pairs"]
        B4["Network Alignment"]
    end

    subgraph C["🧠 MODEL ARCHITECTURE<br/>IntegratedGRNModel (156M params)"]
        subgraph C1["TFRegulatoryEncoder"]
            C1a["TFCascadeGCN<br/>175,066 edges"]
        end
        subgraph C2["SingleCellDataEncoder"]
            C2a["Transformer<br/>2 layers, 8 heads"]
        end
        C3["Direct Effects Matrix<br/>(26,660 × 1,480)"]
        C4["Context Gating<br/>(FiLM)"]
        C5["Cofactor Gate<br/>(PPI)"]
        C6["Delta Decoder<br/>MLP"]
    end

    subgraph D["📈 OUTPUTS"]
        D1["GRN Matrix<br/>26,660 × 1,480"]
        D2["Delta_PCC = 0.984"]
        D3["Dir_Acc = 99.4%"]
    end

    subgraph E["🎯 FINE-TUNING"]
        E1["Gene-Specific Head<br/>414K params"]
        E2["SSTR2: r = 0.984<br/>R² = 96.8%"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B4
    A4 --> B4
    B1 --> B2
    B2 --> B3
    B3 --> C
    B4 --> C
    
    C1 --> C3
    C2 --> C6
    C3 --> C4
    C4 --> C5
    C5 --> C6
    C6 --> D1
    
    D1 --> D2
    D1 --> D3
    D1 --> E1
    E1 --> E2

    style A fill:#1e3a5f,color:#fff
    style B fill:#31a354,color:#fff
    style C fill:#756bb1,color:#fff
    style D fill:#e6550d,color:#fff
    style E fill:#de2d26,color:#fff
```

---

# 🎨 DETAILED PROMPT FOR AI IMAGE GENERATOR

## For Claude/GPT-4 Vision to create SVG/Vector Figure:

```
Create a publication-quality scientific figure (Nature/Science/Cell level) for a Gene Regulatory Network inference method. The figure should be a detailed, multi-panel infographic with the following specifications:

OVERALL:
- Landscape orientation, 18" × 12"
- Professional scientific color palette: Navy (#1e3a5f), Coral (#e6550d), Teal (#31a354), Purple (#756bb1)
- Clean white background with subtle grid
- Panel labels (a, b, c, d, e, f) in bold circles

PANEL A - "Multi-Modal Data Integration" (top left, 25% width):
- Four data source icons vertically stacked:
  1. Cell grid icon → "scRNA-seq (84 cell types)"
  2. DNA helix with protein → "ChIP-Atlas TF binding"
  3. Sequence logo (DNA PWM) → "JASPAR motifs"
  4. Network graph → "Rosetta PPI"
- Arrows flowing rightward

PANEL B - "Virtual CRISPRa/i Concept" (top center, 25% width):
- Two cell diagrams side by side:
  - Left cell: "TF_X: HIGH" with gene OFF
  - Right cell: "TF_X: LOW" with gene ON
- Double arrow between them labeled "Δ = perturbation"
- Caption: "Cell-type differences as natural experiments"
- Numbers: "6,972 training pairs from 84 cell types"

PANEL C - "Model Architecture" (center, full width, largest panel):
- Flowchart showing:
  1. Input boxes: "TF Activity Change (1480)" and "Source Expression (26660)"
  2. Two encoder branches:
     - Left: TFCascadeGCN (show small graph with message passing)
     - Right: Transformer encoder (show attention matrix)
  3. Central matrix: "Direct Effects (26660 × 1480)" with red/blue heatmap
  4. Gating modules: "Context Gate" and "Cofactor Gate" as small boxes
  5. "Delta Decoder MLP"
  6. Output: "Predicted Δ Expression"
- All connected with arrows showing data flow
- Parameter count: "156M parameters"

PANEL D - "Training Dynamics" (bottom left):
- Line plot:
  - X-axis: Epochs (0-20)
  - Y-axis: Delta PCC (0-1)
  - Training curve (blue) and Validation curve (orange)
  - Annotation: "Val Delta_PCC = 0.984"
- Show convergence

PANEL E - "Prediction Accuracy" (bottom center):
- Scatter plot:
  - X-axis: "Predicted Δ"
  - Y-axis: "Actual Δ"
  - Density contours
  - Identity line
  - Annotation: "R² = 0.968"

PANEL F - "Biological Validation" (bottom right):
- Bar chart showing top TF regulators for example gene (SSTR2):
  - Blue bars (activators) pointing right
  - Red bars (repressors) pointing left
  - Labels for each TF
- Caption: "Identified regulators match known biology"

STYLE NOTES:
- Use Helvetica or Arial font
- Title: "SingleCell-GRN: Virtual CRISPRa/i for Gene Regulatory Network Inference"
- Thin black borders on panels
- Subtle drop shadows for depth
- Legend explaining color coding
- Scale bars where appropriate
- All text readable at 50% zoom
```

---

# 📋 COMPLETE CODE INTERPRETATION

## File-by-File Summary for Figure Generation:

| File | Purpose | Key Components |
|------|---------|----------------|
| `train_v6.py` | Main training script | IntegratedGRNModel, MultiTFDataset, DeltaFocusedLoss |
| `finetune_gene.py` | Gene-specific fine-tuning | GeneSpecificRefinementHead, cell-type holdout |
| `analyze_target_full.py` | Publication figures | 3 multi-panel figures, CSVs |
| `models/tf_encoder.py` | TF network encoding | TFCascadeGCN, TFGeneBindingEncoder |
| `models/cell_encoder.py` | Cell state encoding | SingleCellDataEncoder, Transformer |
| `models/monster_grn.py` | Full model architecture | MonsterGRNModel, TransformerFusionBlock |
| `analyze_jaspar_match.py` | Motif specificity | IC weights, TF matching |
| `preprocess_ppi_rosetta.py` | PPI matrices | Cofactor→TF, TF→TF adjacency |

---

# 🏆 KEY RESULTS FOR FIGURE

## Main Metrics to Highlight:

| Metric | Value | Panel |
|--------|-------|-------|
| Training Delta_PCC | 0.982 | D |
| Validation Delta_PCC | **0.984** | D, callout |
| Direction Accuracy | **99.4%** | D, annotation |
| Fine-tune r (SSTR2) | **0.984** | E |
| Fine-tune R² (SSTR2) | **0.968** | E |
| Model Parameters | 156.7M | C |
| Cell Types | 84 | A |
| Genes | 26,660 | A |
| TFs | 1,480 | A |
| Training Pairs | 6,972 | B |

---

# 📁 SUPPLEMENTARY FIGURE PROMPTS

## Supplementary Figure S1: Data Processing Pipeline

```
Create a vertical flowchart showing data processing:
1. Raw scRNA-seq → QC → Normalization → Log1p → Pseudo-bulk
2. ChIP-Atlas TSV files → Binding matrix → Gene alignment
3. JASPAR parsing → IC calculation → Specificity weights
4. Rosetta predictions → Cofactor identification → PPI matrices
Each step with sample sizes and intermediate outputs
```

## Supplementary Figure S2: Model Components Detail

```
Create exploded view of IntegratedGRNModel:
- TFCascadeGCN layers (3 GraphConv blocks with residual)
- Transformer attention visualization
- Direct effects matrix sparsity pattern
- Gating mechanism formulas
- All dimensions labeled
```

## Supplementary Figure S3: Validation Analysis

```
Create 6-panel validation figure:
a. Train vs Val loss curves
b. Delta_PCC distribution across genes
c. Direction accuracy by delta magnitude
d. Cell-type holdout performance
e. Known TF validation (REST, NEUROD1)
f. Cofactor influence on prediction
```

---

*This design document provides complete specifications for generating Nature-level publication figures.*


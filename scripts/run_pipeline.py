#!/usr/bin/env python3
"""
Complete Pipeline for SingleCell-GRN

Runs the full analysis pipeline:
1. Preprocess single-cell data
2. Preprocess TF binding data
3. Train Monster GRN model
4. Query for target gene regulators
5. Generate reports

Usage:
    python scripts/run_pipeline.py \
        --singlecell data/raw/counts.h5ad \
        --metadata data/raw/metadata.csv \
        --tf-matrix ../TF_pathways/transcription_factor_gene_matrix.csv \
        --tf-pathways ../TF_pathways/TF_Pathways/ \
        --target-gene SSTR2 \
        --direction increase \
        --output results/analysis/
        
Example with demo data:
    python scripts/run_pipeline.py --demo
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_step(cmd: list, step_name: str, dry_run: bool = False):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("(Dry run - skipping)")
        return
        
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"Error in {step_name}!")
        sys.exit(1)
        
    print(f"Completed: {step_name}")


def create_demo_data(output_dir: Path):
    """Create demo data for testing the pipeline."""
    import numpy as np
    import pandas as pd
    
    print("Creating demo data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Demo expression matrix
    n_cells = 1000
    n_genes = 2000
    n_tfs = 100
    
    # Generate expression
    np.random.seed(42)
    expression = np.random.exponential(1.0, (n_cells, n_genes))
    
    # Add some TF -> Gene relationships
    for i in range(n_tfs):
        for j in range(n_genes):
            if np.random.random() < 0.1:  # 10% of genes regulated
                expression[:, j] += expression[:, i] * np.random.normal(0.5, 0.2)
                
    expression = np.maximum(expression, 0)
    
    # Gene names
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    gene_names[:n_tfs] = [f"TF{i}" for i in range(n_tfs)]
    gene_names[1500] = "SSTR2"  # Add target gene
    
    # Save expression
    df = pd.DataFrame(expression, columns=gene_names)
    df.index = [f"Cell{i}" for i in range(n_cells)]
    df.to_csv(output_dir / "counts.csv")
    
    # Metadata
    meta = pd.DataFrame({
        'batch': [f"Batch{i % 3}" for i in range(n_cells)],
        'patient': [f"Patient{i % 5}" for i in range(n_cells)],
        'cell_type': ['Cancer'] * n_cells
    }, index=df.index)
    meta.to_csv(output_dir / "metadata.csv")
    
    # TF binding matrix
    binding = np.zeros((n_genes, n_tfs))
    for i in range(n_tfs):
        targets = np.random.choice(n_genes, 50, replace=False)
        binding[targets, i] = np.random.uniform(0.5, 1.0, 50)
        
    binding_df = pd.DataFrame(
        binding,
        index=gene_names,
        columns=[f"TF{i}" for i in range(n_tfs)]
    )
    binding_df.to_csv(output_dir / "tf_binding.csv")
    
    print(f"Demo data created in {output_dir}")
    return output_dir / "counts.csv", output_dir / "metadata.csv", output_dir / "tf_binding.csv"


def main():
    parser = argparse.ArgumentParser(description='Run SingleCell-GRN Pipeline')
    parser.add_argument('--singlecell', type=str, default=None,
                        help='Path to single-cell expression data (h5ad or csv)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to cell metadata CSV')
    parser.add_argument('--tf-matrix', type=str, default=None,
                        help='Path to transcription_factor_gene_matrix.csv')
    parser.add_argument('--tf-pathways', type=str, default=None,
                        help='Path to TF_Pathways directory')
    parser.add_argument('--target-gene', type=str, default='SSTR2',
                        help='Target gene to analyze')
    parser.add_argument('--direction', type=str, default='increase',
                        choices=['increase', 'decrease'],
                        help='Desired direction of change')
    parser.add_argument('--output', type=str, default='results/pipeline',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing (use existing)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--demo', action='store_true',
                        help='Run with demo data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Demo mode
    if args.demo:
        demo_dir = output_dir / "demo_data"
        singlecell_path, metadata_path, tf_matrix_path = create_demo_data(demo_dir)
        args.singlecell = str(singlecell_path)
        args.metadata = str(metadata_path)
        args.tf_matrix = str(tf_matrix_path)
        
    # Validate inputs
    if args.singlecell is None and not args.skip_preprocess:
        print("Error: --singlecell is required (or use --demo)")
        sys.exit(1)
        
    scripts_dir = Path(__file__).parent
    python_exe = sys.executable
    
    # Step 1: Preprocess single-cell data
    if not args.skip_preprocess and args.singlecell:
        sc_output = output_dir / "singlecell"
        
        cmd = [
            python_exe, str(scripts_dir / "preprocess_singlecell.py"),
            "--input", args.singlecell,
            "--output", str(sc_output)
        ]
        
        if args.metadata:
            cmd.extend(["--metadata", args.metadata])
            
        run_step(cmd, "Preprocess Single-Cell Data", args.dry_run)
        
    # Step 2: Preprocess TF binding data
    if not args.skip_preprocess and args.tf_matrix:
        tf_output = output_dir / "tf_binding"
        
        cmd = [
            python_exe, str(scripts_dir / "preprocess_tf_matrix.py"),
            "--input", args.tf_matrix,
            "--output", str(tf_output)
        ]
        
        if args.tf_pathways:
            cmd.extend(["--pathway-dir", args.tf_pathways])
            
        run_step(cmd, "Preprocess TF Binding Data", args.dry_run)
        
    # Step 3: Train model
    if not args.skip_training:
        model_output = output_dir / "model"
        sc_data = output_dir / "singlecell" / "preprocessed.h5ad"
        
        # Check for h5ad or fall back to numpy
        if not sc_data.exists():
            sc_data = output_dir / "singlecell"
            
        cmd = [
            python_exe, str(scripts_dir / "train.py"),
            "--singlecell", str(sc_data),
            "--output", str(model_output),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ]
        
        tf_binding = output_dir / "tf_binding"
        if tf_binding.exists():
            cmd.extend(["--tf-binding", str(tf_binding)])
            
        run_step(cmd, "Train Monster GRN Model", args.dry_run)
        
    # Step 4: Query target gene
    model_path = output_dir / "model" / "best_model.pt"
    if not model_path.exists():
        model_path = output_dir / "model" / "final_model.pt"
        
    analysis_output = output_dir / f"analysis_{args.target_gene}"
    
    cmd = [
        python_exe, str(scripts_dir / "query_target.py"),
        "--model", str(model_path),
        "--gene", args.target_gene,
        "--direction", args.direction,
        "--output", str(analysis_output),
        "--device", args.device
    ]
    
    tf_binding = output_dir / "tf_binding"
    if tf_binding.exists():
        cmd.extend(["--tf-binding", str(tf_binding)])
        
    run_step(cmd, f"Query Target Gene: {args.target_gene}", args.dry_run)
    
    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Model: {output_dir / 'model'}")
    print(f"  - Analysis: {analysis_output}")
    print(f"  - Report: {analysis_output / 'analysis_report.html'}")


if __name__ == '__main__':
    main()


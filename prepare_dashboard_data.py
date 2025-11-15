#!/usr/bin/env python3
"""
Prepare Dashboard Data - Create optimized data file for deployment

This script processes all raw clustering results and creates a single
optimized Parquet file containing only the data needed for the dashboard.
This makes the app lightweight and suitable for online deployment.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Import data processing functions from the main app
from interactive_metrics_viz import (
    load_all_results,
    prepare_data,
    METRICS
)

def prepare_optimized_data(
    source_results_dir: str,
    output_file: str = "dashboard_data.parquet",
    include_feature_importance: bool = False
) -> pd.DataFrame:
    """
    Load and process all data, then save to optimized Parquet file.
    
    Args:
        source_results_dir: Path to directory containing raw CSV results
        output_file: Output filename (will be saved in PROMIS_Dashboard/results/)
        include_feature_importance: Whether to include feature importance data
        
    Returns:
        Processed DataFrame
    """
    print(f"Loading data from: {source_results_dir}")
    print("This may take a few minutes...")
    
    # Load all raw results
    raw_df = load_all_results(source_results_dir)
    
    if raw_df.empty:
        print("ERROR: No data found in source directory!")
        return pd.DataFrame()
    
    print(f"Loaded {len(raw_df)} raw rows from {source_results_dir}")
    
    # Process data (add panel, variant, stratification columns)
    print("Processing data (extracting panels, variants, stratification)...")
    processed_df = prepare_data(raw_df)
    
    print(f"Processed {len(processed_df)} rows")
    
    # Select only essential columns for dashboard
    essential_columns = [
        # Metrics
        'silhouette_score',
        'davies_bouldin_index', 
        'calinski_harabasz_index',
        'wss',
        # Metadata
        'method',  # clustering method
        'dr_method',
        'variant',
        'panel',
        'stratification',
        # Parameters
        'n_clusters',
        'dr_components',
        # Optional: for feature importance merging
        'filename',
        'dataname'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in essential_columns if col in processed_df.columns]
    
    # Keep only essential columns
    optimized_df = processed_df[available_columns].copy()
    
    # Remove rows with all NaN metrics (invalid results)
    metric_cols = [m for m in METRICS if m in optimized_df.columns]
    if metric_cols:
        optimized_df = optimized_df.dropna(subset=metric_cols, how='all')
    
    print(f"Optimized to {len(optimized_df)} rows with {len(available_columns)} columns")
    
    # Save to Parquet (compressed, efficient format)
    output_path = Path(__file__).parent / "results" / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Saving to: {output_path}")
    optimized_df.to_parquet(output_path, compression='snappy', index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total rows: {len(optimized_df):,}")
    print(f"Total columns: {len(optimized_df.columns)}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\nVariants: {sorted(optimized_df['variant'].unique())}")
    print(f"Clustering methods: {sorted(optimized_df['method'].unique())}")
    print(f"DR methods: {sorted(optimized_df['dr_method'].unique())}")
    print(f"Panels: {sorted([p for p in optimized_df['panel'].dropna().unique() if p])}")
    print(f"Stratifications: {sorted(optimized_df['stratification'].unique())}")
    print(f"Number of clusters: {sorted(optimized_df['n_clusters'].dropna().unique())}")
    print(f"DR components: {sorted(optimized_df['dr_components'].dropna().unique())}")
    print("="*60)
    
    return optimized_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare optimized dashboard data file for deployment"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="../results/",
        help="Path to source results directory (default: ../results/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dashboard_data.parquet",
        help="Output filename (default: dashboard_data.parquet)"
    )
    
    args = parser.parse_args()
    
    # Check if source directory exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_path}")
        print(f"Please provide a valid path to your results directory.")
        sys.exit(1)
    
    # Prepare data
    df = prepare_optimized_data(
        source_results_dir=str(source_path),
        output_file=args.output
    )
    
    if not df.empty:
        print(f"\n✓ Success! Optimized data saved to: results/{args.output}")
        print("\nYou can now run the dashboard with minimal data loading:")
        print("  streamlit run app.py")
    else:
        print("\n✗ Failed to prepare data. Please check the source directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()


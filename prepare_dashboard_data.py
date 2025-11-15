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
    fi_output_file: str = "feature_importance_data.parquet"
) -> tuple:
    """
    Load and process all data, then save to optimized Parquet files.
    
    Args:
        source_results_dir: Path to directory containing raw CSV results
        output_file: Output filename for clustering data (will be saved in PROMIS_Dashboard/results/)
        fi_output_file: Output filename for feature importance data
        
    Returns:
        Tuple of (clustering_df, feature_importance_df)
    """
    print(f"Loading data from: {source_results_dir}")
    print("This may take a few minutes...")
    
    # Load all raw clustering results
    raw_df = load_all_results(source_results_dir)
    
    if raw_df.empty:
        print("ERROR: No clustering data found in source directory!")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Loaded {len(raw_df)} raw clustering rows from {source_results_dir}")
    
    # Process data (add panel, variant, stratification columns)
    print("Processing clustering data (extracting panels, variants, stratification)...")
    processed_df = prepare_data(raw_df)
    
    print(f"Processed {len(processed_df)} clustering rows")
    
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
    
    print(f"Optimized clustering data to {len(optimized_df)} rows with {len(available_columns)} columns")
    
    # Load feature importance data
    print("\nLoading feature importance data...")
    from interactive_metrics_viz import load_feature_importance_data
    fi_df = load_feature_importance_data(source_results_dir)
    
    if not fi_df.empty:
        print(f"Loaded {len(fi_df)} feature importance rows")
        print(f"Feature importance columns: {list(fi_df.columns)}")
    else:
        print("No feature importance data found (this is optional)")
    
    # Save clustering data to Parquet
    output_path = Path(__file__).parent / "results" / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving clustering data to: {output_path}")
    optimized_df.to_parquet(output_path, compression='snappy', index=False)
    
    # Save feature importance data if available
    fi_output_path = Path(__file__).parent / "results" / fi_output_file
    if not fi_df.empty:
        print(f"Saving feature importance data to: {fi_output_path}")
        fi_df.to_parquet(fi_output_path, compression='snappy', index=False)
        fi_size = fi_output_path.stat().st_size / (1024*1024)
    else:
        fi_size = 0
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Clustering data:")
    print(f"  Total rows: {len(optimized_df):,}")
    print(f"  Total columns: {len(optimized_df.columns)}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    if not fi_df.empty:
        print(f"\nFeature importance data:")
        print(f"  Total rows: {len(fi_df):,}")
        print(f"  Total columns: {len(fi_df.columns)}")
        print(f"  File size: {fi_size:.2f} MB")
    
    print(f"\nVariants: {sorted(optimized_df['variant'].unique())}")
    print(f"Clustering methods: {sorted(optimized_df['method'].unique())}")
    print(f"DR methods: {sorted(optimized_df['dr_method'].unique())}")
    print(f"Panels: {sorted([p for p in optimized_df['panel'].dropna().unique() if p])}")
    print(f"Stratifications: {sorted(optimized_df['stratification'].unique())}")
    print(f"Number of clusters: {sorted(optimized_df['n_clusters'].dropna().unique())}")
    print(f"DR components: {sorted(optimized_df['dr_components'].dropna().unique())}")
    print("="*60)
    
    return optimized_df, fi_df


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
    clustering_df, fi_df = prepare_optimized_data(
        source_results_dir=str(source_path),
        output_file=args.output,
        fi_output_file="feature_importance_data.parquet"
    )
    
    if not clustering_df.empty:
        print(f"\n✓ Success! Optimized data saved:")
        print(f"  - Clustering data: results/{args.output}")
        if not fi_df.empty:
            print(f"  - Feature importance: results/feature_importance_data.parquet")
        print("\nYou can now run the dashboard with minimal data loading:")
        print("  streamlit run app.py")
    else:
        print("\n✗ Failed to prepare data. Please check the source directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()


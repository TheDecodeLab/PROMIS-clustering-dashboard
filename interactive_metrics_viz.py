#!/usr/bin/env python3
"""
Interactive Visualization for Clustering Metrics

This script creates an interactive Streamlit app to visualize clustering metrics
across different variants, with filtering capabilities.
"""

import re
import glob
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Check for required dependencies
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy import stats
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install required packages: pip install numpy pandas plotly scipy")
    sys.exit(1)

try:
    import streamlit as st
except ImportError:
    print("Error: streamlit is not installed.")
    print("Please install it with: pip install streamlit")
    print("Or: conda install -c conda-forge streamlit")
    sys.exit(1)

# Metrics to visualize (ordered: silhouette, DBI, CHI, WSS)
METRICS = ['silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_index', 'wss']
METRIC_LABELS = {
    'wss': 'Within Sum of Squares (WSS)',
    'silhouette_score': 'Silhouette Score',
    'davies_bouldin_index': 'Davies-Bouldin Index',
    'calinski_harabasz_index': 'Calinski-Harabasz Index'
}

METRIC_ABBREVIATIONS = {
    'wss': 'WSS',
    'silhouette_score': 'Silhouette',
    'davies_bouldin_index': 'DBI',
    'calinski_harabasz_index': 'CHI'
}

# Metric direction indicators (for y-axis titles)
METRIC_DIRECTIONS = {
    'wss': 'lower is better',
    'silhouette_score': 'higher is better',
    'davies_bouldin_index': 'lower is better',
    'calinski_harabasz_index': 'higher is better'
}

# Default y-axis limits for each metric (to exclude very bad results)
# Format: (min, max) or None for auto
# Defaults match run_raw_analysis.py thresholds:
#   --min-silhouette 0.25, --max-dbi 2.0, --min-chi 100.0
# For WSS, we use 99th percentile as default max (lower is better)
METRIC_Y_LIMITS = {
    'wss': (None, None),  # Will be set dynamically based on percentiles
    'silhouette_score': (-1.0, 1.0),  # Full range (min threshold: 0.25)
    'davies_bouldin_index': (0.0, 2.0),  # Max threshold: 2.0 (lower is better)
    'calinski_harabasz_index': (100.0, None)  # Min threshold: 100.0 (higher is better)
}

# Default threshold values (for display and editing)
# These match run_raw_analysis.py defaults
METRIC_THRESHOLDS = {
    'wss': {'min': None, 'max': None},  # Will use percentile-based defaults
    'silhouette_score': {'min': 0.25, 'max': None},  # Min threshold: 0.25
    'davies_bouldin_index': {'min': None, 'max': 2.0},  # Max threshold: 2.0
    'calinski_harabasz_index': {'min': 100.0, 'max': None}  # Min threshold: 100.0
}

# Variant order for x-axis
# Note: Aligned with PROMIS_Dashboard variant naming
VARIANT_ORDER = ['raw', 'tabpfn', 'ufs', 'denoising']

# Marker mapping for DR methods
DR_MARKER_MAP = {
    'pca': 'circle',
    'tsne': 'square',
    'umap': 'diamond',
    'denoising': 'triangle-up',
    'standard': 'x',
    'vae': 'star',
    'none': 'circle'  # fallback
}

# Color palette for clustering methods (Plotly default colors)
CLUSTERING_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


@st.cache_data
def load_all_results(results_dir: str, progress_callback=None) -> pd.DataFrame:
    """
    Recursively scan results_dir for clustering CSV files and load them.
    
    Args:
        results_dir: Path to results directory
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Combined DataFrame with all results
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        if progress_callback:
            progress_callback(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    # Find all clustering results CSV files
    pattern = str(results_dir / "**" / "clustering_*_results_*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        if progress_callback:
            progress_callback(f"No clustering results CSV files found in {results_dir}")
        return pd.DataFrame()
    
    if progress_callback:
        progress_callback(f"Found {len(csv_files)} CSV files. Loading...")
    
    all_data = []
    total_files = len(csv_files)
    
    for idx, csv_file in enumerate(csv_files):
        try:
            if progress_callback:
                progress_callback(f"Loading file {idx + 1}/{total_files}: {Path(csv_file).name}")
            
            df = pd.read_csv(csv_file)
            # Add source directory column
            df['__source_dir'] = str(Path(csv_file).parent)
            all_data.append(df)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error loading {csv_file}: {e}")
            continue
    
    if not all_data:
        if progress_callback:
            progress_callback("No data loaded successfully")
        return pd.DataFrame()
    
    # Combine all DataFrames
    if progress_callback:
        progress_callback(f"Combining {len(all_data)} DataFrames...")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if progress_callback:
        progress_callback(f"Loaded {len(combined_df)} total rows from {len(csv_files)} files")
    
    return combined_df


def extract_panel_from_dataname(dataname: str, source_dir: str = None, data_file: str = None) -> Optional[str]:
    """
    Extract panel (all, 12, 13, 23, rz) from dataname, source directory, or data_file.
    
    Args:
        dataname: The dataname column value
        source_dir: Source directory path
        data_file: The data_file column value
        
    Returns:
        Panel string ("all", "12", "13", "23", "rz") or None
    """
    if pd.isna(dataname):
        dataname = ""
    else:
        dataname = str(dataname)
    
    # Try to extract from dataname first
    # Raw: data_all, data_12, data_13, data_23, data_rz
    match = re.search(r'data_(all|12|13|23|rz)', dataname)
    if match:
        return match.group(1)
    
    # TabPFN: data_all_tabpfn_123
    match = re.search(r'data_(all|12|13|23|rz)_tabpfn', dataname)
    if match:
        return match.group(1)
    
    # UFS: ufs_data_all_fw_k10_123
    match = re.search(r'ufs_data_(all|12|13|23|rz)', dataname)
    if match:
        return match.group(1)
    
    # Embeddings: Check data_file column if available
    if data_file and not pd.isna(data_file):
        data_file_str = str(data_file)
        match = re.search(r'data_(all|12|13|23|rz)', data_file_str)
        if match:
            return match.group(1)
    
    # Embeddings: Check source directory for feature importance files
    if source_dir:
        source_path = Path(source_dir)
        # Look for feature importance files that might contain panel info
        fi_files = list(source_path.glob("feature_importance_*_data_*.csv"))
        if fi_files:
            # Extract from first matching file name
            match = re.search(r'data_(all|12|13|23|rz)', fi_files[0].name)
            if match:
                return match.group(1)
    
    return None


def map_to_variant(dataname: str, source_dir: str = None) -> str:
    """
    Map dataname/source to variant category.
    
    Args:
        dataname: The dataname column value
        source_dir: Source directory path
        
    Returns:
        Variant string: "raw", "tabpfn", "ufs", or "denoising"
    """
    if pd.isna(dataname):
        dataname = ""
    else:
        dataname = str(dataname)
    
    # TabPFN variant: contains "tabpfn" (check first)
    if 'tabpfn' in dataname.lower():
        return 'tabpfn'
    
    # Denoising variant: contains "denoising" or embeddings/autoencoder patterns
    if 'denoising' in dataname.lower():
        return 'denoising'
    
    # UFS variant: starts with "ufs_data_"
    if re.match(r'^ufs_data_', dataname):
        return 'ufs'
    
    # Embeddings/autoencoders (denoising): embeddings_lstm, embeddings_gru, etc.
    if dataname in ['embeddings_lstm', 'embeddings_gru', 'embeddings_lstm_attn', 'embeddings_gru_attn']:
        return 'denoising'
    
    # Raw variant: exact match pattern (applied last as fallback)
    if re.match(r'^data_(all|12|13|23|rz)(?:_(male|female|young|old))?$', dataname):
        return 'raw'
    
    # Fallback: try to infer from source directory
    if source_dir:
        source_str = str(source_dir)
        if 'denoising' in source_str.lower():
            return 'denoising'
        if 'ae_' in source_str or 'embeddings' in source_str.lower():
            return 'denoising'
        if 'tabpfn' in source_str.lower():
            return 'tabpfn'
        if 'ufs' in source_str.lower():
            return 'ufs'
        if 'raw_' in source_str.lower():
            return 'raw'
    
    # Default fallback
    return 'unknown'


def extract_panel_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized panel extraction from DataFrame columns.
    
    Args:
        df: DataFrame with 'dataname', '__source_dir', and optionally 'data_file' columns
        
    Returns:
        Series with panel values
    """
    # Initialize result series with None
    panel_series = pd.Series([None] * len(df), index=df.index)
    
    # Convert dataname to string, handling NaN
    dataname_str = df['dataname'].fillna('').astype(str)
    
    # Pattern 1: Raw/TabPFN/UFS from dataname - data_(all|12|13|23|rz)
    pattern1 = r'data_(all|12|13|23|rz)'
    match1 = dataname_str.str.extract(pattern1, expand=False)
    panel_series = panel_series.fillna(match1)
    
    # Pattern 2: TabPFN specific - data_(all|12|13|23|rz)_tabpfn
    pattern2 = r'data_(all|12|13|23|rz)_tabpfn'
    match2 = dataname_str.str.extract(pattern2, expand=False)
    panel_series = panel_series.fillna(match2)
    
    # Pattern 3: UFS - ufs_data_(all|12|13|23|rz)
    pattern3 = r'ufs_data_(all|12|13|23|rz)'
    match3 = dataname_str.str.extract(pattern3, expand=False)
    panel_series = panel_series.fillna(match3)
    
    # Pattern 4: From data_file column if available
    if 'data_file' in df.columns:
        data_file_str = df['data_file'].fillna('').astype(str)
        match4 = data_file_str.str.extract(pattern1, expand=False)
        panel_series = panel_series.fillna(match4)
    
    # Pattern 5: From source directory (only for missing values - expensive operation)
    # Check source_dir files only for rows where panel is still None
    missing_mask = panel_series.isna()
    if missing_mask.any() and '__source_dir' in df.columns:
        # For rows with missing panel, try to extract from source directory files
        # This is slower but only runs for missing values
        missing_indices = df.index[missing_mask]
        for idx in missing_indices:
            source_dir = df.loc[idx, '__source_dir']
            if pd.notna(source_dir):
                try:
                    source_path = Path(str(source_dir))
                    fi_files = list(source_path.glob("feature_importance_*_data_*.csv"))
                    if fi_files:
                        match = re.search(r'data_(all|12|13|23|rz)', fi_files[0].name)
                        if match:
                            panel_series.loc[idx] = match.group(1)
                except Exception:
                    # Skip if there's an error accessing the directory
                    pass
    
    return panel_series


def extract_stratification_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized stratification extraction from DataFrame columns.
    Extracts 'overall', 'male', 'female', 'old', 'young' from dataname, data_file, or source_dir.
    
    Args:
        df: DataFrame with 'dataname', '__source_dir', and optionally 'data_file' columns
        
    Returns:
        Series with stratification values: 'overall', 'male', 'female', 'old', 'young', or None
    """
    # Initialize result series with None
    strat_series = pd.Series([None] * len(df), index=df.index)
    
    # Convert columns to string, handling NaN
    dataname_str = df['dataname'].fillna('').astype(str)
    data_file_str = df['data_file'].fillna('').astype(str) if 'data_file' in df.columns else pd.Series([''] * len(df), index=df.index)
    source_dir_str = df['__source_dir'].fillna('').astype(str) if '__source_dir' in df.columns else pd.Series([''] * len(df), index=df.index)
    
    # Combine all strings for pattern matching
    combined_str = dataname_str + ' ' + data_file_str + ' ' + source_dir_str
    
    # Pattern for stratification: _male, _female, _young, _old
    # Check for male
    male_pattern = r'_(male)(?:_|$|\.)'
    male_mask = combined_str.str.contains(male_pattern, regex=True, na=False)
    strat_series[male_mask] = 'male'
    
    # Check for female
    female_pattern = r'_(female)(?:_|$|\.)'
    female_mask = combined_str.str.contains(female_pattern, regex=True, na=False)
    strat_series[female_mask] = 'female'
    
    # Check for old
    old_pattern = r'_(old)(?:_|$|\.)'
    old_mask = combined_str.str.contains(old_pattern, regex=True, na=False)
    strat_series[old_mask] = 'old'
    
    # Check for young
    young_pattern = r'_(young)(?:_|$|\.)'
    young_mask = combined_str.str.contains(young_pattern, regex=True, na=False)
    strat_series[young_mask] = 'young'
    
    # If no stratification found, mark as 'overall'
    strat_series[strat_series.isna()] = 'overall'
    
    return strat_series


def map_to_variant_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized variant mapping from DataFrame columns.
    
    Args:
        df: DataFrame with 'dataname' and '__source_dir' columns
        
    Returns:
        Series with variant values
    """
    # Initialize result series with 'unknown'
    variant_series = pd.Series(['unknown'] * len(df), index=df.index)
    
    # Convert dataname to string, handling NaN
    dataname_str = df['dataname'].fillna('').astype(str)
    
    # Raw variant: exact match pattern
    raw_pattern = r'^data_(all|12|13|23|rz)(?:_(male|female|young|old))?$'
    raw_mask = dataname_str.str.match(raw_pattern, na=False)
    variant_series[raw_mask] = 'raw'
    
    # TabPFN variant: contains "tabpfn" (case insensitive)
    tabpfn_mask = dataname_str.str.contains('tabpfn', case=False, na=False)
    variant_series[tabpfn_mask] = 'tabpfn'
    
    # Denoising variant: contains "denoising" or embeddings/autoencoder patterns
    denoising_mask = dataname_str.str.contains('denoising', case=False, na=False)
    variant_series[denoising_mask] = 'denoising'
    
    # UFS variant: starts with "ufs_data_"
    ufs_mask = dataname_str.str.match(r'^ufs_data_', na=False)
    variant_series[ufs_mask] = 'ufs'
    
    # Embeddings/autoencoders (denoising): exact matches
    embeddings_list = ['embeddings_lstm', 'embeddings_gru', 'embeddings_lstm_attn', 'embeddings_gru_attn']
    ehr_mask = dataname_str.isin(embeddings_list)
    variant_series[ehr_mask] = 'denoising'
    
    # Fallback: check source directory for remaining 'unknown' values
    if '__source_dir' in df.columns:
        source_str = df['__source_dir'].fillna('').astype(str)
        unknown_mask = variant_series == 'unknown'
        
        # Check source directory patterns
        denoising_src_mask = unknown_mask & (source_str.str.contains('denoising', case=False, na=False) |
                                             source_str.str.contains('ae_', case=False, na=False) | 
                                             source_str.str.contains('embeddings', case=False, na=False))
        variant_series[denoising_src_mask] = 'denoising'
        
        tabpfn_src_mask = unknown_mask & source_str.str.contains('tabpfn', case=False, na=False)
        variant_series[tabpfn_src_mask] = 'tabpfn'
        
        ufs_src_mask = unknown_mask & source_str.str.contains('ufs', case=False, na=False)
        variant_series[ufs_src_mask] = 'ufs'
        
        raw_src_mask = unknown_mask & source_str.str.contains('raw_', case=False, na=False)
        variant_series[raw_src_mask] = 'raw'
    
    return variant_series


@st.cache_data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by adding panel, variant, and stratification columns using vectorized operations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with panel, variant, and stratification columns added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Add stratification column
    df['stratification'] = extract_stratification_vectorized(df)
    
    # Extract panel using vectorized operations
    df['panel'] = extract_panel_vectorized(df)
    
    # Map to variant using vectorized operations
    df['variant'] = map_to_variant_vectorized(df)
    
    # Filter out rows where panel extraction failed (optional - comment out if you want to see all)
    # df = df[df['panel'].notna()]
    
    return df


def get_marker_symbol(dr_method: str) -> str:
    """Get marker symbol for DR method."""
    if pd.isna(dr_method):
        return 'circle'
    dr_method = str(dr_method).lower()
    return DR_MARKER_MAP.get(dr_method, 'circle')


def create_scatter_plot(
    df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    clustering_methods: List[str],
    dr_methods: List[str],
    max_points_per_trace: int = 500,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_log_scale: bool = False
) -> go.Figure:
    """
    Create a scatter plot for a single metric.
    
    Args:
        df: Filtered DataFrame
        metric: Metric column name
        variant_order: Order of variants for x-axis
        clustering_methods: List of clustering methods to include
        dr_methods: List of DR methods to include
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Create x-axis positions: group DR methods by variant
    # Build position mapping: variant -> dr_method -> position
    x_positions = []
    x_tick_labels = []
    variant_group_positions = []
    variant_labels = []
    dr_position_map = {}  # Maps (variant, dr_method) to x position
    
    position = 0
    for variant in variant_order:
        variant_start = position
        variant_dr_count = 0
        
        for dr_method in dr_methods:
            # Check if this variant+dr combination has data
            variant_dr_df = df[(df['variant'] == variant) & (df['dr_method'] == dr_method)]
            if not variant_dr_df.empty and not variant_dr_df[metric].dropna().empty:
                dr_position_map[(variant, dr_method)] = position
                x_positions.append(position)
                x_tick_labels.append(dr_method)
                position += 1
                variant_dr_count += 1
        
        if variant_dr_count > 0:
            variant_end = position - 0.5
            variant_group_positions.append((variant_start - 0.5, variant_end))
            variant_labels.append((variant, (variant_start + variant_end) / 2))
            position += 0.5  # Add spacing between variant groups
    
    # Get unique combinations of clustering method and DR method
    legend_added_cluster = set()
    
    for cluster_method in clustering_methods:
        cluster_df = df[df['method'] == cluster_method]
        
        for dr_method in dr_methods:
            dr_df = cluster_df[cluster_df['dr_method'] == dr_method]
            
            if dr_df.empty:
                continue
            
            # Group by variant and get metric values
            for variant in variant_order:
                variant_df = dr_df[dr_df['variant'] == variant]
                
                if variant_df.empty:
                    continue
                
                # Get base x position for this variant+dr combination
                if (variant, dr_method) not in dr_position_map:
                    continue
                
                base_x_pos = dr_position_map[(variant, dr_method)]
                metric_values = variant_df[metric].dropna().values
                
                if len(metric_values) == 0:
                    continue
                
                # Sample data if too many points
                if len(metric_values) > max_points_per_trace:
                    trace_hash = hash((cluster_method, dr_method, variant))
                    rng = np.random.RandomState(seed=abs(trace_hash) % (2**31))
                    sample_indices = rng.choice(len(metric_values), max_points_per_trace, replace=False)
                    sample_indices = np.sort(sample_indices)
                    metric_values = metric_values[sample_indices]
                    variant_df = variant_df.iloc[sample_indices]
                
                # Add small jitter for visibility
                jitter = (np.arange(len(metric_values)) % 10 - 5) * 0.02
                x_positions_scatter = base_x_pos + jitter
                
                # Get marker symbol and color
                marker_symbol = get_marker_symbol(dr_method)
                color_idx = clustering_methods.index(cluster_method) % len(CLUSTERING_COLORS)
                
                # Create hover text
                n_clusters_vals = variant_df['n_clusters'].fillna('N/A').astype(str).values
                dr_components_vals = variant_df['dr_components'].fillna('N/A').astype(str).values
                
                hover_text = [
                    f"Variant: {variant}<br>"
                    f"Clustering: {cluster_method}<br>"
                    f"DR: {dr_method}<br>"
                    f"{METRIC_LABELS[metric]}: {val:.4f}<br>"
                    f"Clusters: {n_clust}<br>"
                    f"DR Components: {dr_comp}"
                    for val, n_clust, dr_comp in zip(metric_values, n_clusters_vals, dr_components_vals)
                ]
                
                # Show clustering method in legend (once per method)
                show_cluster_legend = cluster_method not in legend_added_cluster
                if show_cluster_legend:
                    legend_added_cluster.add(cluster_method)
                
                fig.add_trace(go.Scatter(
                    x=x_positions_scatter,
                    y=metric_values,
                    mode='markers',
                    name=cluster_method if show_cluster_legend else "",
                    marker=dict(
                        symbol=marker_symbol,
                        color=CLUSTERING_COLORS[color_idx],
                        size=8,
                        line=dict(width=0.5, color='white'),
                        opacity=0.7
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=cluster_method,
                    showlegend=show_cluster_legend
                ))
    
    # Add vertical separator lines between variant groups
    shapes = []
    annotations = []
    
    for (start, end), (variant_name, center_pos) in zip(variant_group_positions, variant_labels):
        shapes.append(dict(
            type='line',
            xref='x',
            yref='paper',
            x0=end,
            y0=0,
            x1=end,
            y1=1,
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        annotations.append(dict(
            x=center_pos,
            y=1.05,
            xref='x',
            yref='paper',
            text=f"<b>{variant_name}</b>",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=4
        ))
    
    # Get metric direction for y-axis title
    direction = METRIC_DIRECTIONS.get(metric, '')
    y_title = f"{METRIC_LABELS[metric]} ({direction})" if direction else METRIC_LABELS[metric]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{METRIC_LABELS[metric]} by Variant and DR Method (Scatter Plot)",
            x=0.5,  # Center the title
            xanchor='center',
            y=0.98,  # Position below legend
            yanchor='top',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text="DR Method", font=dict(size=16)),
            tickmode='array',
            tickvals=x_positions,
            ticktext=x_tick_labels,
            showgrid=True,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=16)),
            showgrid=True,
            range=[y_min, y_max] if (y_min is not None and y_max is not None) else None,
            type='log' if y_log_scale else 'linear',
            tickfont=dict(size=14)
        ),
        height=400,
        hovermode='closest',
        showlegend=True,
        margin=dict(l=80, r=20, t=180, b=80),  # Increased top margin for legend and title
        legend=dict(
            title="Clustering Method (Color)",
            yanchor="bottom",
            y=1.15,  # Moved even higher to avoid overlap with title
            xanchor="center",
            x=0.5,
            orientation="h",
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        shapes=shapes,
        annotations=annotations
    )
    
    return fig


def create_violin_plot(
    df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    clustering_methods: List[str],
    dr_methods: List[str],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_log_scale: bool = False
) -> go.Figure:
    """
    Create a violin plot for a single metric with all DR methods in one figure.
    
    Args:
        df: Filtered DataFrame
        metric: Metric column name
        variant_order: Order of variants for x-axis
        clustering_methods: List of clustering methods to include
        dr_methods: List of DR methods to include
        y_min: Optional y-axis minimum
        y_max: Optional y-axis maximum
        y_log_scale: Whether to use log scale
        
    Returns:
        Plotly Figure
    """
    # Prepare data for violin plot
    plot_data = []
    for variant in variant_order:
        variant_df = df[df['variant'] == variant]
        if variant_df.empty:
            continue
        
        for cluster_method in clustering_methods:
            cluster_df = variant_df[variant_df['method'] == cluster_method]
            if cluster_df.empty:
                continue
            
            for dr_method in dr_methods:
                dr_df = cluster_df[cluster_df['dr_method'] == dr_method]
                if dr_df.empty:
                    continue
                
                metric_values = dr_df[metric].dropna().values
                if len(metric_values) == 0:
                    continue
                
                for val in metric_values:
                    plot_data.append({
                        'variant': variant,
                        'method': cluster_method,
                        'dr_method': dr_method,
                        'value': val,
                        'variant_dr': f"{variant}\n({dr_method})"  # Combined label
                    })
    
    if not plot_data:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected filters", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create x-axis with just DR methods, but group by variant
    # We'll use numeric positions and custom tick labels
    x_positions = []
    x_tick_labels = []
    variant_group_positions = []  # Track where each variant group starts/ends
    variant_labels = []  # Track variant names for annotations
    
    position = 0
    for variant_idx, variant in enumerate(variant_order):
        variant_start = position
        variant_dr_count = 0
        
        for dr_method in dr_methods:
            # Check if this variant+dr combination has data
            if len(plot_df[(plot_df['variant'] == variant) & (plot_df['dr_method'] == dr_method)]) > 0:
                x_positions.append(position)
                x_tick_labels.append(dr_method)
                position += 1
                variant_dr_count += 1
        
        if variant_dr_count > 0:
            variant_end = position - 0.5
            variant_group_positions.append((variant_start - 0.5, variant_end))
            variant_labels.append((variant, (variant_start + variant_end) / 2))
            position += 0.5  # Add spacing between variant groups
    
    # Map variant_dr to numeric positions
    position_map = {}
    pos_idx = 0
    for variant in variant_order:
        for dr_method in dr_methods:
            key = f"{variant}\n({dr_method})"
            if key in plot_df['variant_dr'].values:
                position_map[key] = x_positions[pos_idx]
                pos_idx += 1
    
    # Create a numeric x column for positioning
    plot_df['x_pos'] = plot_df['variant_dr'].map(position_map)
    plot_df = plot_df.dropna(subset=['x_pos'])
    
    # Create violin plot using plotly express
    fig = px.violin(
        plot_df,
        x='x_pos',
        y='value',
        color='method',
        box=True,
        points='outliers',
        hover_data=['dr_method', 'variant'],
        category_orders={'variant': variant_order, 'dr_method': dr_methods}
    )
    
    # Update x-axis with custom labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_positions,
        ticktext=x_tick_labels,
        title=dict(text="DR Method", font=dict(size=16)),
        showgrid=True,
        tickfont=dict(size=14)
    )
    
    # Add vertical separator lines between variant groups
    shapes = []
    annotations = []
    
    for (start, end), (variant_name, center_pos) in zip(variant_group_positions, variant_labels):
        # Add vertical line separator
        shapes.append(dict(
            type='line',
            xref='x',
            yref='paper',
            x0=end,
            y0=0,
            x1=end,
            y1=1,
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Add variant label annotation above the group
        annotations.append(dict(
            x=center_pos,
            y=1.05,
            xref='x',
            yref='paper',
            text=f"<b>{variant_name}</b>",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=4
        ))
    
    # Get metric direction for y-axis title
    direction = METRIC_DIRECTIONS.get(metric, '')
    y_title = f"{METRIC_LABELS[metric]} ({direction})" if direction else METRIC_LABELS[metric]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{METRIC_LABELS[metric]} by Variant and DR Method (Violin Plot)",
            x=0.5,  # Center the title
            xanchor='center',
            y=0.98,  # Position below legend
            yanchor='top',
            font=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=16)),
            showgrid=True,
            range=[y_min, y_max] if (y_min is not None and y_max is not None) else None,
            type='log' if y_log_scale else 'linear',
            tickfont=dict(size=14)
        ),
        height=600,
        margin=dict(l=80, r=20, t=180, b=80),  # Increased top margin for legend and title
        legend=dict(
            title="Clustering Method (Color)",
            yanchor="bottom",
            y=1.15,  # Moved even higher to avoid overlap with title
            xanchor="center",
            x=0.5,  # Center horizontally
            orientation="h",  # Horizontal layout
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        shapes=shapes,
        annotations=annotations
    )
    
    # Update colors to match our color scheme
    for i, trace in enumerate(fig.data):
        if trace.name in clustering_methods:
            color_idx = clustering_methods.index(trace.name) % len(CLUSTERING_COLORS)
            trace.marker.color = CLUSTERING_COLORS[color_idx]
            trace.line.color = CLUSTERING_COLORS[color_idx]
            trace.fillcolor = CLUSTERING_COLORS[color_idx]
    
    return fig


def create_boxen_plot(
    df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    clustering_methods: List[str],
    dr_methods: List[str],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_log_scale: bool = False
) -> go.Figure:
    """
    Create a box plot for a single metric with all DR methods in one figure.
    Note: Plotly doesn't have native boxen, so we use box plot.
    
    Args:
        df: Filtered DataFrame
        metric: Metric column name
        variant_order: Order of variants for x-axis
        clustering_methods: List of clustering methods to include
        dr_methods: List of DR methods to include
        y_min: Optional y-axis minimum
        y_max: Optional y-axis maximum
        y_log_scale: Whether to use log scale
        
    Returns:
        Plotly Figure
    """
    # Prepare data for box plot
    plot_data = []
    for variant in variant_order:
        variant_df = df[df['variant'] == variant]
        if variant_df.empty:
            continue
        
        for cluster_method in clustering_methods:
            cluster_df = variant_df[variant_df['method'] == cluster_method]
            if cluster_df.empty:
                continue
            
            for dr_method in dr_methods:
                dr_df = cluster_df[cluster_df['dr_method'] == dr_method]
                if dr_df.empty:
                    continue
                
                metric_values = dr_df[metric].dropna().values
                if len(metric_values) == 0:
                    continue
                
                for val in metric_values:
                    plot_data.append({
                        'variant': variant,
                        'method': cluster_method,
                        'dr_method': dr_method,
                        'value': val,
                        'variant_dr': f"{variant}\n({dr_method})"  # Combined label
                    })
    
    if not plot_data:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected filters", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create x-axis with just DR methods, but group by variant
    # We'll use numeric positions and custom tick labels
    x_positions = []
    x_tick_labels = []
    variant_group_positions = []  # Track where each variant group starts/ends
    variant_labels = []  # Track variant names for annotations
    
    position = 0
    for variant_idx, variant in enumerate(variant_order):
        variant_start = position
        variant_dr_count = 0
        
        for dr_method in dr_methods:
            # Check if this variant+dr combination has data
            if len(plot_df[(plot_df['variant'] == variant) & (plot_df['dr_method'] == dr_method)]) > 0:
                x_positions.append(position)
                x_tick_labels.append(dr_method)
                position += 1
                variant_dr_count += 1
        
        if variant_dr_count > 0:
            variant_end = position - 0.5
            variant_group_positions.append((variant_start - 0.5, variant_end))
            variant_labels.append((variant, (variant_start + variant_end) / 2))
            position += 0.5  # Add spacing between variant groups
    
    # Map variant_dr to numeric positions
    position_map = {}
    pos_idx = 0
    for variant in variant_order:
        for dr_method in dr_methods:
            key = f"{variant}\n({dr_method})"
            if key in plot_df['variant_dr'].values:
                position_map[key] = x_positions[pos_idx]
                pos_idx += 1
    
    # Create a numeric x column for positioning
    plot_df['x_pos'] = plot_df['variant_dr'].map(position_map)
    plot_df = plot_df.dropna(subset=['x_pos'])
    
    # Create box plot using plotly express
    fig = px.box(
        plot_df,
        x='x_pos',
        y='value',
        color='method',
        points=False,
        hover_data=['dr_method', 'variant'],
        category_orders={'variant': variant_order, 'dr_method': dr_methods},
        notched=False
    )
    
    # Update x-axis with custom labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_positions,
        ticktext=x_tick_labels,
        title=dict(text="DR Method", font=dict(size=16)),
        showgrid=True,
        tickfont=dict(size=14)
    )
    
    # Add vertical separator lines between variant groups
    shapes = []
    annotations = []
    
    for (start, end), (variant_name, center_pos) in zip(variant_group_positions, variant_labels):
        # Add vertical line separator
        shapes.append(dict(
            type='line',
            xref='x',
            yref='paper',
            x0=end,
            y0=0,
            x1=end,
            y1=1,
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Add variant label annotation above the group
        annotations.append(dict(
            x=center_pos,
            y=1.05,
            xref='x',
            yref='paper',
            text=f"<b>{variant_name}</b>",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=4
        ))
    
    # Get metric direction for y-axis title
    direction = METRIC_DIRECTIONS.get(metric, '')
    y_title = f"{METRIC_LABELS[metric]} ({direction})" if direction else METRIC_LABELS[metric]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{METRIC_LABELS[metric]} by Variant and DR Method (Box Plot)",
            x=0.5,  # Center the title
            xanchor='center',
            y=0.98,  # Position below legend
            yanchor='top',
            font=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=16)),
            showgrid=True,
            range=[y_min, y_max] if (y_min is not None and y_max is not None) else None,
            type='log' if y_log_scale else 'linear',
            tickfont=dict(size=14)
        ),
        height=600,
        margin=dict(l=80, r=20, t=180, b=80),  # Increased top margin for legend and title
        legend=dict(
            title="Clustering Method (Color)",
            yanchor="bottom",
            y=1.15,  # Moved even higher to avoid overlap with title
            xanchor="center",
            x=0.5,  # Center horizontally
            orientation="h",  # Horizontal layout
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        shapes=shapes,
        annotations=annotations
    )
    
    # Update colors to match our color scheme
    for i, trace in enumerate(fig.data):
        if trace.name in clustering_methods:
            color_idx = clustering_methods.index(trace.name) % len(CLUSTERING_COLORS)
            trace.marker.color = CLUSTERING_COLORS[color_idx]
            trace.line.color = CLUSTERING_COLORS[color_idx]
            trace.fillcolor = CLUSTERING_COLORS[color_idx]
    
    return fig


def calculate_statistics_table(
    df: pd.DataFrame,
    metric: str,
    clustering_methods: List[str],
    dr_methods: List[str],
    variant_order: List[str],
    stratification_selection: str = "overall"
) -> pd.DataFrame:
    """
    Calculate statistics table with mean and 95% CI for each combination.
    
    Args:
        df: Filtered DataFrame
        metric: Metric column name
        clustering_methods: List of clustering methods
        dr_methods: List of DR methods
        variant_order: Order of variants
        
    Returns:
        DataFrame with statistics formatted as "mean (lower-upper)"
    """
    # Filter to only selected methods
    df_filtered = df[
        (df['method'].isin(clustering_methods)) &
        (df['dr_method'].isin(dr_methods)) &
        (df['variant'].isin(variant_order))
    ].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Determine stratification groups
    if stratification_selection == "sex-stratified":
        strat_groups = ['male', 'female']
        strat_labels = ['Male', 'Female']
    elif stratification_selection == "age-stratified":
        strat_groups = ['old', 'young']
        strat_labels = ['Old', 'Young']
    else:
        strat_groups = ['overall']
        strat_labels = ['Overall']
    
    # Group by clustering, DR method, variant, and stratification
    results = []
    
    for cluster_method in clustering_methods:
        for dr_method in dr_methods:
            # For stratified views, create one row per stratification group
            # For overall, create one row
            if stratification_selection != "overall":
                for strat_group, strat_label in zip(strat_groups, strat_labels):
                    row_data = {
                        'Clustering': cluster_method,
                        'DR': dr_method,
                        'Stratification': strat_label
                    }
                    
                    for variant in variant_order:
                        variant_data = df_filtered[
                            (df_filtered['method'] == cluster_method) &
                            (df_filtered['dr_method'] == dr_method) &
                            (df_filtered['variant'] == variant) &
                            (df_filtered['stratification'] == strat_group)
                        ][metric].dropna()
                        
                        # Filter out infinite values and NaN
                        variant_data = variant_data[np.isfinite(variant_data)]
                        
                        if len(variant_data) > 0:
                            mean_val = float(variant_data.mean())
                            
                            # Calculate 95% CI using t-distribution
                            if len(variant_data) > 1:
                                ci = stats.t.interval(
                                    0.95,
                                    len(variant_data) - 1,
                                    loc=mean_val,
                                    scale=stats.sem(variant_data)
                                )
                                lower = float(ci[0])
                                upper = float(ci[1])
                            else:
                                # If only one value, use the value itself
                                lower = mean_val
                                upper = mean_val
                            
                            # Format based on metric type
                            if metric == 'calinski_harabasz_index':
                                row_data[variant] = f"{mean_val:.0f} ({lower:.0f}; {upper:.0f})"
                            else:
                                row_data[variant] = f"{mean_val:.1f} ({lower:.1f}; {upper:.1f})"
                        else:
                            row_data[variant] = "N/A"
                    
                    results.append(row_data)
            else:
                # Overall view - no stratification column
                row_data = {
                    'Clustering': cluster_method,
                    'DR': dr_method
                }
                
                for variant in variant_order:
                    variant_data = df_filtered[
                        (df_filtered['method'] == cluster_method) &
                        (df_filtered['dr_method'] == dr_method) &
                        (df_filtered['variant'] == variant)
                    ][metric].dropna()
                    
                    # Filter out infinite values and NaN
                    variant_data = variant_data[np.isfinite(variant_data)]
                    
                    if len(variant_data) > 0:
                        mean_val = float(variant_data.mean())
                        
                        # Calculate 95% CI using t-distribution
                        if len(variant_data) > 1:
                            ci = stats.t.interval(
                                0.95,
                                len(variant_data) - 1,
                                loc=mean_val,
                                scale=stats.sem(variant_data)
                            )
                            lower = float(ci[0])
                            upper = float(ci[1])
                        else:
                            # If only one value, use the value itself
                            lower = mean_val
                            upper = mean_val
                        
                        # Format based on metric type
                        if metric == 'calinski_harabasz_index':
                            row_data[variant] = f"{mean_val:.0f} ({lower:.0f}; {upper:.0f})"
                        else:
                            row_data[variant] = f"{mean_val:.1f} ({lower:.1f}; {upper:.1f})"
                    else:
                        row_data[variant] = "N/A"
                
                results.append(row_data)
    
    stats_df = pd.DataFrame(results)
    return stats_df


def calculate_p_value(group1_data: pd.Series, group2_data: pd.Series) -> float:
    """
    Calculate p-value between two groups using t-test or Mann-Whitney U test.
    
    Args:
        group1_data: Series of values for group 1
        group2_data: Series of values for group 2
        
    Returns:
        p-value (float)
    """
    if len(group1_data) < 2 or len(group2_data) < 2:
        return np.nan
    
    # Remove NaN values
    group1_clean = group1_data.dropna()
    group2_clean = group2_data.dropna()
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan
    
    try:
        # Try t-test first (assumes normal distribution)
        from scipy.stats import ttest_ind, mannwhitneyu
        
        # Check if we can use t-test (both groups have at least 3 values)
        if len(group1_clean) >= 3 and len(group2_clean) >= 3:
            try:
                # Use Welch's t-test by default (more robust)
                t_stat, p_value = ttest_ind(group1_clean, group2_clean, equal_var=False)
                return float(p_value)
            except:
                # Fallback to Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_value = mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
                    return float(p_value)
                except:
                    return np.nan
        else:
            # Use Mann-Whitney U test for small samples
            try:
                u_stat, p_value = mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
                return float(p_value)
            except:
                return np.nan
    except ImportError:
        return np.nan


def calculate_all_groups_p_value(group_data_dict: dict) -> float:
    """
    Calculate p-value comparing all groups using ANOVA or Kruskal-Wallis test.
    
    Args:
        group_data_dict: Dictionary mapping group names to Series of values
        
    Returns:
        p-value (float)
    """
    if len(group_data_dict) < 2:
        return np.nan
    
    # Collect all groups' data
    groups = []
    for group_name, group_data in group_data_dict.items():
        clean_data = group_data.dropna()
        # Filter out infinite values
        clean_data = clean_data[np.isfinite(clean_data)]
        if len(clean_data) > 0:
            groups.append(clean_data.values)
    
    if len(groups) < 2:
        return np.nan
    
    try:
        from scipy.stats import f_oneway, kruskal
        
        # Try ANOVA first (assumes normal distribution)
        if all(len(g) >= 3 for g in groups):
            try:
                f_stat, p_value = f_oneway(*groups)
                return float(p_value)
            except:
                # Fallback to Kruskal-Wallis (non-parametric)
                try:
                    h_stat, p_value = kruskal(*groups)
                    return float(p_value)
                except:
                    return np.nan
        else:
            # Use Kruskal-Wallis for small samples
            try:
                h_stat, p_value = kruskal(*groups)
                return float(p_value)
            except:
                return np.nan
    except ImportError:
        return np.nan


def create_stratified_pairwise_table(
    df: pd.DataFrame,
    metrics: List[str],  # List of metric column names
    compare_by: str,  # 'method', 'dr_method', 'variant', 'n_clusters', 'dr_components'
    strat_groups: List[str],  # e.g., ['male', 'female'] or ['old', 'young']
    strat_labels: List[str],  # e.g., ['Male', 'Female'] or ['Old', 'Young']
    other_filters: dict = None  # Additional filters to apply
) -> pd.DataFrame:
    """
    Create a pairwise comparison table for stratified analysis.
    Compares stratified groups (e.g., male vs female) for each value of compare_by.
    
    Args:
        df: Filtered DataFrame
        metrics: List of metric column names to include
        compare_by: Dimension to compare ('method', 'dr_method', 'variant', 'n_clusters', 'dr_components')
        strat_groups: List of stratification group values (e.g., ['male', 'female'])
        strat_labels: List of stratification group labels (e.g., ['Male', 'Female'])
        other_filters: Dict of additional filters {column: [values]}
        
    Returns:
        DataFrame with pairwise comparison statistics
    """
    # Apply additional filters if provided
    filtered_df = df.copy()
    if other_filters:
        for col, values in other_filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # For n_clusters, group values > 10 into ">10"
    if compare_by == 'n_clusters' and 'n_clusters' in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['n_clusters_grouped'] = filtered_df['n_clusters'].apply(
            lambda x: ">10" if pd.notna(x) and x > 10 else x
        )
        compare_by = 'n_clusters_grouped'
    
    # Get unique values for comparison dimension
    def sort_key(x):
        if x == ">10":
            return (1, float('inf'))
        else:
            try:
                return (0, float(x))
            except (ValueError, TypeError):
                return (0, float('inf'))
    
    compare_values = sorted([v for v in filtered_df[compare_by].dropna().unique() if pd.notna(v)], key=sort_key)
    
    if len(compare_values) == 0 or len(strat_groups) != 2:
        return pd.DataFrame()  # Need exactly 2 stratified groups for pairwise comparison
    
    # Map compare_by to readable column names
    column_name_map = {
        'method': 'Clustering Method',
        'dr_method': 'DR Method',
        'variant': 'Variant',
        'n_clusters': 'Number of Clusters',
        'n_clusters_grouped': 'Number of Clusters',
        'dr_components': 'N Dim'
    }
    compare_column_name = column_name_map.get(compare_by, compare_by.title())
    
    # Calculate statistics for each compare_by value and each stratified group
    results = []
    
    for compare_val in compare_values:
        row = {compare_column_name: compare_val}
        
        # For each metric, calculate stats for both stratified groups and p-value
        for metric in metrics:
            if metric not in filtered_df.columns:
                continue
            
            metric_abbr = METRIC_ABBREVIATIONS.get(metric, metric)
            format_str = ".0f" if metric == 'calinski_harabasz_index' else ".1f"
            
            # Get data for each stratified group
            group1_data = filtered_df[
                (filtered_df[compare_by] == compare_val) &
                (filtered_df['stratification'] == strat_groups[0])
            ][metric].dropna()
            group1_data = group1_data[np.isfinite(group1_data)]
            
            group2_data = filtered_df[
                (filtered_df[compare_by] == compare_val) &
                (filtered_df['stratification'] == strat_groups[1])
            ][metric].dropna()
            group2_data = group2_data[np.isfinite(group2_data)]
            
            # Calculate statistics for group 1
            if len(group1_data) > 0:
                mean1 = float(group1_data.mean())
                if len(group1_data) > 1:
                    ci1 = stats.t.interval(0.95, len(group1_data) - 1, loc=mean1, scale=stats.sem(group1_data))
                    lower1 = float(ci1[0])
                    upper1 = float(ci1[1])
                else:
                    lower1 = mean1
                    upper1 = mean1
                n1 = len(group1_data)
                group1_str = f"{mean1:{format_str}} ({lower1:{format_str}}; {upper1:{format_str}})"
            else:
                group1_str = "N/A"
                n1 = 0
            
            # Calculate statistics for group 2
            if len(group2_data) > 0:
                mean2 = float(group2_data.mean())
                if len(group2_data) > 1:
                    ci2 = stats.t.interval(0.95, len(group2_data) - 1, loc=mean2, scale=stats.sem(group2_data))
                    lower2 = float(ci2[0])
                    upper2 = float(ci2[1])
                else:
                    lower2 = mean2
                    upper2 = mean2
                n2 = len(group2_data)
                group2_str = f"{mean2:{format_str}} ({lower2:{format_str}}; {upper2:{format_str}})"
            else:
                group2_str = "N/A"
                n2 = 0
            
            # Calculate p-value between groups
            p_value = calculate_p_value(group1_data, group2_data)
            if np.isnan(p_value):
                p_value_str = "N/A"
            elif p_value < 0.05:
                p_value_str = "<0.05"
            else:
                p_value_str = f"{p_value:.4f}"
            
            # Add columns: mean (CI) for each group, and p-value (N columns removed for space)
            row[f"{metric_abbr} ({strat_labels[0]})"] = group1_str
            row[f"{metric_abbr} ({strat_labels[1]})"] = group2_str
            row[f"{metric_abbr} (p-value)"] = p_value_str
        
        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Convert all columns to strings to avoid Arrow serialization issues
    # This ensures consistent types for Streamlit display
    for col in result_df.columns:
        result_df[col] = result_df[col].astype(str)
    
    return result_df


def load_feature_importance_data(results_path: str) -> pd.DataFrame:
    """
    Load feature importance data from CSV files in the results directory.
    
    Args:
        results_path: Path to results directory
        
    Returns:
        DataFrame with feature importance data
    """
    results_dir = Path(results_path)
    fi_files = list(results_dir.glob("**/feature_importance_results_*.csv"))
    
    if not fi_files:
        return pd.DataFrame()
    
    frames = []
    for fi_file in fi_files:
        try:
            df = pd.read_csv(fi_file)
            # Extract metadata from filename or columns
            if 'label_file' in df.columns:
                frames.append(df)
        except Exception as e:
            # Silently skip files that can't be read
            continue
    
    if not frames:
        return pd.DataFrame()
    
    fi_df = pd.concat(frames, ignore_index=True)
    return fi_df


def create_feature_importance_comparison(
    fi_df: pd.DataFrame,
    clustering_df: pd.DataFrame,
    strat_groups: List[str],
    strat_labels: List[str],
    other_filters: dict = None,
    top_k: int = 10
) -> go.Figure:
    """
    Create errorbar plot comparing feature importance between stratified groups.
    Shows union of top-k features from each group.
    
    Args:
        fi_df: Feature importance DataFrame
        clustering_df: Clustering results DataFrame (for filtering)
        strat_groups: List of stratification group values (e.g., ['male', 'female'])
        strat_labels: List of stratification group labels (e.g., ['Male', 'Female'])
        other_filters: Dict of additional filters {column: [values]}
        top_k: Number of top features to consider from each group
        
    Returns:
        Plotly figure with errorbar plot
    """
    if fi_df.empty or clustering_df.empty:
        return go.Figure()
    
    # Merge feature importance with clustering data to get stratification info
    if 'label_file' not in fi_df.columns or 'filename' not in clustering_df.columns:
        return go.Figure()
    
    # Merge on label_file/filename
    # Check which columns exist in clustering_df
    merge_cols = ['filename']
    for col in ['method', 'dr_method', 'variant', 'stratification', 'dataname', 'n_clusters', 'dr_components', 'panel']:
        if col in clustering_df.columns:
            merge_cols.append(col)
    
    # Ensure we have the columns we need
    if 'filename' not in clustering_df.columns or 'label_file' not in fi_df.columns:
        return go.Figure()
    
    # Merge on label_file/filename
    # Try exact match first, then try basename match if that fails
    merged_df = fi_df.merge(
        clustering_df[merge_cols],
        left_on='label_file',
        right_on='filename',
        how='inner'
    )
    
    # If merge failed, try matching on basename
    if merged_df.empty and len(fi_df) > 0:
        # Extract basename from label_file for matching
        fi_df_copy = fi_df.copy()
        fi_df_copy['label_basename'] = fi_df_copy['label_file'].apply(lambda x: Path(x).name if pd.notna(x) else '')
        clustering_df_copy = clustering_df.copy()
        clustering_df_copy['filename_basename'] = clustering_df_copy['filename'].apply(lambda x: Path(x).name if pd.notna(x) else '')
        
        merged_df = fi_df_copy.merge(
            clustering_df_copy[merge_cols + ['filename_basename']],
            left_on='label_basename',
            right_on='filename_basename',
            how='inner'
        )
        # Remove the temporary basename columns
        if 'label_basename' in merged_df.columns:
            merged_df = merged_df.drop(columns=['label_basename'])
        if 'filename_basename' in merged_df.columns:
            merged_df = merged_df.drop(columns=['filename_basename'])
    
    # Map column names if needed (feature importance uses 'clustering_method', clustering uses 'method')
    if 'clustering_method' in merged_df.columns and 'method' not in merged_df.columns:
        merged_df['method'] = merged_df['clustering_method']
    
    # Apply additional filters
    if other_filters:
        for col, values in other_filters.items():
            if not values:  # Skip if no values to filter
                continue
            if col in merged_df.columns:
                # For dr_components, handle float comparison
                if col == 'dr_components':
                    merged_df = merged_df[merged_df[col].isin([float(v) for v in values])]
                else:
                    merged_df = merged_df[merged_df[col].isin(values)]
            # Also check for alternative column names
            elif col == 'method' and 'clustering_method' in merged_df.columns:
                merged_df = merged_df[merged_df['clustering_method'].isin(values)]
            # For panel, also check dataname if panel column doesn't exist
            elif col == 'panel' and 'panel' not in merged_df.columns and 'dataname' in merged_df.columns:
                # Try to filter by dataname pattern if panel info is in dataname
                panel_values = values if isinstance(values, list) else [values]
                panel_mask = pd.Series([False] * len(merged_df), index=merged_df.index)
                for panel_val in panel_values:
                    if panel_val == "all":
                        panel_mask = pd.Series([True] * len(merged_df), index=merged_df.index)
                        break
                    else:
                        # Check if dataname contains the panel info
                        panel_mask |= merged_df['dataname'].str.contains(f"_{panel_val}_", na=False, regex=False) | \
                                     merged_df['dataname'].str.contains(f"panel{panel_val}", na=False, regex=False)
                merged_df = merged_df[panel_mask]
    
    if merged_df.empty:
        return go.Figure()
    
    # Get importance columns
    importance_cols = [col for col in merged_df.columns if col.startswith('importance_')]
    if not importance_cols:
        return go.Figure()
    
    # Extract feature names
    feature_names = [col.replace('importance_', '') for col in importance_cols]
    
    # Calculate top features for each group
    group_top_features = {}
    all_features_data = {}
    
    for strat_group, strat_label in zip(strat_groups, strat_labels):
        group_df = merged_df[merged_df['stratification'] == strat_group]
        
        if group_df.empty:
            continue
        
        # Calculate mean importance for each feature across all runs
        feature_means = {}
        feature_data = {}
        
        for feature_col, feature_name in zip(importance_cols, feature_names):
            if feature_col in group_df.columns:
                values = group_df[feature_col].dropna()
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    feature_means[feature_name] = float(values.mean())
                    feature_data[feature_name] = values.values
        
        # Get top k features
        sorted_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:top_k]]
        group_top_features[strat_label] = top_features
        all_features_data[strat_label] = feature_data
    
    # Get union of top features
    union_features = set()
    for features in group_top_features.values():
        union_features.update(features)
    union_features = sorted(list(union_features))
    
    if not union_features:
        return go.Figure()
    
    # Calculate statistics for each feature and each group
    plot_data = []
    
    for feature in union_features:
        for strat_group, strat_label in zip(strat_groups, strat_labels):
            group_df = merged_df[merged_df['stratification'] == strat_group]
            
            if group_df.empty:
                continue
            
            feature_col = f'importance_{feature}'
            if feature_col not in group_df.columns:
                continue
            
            values = group_df[feature_col].dropna()
            values = values[np.isfinite(values)]
            
            if len(values) > 0:
                mean_val = float(values.mean())
                
                # Calculate 95% CI
                if len(values) > 1:
                    ci = stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=stats.sem(values))
                    lower = float(ci[0])
                    upper = float(ci[1])
                else:
                    lower = mean_val
                    upper = mean_val
                
                plot_data.append({
                    'feature': feature,
                    'group': strat_label,
                    'mean': mean_val,
                    'lower': lower,
                    'upper': upper,
                    'n': len(values)
                })
    
    if not plot_data:
        return go.Figure()
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create errorbar plot
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    for idx, (strat_group, strat_label) in enumerate(zip(strat_groups, strat_labels)):
        group_data = plot_df[plot_df['group'] == strat_label]
        
        if group_data.empty:
            continue
        
        # Sort by mean importance
        group_data = group_data.sort_values('mean', ascending=True)
        
        # Convert to lists for Plotly
        x_vals = group_data['mean'].tolist()
        y_vals = group_data['feature'].tolist()
        error_plus = (group_data['upper'] - group_data['mean']).tolist()
        error_minus = (group_data['mean'] - group_data['lower']).tolist()
        
        # Create hover text with mean and CI
        hover_text = [
            f"Group: {strat_label}<br>"
            f"Feature: {feat}<br>"
            f"Mean: {mean:.4f}<br>"
            f"95% CI: [{lower:.4f}, {upper:.4f}]<br>"
            f"Lower: {lower:.4f}<br>"
            f"Upper: {upper:.4f}"
            for feat, mean, lower, upper in zip(
                group_data['feature'],
                group_data['mean'],
                group_data['lower'],
                group_data['upper']
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=strat_label,
            marker=dict(
                color=colors[idx % len(colors)],
                size=10,
                line=dict(width=1, color='darkblue' if idx == 0 else 'darkorange')
            ),
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                thickness=2,
                width=3,
                color=colors[idx % len(colors)]
            ),
            text=hover_text,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Feature Importance Comparison: {strat_labels[0]} vs {strat_labels[1]}<br><sub>Union of Top-{top_k} Features from Each Group</sub>",
            font=dict(size=18)
        ),
        xaxis_title=dict(text="Feature Importance (95% CI)", font=dict(size=16)),
        yaxis_title=dict(text="Feature", font=dict(size=16)),
        height=max(400, len(union_features) * 30),
        hovermode='closest',
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        )
    )
    
    return fig


def create_overall_feature_importance(
    fi_df: pd.DataFrame,
    clustering_df: pd.DataFrame,
    other_filters: dict = None,
    top_k: int = 10
) -> go.Figure:
    """
    Create errorbar plot showing overall feature importance (not a comparison).
    
    Args:
        fi_df: Feature importance DataFrame
        clustering_df: Clustering results DataFrame (for filtering)
        other_filters: Dict of additional filters {column: [values]}
        top_k: Number of top features to show
        
    Returns:
        Plotly figure with errorbar plot
    """
    if fi_df.empty or clustering_df.empty:
        return go.Figure()
    
    # Merge feature importance with clustering data
    if 'label_file' not in fi_df.columns or 'filename' not in clustering_df.columns:
        return go.Figure()
    
    merge_cols = ['filename']
    for col in ['method', 'dr_method', 'variant', 'dataname']:
        if col in clustering_df.columns:
            merge_cols.append(col)
    
    # Merge on label_file/filename
    merged_df = fi_df.merge(
        clustering_df[merge_cols],
        left_on='label_file',
        right_on='filename',
        how='inner'
    )
    
    # If merge failed, try matching on basename
    if merged_df.empty:
        clustering_df['filename_basename'] = clustering_df['filename'].apply(lambda x: Path(x).name if pd.notna(x) else x)
        fi_df['label_file_basename'] = fi_df['label_file'].apply(lambda x: Path(x).name if pd.notna(x) else x)
        merged_df = fi_df.merge(
            clustering_df[merge_cols + ['filename_basename']],
            left_on='label_file_basename',
            right_on='filename_basename',
            how='inner'
        )
    
    if merged_df.empty:
        return go.Figure()
    
    # Map column names if needed (feature importance uses 'clustering_method', clustering uses 'method')
    if 'clustering_method' in merged_df.columns and 'method' not in merged_df.columns:
        merged_df['method'] = merged_df['clustering_method']
    
    # Apply filters
    if other_filters:
        for col, values in other_filters.items():
            if not values:  # Skip if no values to filter
                continue
            if col in merged_df.columns:
                # For dr_components, handle float comparison
                if col == 'dr_components':
                    merged_df = merged_df[merged_df[col].isin([float(v) for v in values])]
                else:
                    merged_df = merged_df[merged_df[col].isin(values)]
            # Also check for alternative column names
            elif col == 'method' and 'clustering_method' in merged_df.columns:
                merged_df = merged_df[merged_df['clustering_method'].isin(values)]
            # For panel, also check dataname if panel column doesn't exist
            elif col == 'panel' and 'panel' not in merged_df.columns and 'dataname' in merged_df.columns:
                # Try to filter by dataname pattern if panel info is in dataname
                panel_values = values if isinstance(values, list) else [values]
                panel_mask = pd.Series([False] * len(merged_df), index=merged_df.index)
                for panel_val in panel_values:
                    if panel_val == "all":
                        panel_mask = pd.Series([True] * len(merged_df), index=merged_df.index)
                        break
                    else:
                        # Check if dataname contains the panel info
                        panel_mask |= merged_df['dataname'].str.contains(f"_{panel_val}_", na=False, regex=False) | \
                                     merged_df['dataname'].str.contains(f"panel{panel_val}", na=False, regex=False)
                merged_df = merged_df[panel_mask]
    
    if merged_df.empty:
        return go.Figure()
    
    # Get feature importance columns
    importance_cols = [col for col in merged_df.columns if col.startswith('importance_')]
    if not importance_cols:
        return go.Figure()
    
    # Extract feature names from column names
    feature_names = [col.replace('importance_', '') for col in importance_cols]
    
    # Calculate mean importance for each feature across all runs
    feature_means = {}
    feature_data = {}
    
    for feature_col, feature_name in zip(importance_cols, feature_names):
        if feature_col in merged_df.columns:
            values = merged_df[feature_col].dropna()
            values = values[np.isfinite(values)]
            if len(values) > 0:
                feature_means[feature_name] = float(values.mean())
                feature_data[feature_name] = values.values
    
    if not feature_means:
        return go.Figure()
    
    # Get top k features
    sorted_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_k]]
    
    if not top_features:
        return go.Figure()
    
    # Calculate statistics for each feature
    plot_data = []
    for feature in top_features:
        if feature in feature_data:
            values = feature_data[feature]
            mean_val = float(np.mean(values))
            
            # Calculate 95% CI
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=stats.sem(values))
                lower = float(ci[0])
                upper = float(ci[1])
            else:
                lower = mean_val
                upper = mean_val
            
            plot_data.append({
                'feature': feature,
                'mean': mean_val,
                'lower': lower,
                'upper': upper
            })
    
    if not plot_data:
        return go.Figure()
    
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('mean', ascending=True)
    
    # Create figure
    fig = go.Figure()
    
    x_vals = plot_df['mean'].tolist()
    y_vals = plot_df['feature'].tolist()
    error_plus = (plot_df['upper'] - plot_df['mean']).tolist()
    error_minus = (plot_df['mean'] - plot_df['lower']).tolist()
    
    # Create hover text with mean and CI
    hover_text = [
        f"Feature: {feat}<br>"
        f"Mean: {mean:.4f}<br>"
        f"95% CI: [{lower:.4f}, {upper:.4f}]<br>"
        f"Lower: {lower:.4f}<br>"
        f"Upper: {upper:.4f}"
        for feat, mean, lower, upper in zip(
            plot_df['feature'],
            plot_df['mean'],
            plot_df['lower'],
            plot_df['upper']
        )
    ]
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        name='Overall',
        marker=dict(
            color='#1f77b4',
            size=10,
            line=dict(width=1, color='darkblue')
        ),
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            thickness=2,
            width=3,
            color='#1f77b4'
        ),
        text=hover_text,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Overall Feature Importance<br><sub>Top-{top_k} Features</sub>",
            font=dict(size=18)
        ),
        xaxis_title=dict(text="Feature Importance (95% CI)", font=dict(size=16)),
        yaxis_title=dict(text="Feature", font=dict(size=16)),
        height=max(400, len(top_features) * 30),
        hovermode='closest',
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        showlegend=False
    )
    
    return fig


def create_comparative_plot(
    df: pd.DataFrame,
    metrics: List[str],
    compare_by: str,
    other_filters: dict = None,
    selected_metrics: List[str] = None
) -> go.Figure:
    """
    Create errorbar plot comparing groups for selected metrics from comparative table data.
    
    Args:
        df: Filtered DataFrame with raw data
        metrics: List of available metric column names
        compare_by: Dimension to compare ('method', 'dr_method', 'variant', 'n_clusters', 'dr_components')
        other_filters: Dict of additional filters {column: [values]}
        selected_metrics: List of metrics to plot (default: ['silhouette_score'] and one other if available)
        
    Returns:
        Plotly figure with errorbar plots
    """
    if df.empty:
        return go.Figure()
    
    # Apply additional filters if provided
    filtered_df = df.copy()
    if other_filters:
        for col, values in other_filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    if filtered_df.empty:
        return go.Figure()
    
    # For n_clusters, group values > 10 into ">10"
    if compare_by == 'n_clusters' and 'n_clusters' in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['n_clusters_grouped'] = filtered_df['n_clusters'].apply(
            lambda x: ">10" if pd.notna(x) and x > 10 else x
        )
        compare_by = 'n_clusters_grouped'
    
    # Get unique values for comparison dimension
    def sort_key(x):
        if x == ">10":
            return (1, float('inf'))
        else:
            try:
                return (0, float(x))
            except (ValueError, TypeError):
                return (0, float('inf'))
    
    compare_values = sorted([v for v in filtered_df[compare_by].dropna().unique() if pd.notna(v)], key=sort_key)
    
    if len(compare_values) < 2:
        return go.Figure()  # Need at least 2 groups to compare
    
    # Default to silhouette_score and one other metric if available
    if selected_metrics is None:
        selected_metrics = ['silhouette_score']
        other_metrics = [m for m in metrics if m != 'silhouette_score' and m in filtered_df.columns]
        if other_metrics:
            selected_metrics.append(other_metrics[0])
    
    # Filter to only metrics that exist in the data
    selected_metrics = [m for m in selected_metrics if m in filtered_df.columns and m in metrics]
    
    if not selected_metrics:
        return go.Figure()
    
    # Calculate statistics for each group and each selected metric
    plot_data = []
    for metric in selected_metrics:
        for compare_val in compare_values:
            group_data = filtered_df[filtered_df[compare_by] == compare_val][metric].dropna()
            group_data = group_data[np.isfinite(group_data)]
            
            if len(group_data) > 0:
                mean_val = float(group_data.mean())
                
                # Calculate 95% CI
                if len(group_data) > 1:
                    ci = stats.t.interval(0.95, len(group_data) - 1, loc=mean_val, scale=stats.sem(group_data))
                    lower = float(ci[0])
                    upper = float(ci[1])
                else:
                    lower = mean_val
                    upper = mean_val
                
                plot_data.append({
                    'group': str(compare_val),
                    'metric': METRIC_LABELS.get(metric, metric),
                    'mean': mean_val,
                    'lower': lower,
                    'upper': upper
                })
    
    if not plot_data:
        return go.Figure()
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplots if multiple metrics
    from plotly.subplots import make_subplots
    
    n_metrics = len(selected_metrics)
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in selected_metrics],
        shared_yaxes=False,
        horizontal_spacing=0.15
    )
    
    colors = px.colors.qualitative.Set3[:len(compare_values)]
    
    for col_idx, metric in enumerate(selected_metrics, 1):
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_data = plot_df[plot_df['metric'] == metric_label]
        
        if metric_data.empty:
            continue
        
        for idx, compare_val in enumerate(compare_values):
            group_data = metric_data[metric_data['group'] == str(compare_val)]
            
            if group_data.empty:
                continue
            
            row = group_data.iloc[0]
            error_plus = row['upper'] - row['mean']
            error_minus = row['mean'] - row['lower']
            
            fig.add_trace(
                go.Scatter(
                    x=[row['mean']],
                    y=[compare_val],
                    mode='markers',
                    name=str(compare_val) if col_idx == 1 else '',
                    marker=dict(
                        color=colors[idx % len(colors)],
                        size=10
                    ),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[error_plus],
                        arrayminus=[error_minus]
                    ),
                    showlegend=bool(col_idx == 1)
                ),
                row=1,
                col=col_idx
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Comparison by {compare_by.replace('_', ' ').title()}",
            font=dict(size=18)
        ),
        height=400,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        )
    )
    
    # Update axes
    for col_idx in range(1, n_metrics + 1):
        metric = selected_metrics[col_idx - 1]
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_direction = METRIC_DIRECTIONS.get(metric, "")
        
        fig.update_xaxes(
            title_text=f"{metric_label} {metric_direction}",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            row=1,
            col=col_idx
        )
        # For n_clusters, ensure y-axis shows all values including ">10" in correct order
        if compare_by == 'n_clusters_grouped':
            fig.update_yaxes(
                title_text="Number of Clusters",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                categoryorder='array',
                categoryarray=compare_values,
                row=1,
                col=col_idx
            )
        else:
            fig.update_yaxes(
                title_text=compare_by.replace('_', ' ').title(),
                title_font=dict(size=16),
                tickfont=dict(size=14),
                row=1,
                col=col_idx
            )
    
    return fig


def create_stratified_comparative_plot(
    df: pd.DataFrame,
    metrics: List[str],
    compare_by: str,
    strat_groups: List[str],
    strat_labels: List[str],
    other_filters: dict = None,
    selected_metrics: List[str] = None
) -> go.Figure:
    """
    Create errorbar plot comparing stratified groups for each value of compare_by dimension.
    Shows pairwise comparison (e.g., male vs female) for each clustering method, DR method, etc.
    
    Args:
        df: Filtered DataFrame with raw data
        metrics: List of available metric column names
        compare_by: Dimension to compare ('method', 'dr_method', 'variant', 'n_clusters', 'dr_components')
        strat_groups: List of stratification group values (e.g., ['male', 'female'])
        strat_labels: List of stratification group labels (e.g., ['Male', 'Female'])
        other_filters: Dict of additional filters {column: [values]}
        selected_metrics: List of metrics to plot (default: ['silhouette_score'] and one other if available)
        
    Returns:
        Plotly figure with errorbar plots
    """
    if df.empty or len(strat_groups) != 2:
        return go.Figure()
    
    # Apply additional filters if provided
    filtered_df = df.copy()
    if other_filters:
        for col, values in other_filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    # Filter to only stratified groups
    filtered_df = filtered_df[filtered_df['stratification'].isin(strat_groups)]
    
    if filtered_df.empty:
        return go.Figure()
    
    # For n_clusters, group values > 10 into ">10"
    if compare_by == 'n_clusters' and 'n_clusters' in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['n_clusters_grouped'] = filtered_df['n_clusters'].apply(
            lambda x: ">10" if pd.notna(x) and x > 10 else x
        )
        compare_by = 'n_clusters_grouped'
    
    # Get unique values for comparison dimension
    def sort_key(x):
        if x == ">10":
            return (1, float('inf'))
        else:
            try:
                return (0, float(x))
            except (ValueError, TypeError):
                return (0, float('inf'))
    
    compare_values = sorted([v for v in filtered_df[compare_by].dropna().unique() if pd.notna(v)], key=sort_key)
    
    if len(compare_values) < 1:
        return go.Figure()
    
    # Default to silhouette_score and one other metric if available
    if selected_metrics is None:
        selected_metrics = ['silhouette_score']
        other_metrics = [m for m in metrics if m != 'silhouette_score' and m in filtered_df.columns]
        if other_metrics:
            selected_metrics.append(other_metrics[0])
    
    # Filter to only metrics that exist in the data
    selected_metrics = [m for m in selected_metrics if m in filtered_df.columns and m in metrics]
    
    if not selected_metrics:
        return go.Figure()
    
    # Calculate statistics for each group, each compare value, and each selected metric
    plot_data = []
    for metric in selected_metrics:
        for compare_val in compare_values:
            for strat_group, strat_label in zip(strat_groups, strat_labels):
                group_data = filtered_df[
                    (filtered_df[compare_by] == compare_val) & 
                    (filtered_df['stratification'] == strat_group)
                ][metric].dropna()
                group_data = group_data[np.isfinite(group_data)]
                
                if len(group_data) > 0:
                    mean_val = float(group_data.mean())
                    
                    # Calculate 95% CI
                    if len(group_data) > 1:
                        ci = stats.t.interval(0.95, len(group_data) - 1, loc=mean_val, scale=stats.sem(group_data))
                        lower = float(ci[0])
                        upper = float(ci[1])
                    else:
                        lower = mean_val
                        upper = mean_val
                    
                    plot_data.append({
                        'compare_val': str(compare_val),
                        'group': strat_label,
                        'metric': METRIC_LABELS.get(metric, metric),
                        'mean': mean_val,
                        'lower': lower,
                        'upper': upper
                    })
    
    if not plot_data:
        return go.Figure()
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplots if multiple metrics
    from plotly.subplots import make_subplots
    
    n_metrics = len(selected_metrics)
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in selected_metrics],
        shared_yaxes=False,
        horizontal_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange for two groups
    
    for col_idx, metric in enumerate(selected_metrics, 1):
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_data = plot_df[plot_df['metric'] == metric_label]
        
        if metric_data.empty:
            continue
        
        for compare_val in compare_values:
            for idx, (strat_group, strat_label) in enumerate(zip(strat_groups, strat_labels)):
                group_data = metric_data[
                    (metric_data['compare_val'] == str(compare_val)) & 
                    (metric_data['group'] == strat_label)
                ]
                
                if group_data.empty:
                    continue
                
                row = group_data.iloc[0]
                error_plus = row['upper'] - row['mean']
                error_minus = row['mean'] - row['lower']
                
                # Create a combined label for y-axis
                y_pos = f"{compare_val}"
                
                # Offset x position slightly for each group to avoid overlap
                x_offset = 0.02 if idx == 0 else -0.02
                x_pos = row['mean'] * (1 + x_offset) if row['mean'] != 0 else x_offset
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers',
                        name=strat_label if col_idx == 1 and compare_val == compare_values[0] else '',
                        marker=dict(
                            color=colors[idx % len(colors)],
                            size=10
                        ),
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[error_plus],
                            arrayminus=[error_minus]
                        ),
                        showlegend=bool(col_idx == 1 and compare_val == compare_values[0])
                    ),
                    row=1,
                    col=col_idx
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Pairwise Comparison by {compare_by.replace('_', ' ').title()} ({strat_labels[0]} vs {strat_labels[1]})",
            font=dict(size=18)
        ),
        height=max(400, len(compare_values) * 60),
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        )
    )
    
    # Update axes
    for col_idx in range(1, n_metrics + 1):
        metric = selected_metrics[col_idx - 1]
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_direction = METRIC_DIRECTIONS.get(metric, "")
        
        fig.update_xaxes(
            title_text=f"{metric_label} {metric_direction}",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            row=1,
            col=col_idx
        )
        # For n_clusters, ensure y-axis shows all values including ">10" in correct order
        if compare_by == 'n_clusters_grouped':
            fig.update_yaxes(
                title_text="Number of Clusters",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                categoryorder='array',
                categoryarray=compare_values,
                row=1,
                col=col_idx
            )
        else:
            fig.update_yaxes(
                title_text=f"{compare_by.replace('_', ' ').title()}",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                row=1,
                col=col_idx
            )
    
    return fig


def create_comparative_table(
    df: pd.DataFrame,
    metrics: List[str],  # List of metric column names
    compare_by: str,  # 'method', 'dr_method', 'variant', 'stratification'
    other_filters: dict = None  # Additional filters to apply
) -> pd.DataFrame:
    """
    Create a comparative table comparing groups by a specific dimension.
    Shows one row per group with all metrics as columns, and p-value row at the end.
    
    Args:
        df: Filtered DataFrame
        metrics: List of metric column names to include
        compare_by: Dimension to compare ('method', 'dr_method', 'variant', 'stratification')
        other_filters: Dict of additional filters {column: [values]}
        
    Returns:
        DataFrame with comparison statistics (one row per group + p-value row)
    """
    # Apply additional filters if provided
    filtered_df = df.copy()
    if other_filters:
        for col, values in other_filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # For n_clusters, group values > 10 into ">10"
    if compare_by == 'n_clusters' and 'n_clusters' in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['n_clusters_grouped'] = filtered_df['n_clusters'].apply(
            lambda x: ">10" if pd.notna(x) and x > 10 else x
        )
        compare_by = 'n_clusters_grouped'
    
    # Get unique values for comparison dimension
    # Sort: numeric values first (ascending), then ">10" at the end
    def sort_key(x):
        if x == ">10":
            return (1, float('inf'))  # Put ">10" at the end
        else:
            try:
                return (0, float(x))  # Sort numeric values
            except (ValueError, TypeError):
                return (0, float('inf'))  # Fallback for non-numeric
    
    compare_values = sorted([v for v in filtered_df[compare_by].dropna().unique() if pd.notna(v)], key=sort_key)
    
    if len(compare_values) < 2:
        return pd.DataFrame()  # Need at least 2 groups to compare
    
    # Map compare_by to readable column names
    column_name_map = {
        'method': 'Clustering Method',
        'dr_method': 'DR Method',
        'variant': 'Variant',
        'stratification': 'Stratification',
        'n_clusters': 'Number of Clusters',
        'n_clusters_grouped': 'Number of Clusters',
        'dr_components': 'N Dim'
    }
    column_name = column_name_map.get(compare_by, compare_by.title())
    
    # Calculate statistics for each group and each metric
    all_group_stats = {}  # {compare_val: {metric: stats}}
    all_group_data = {}   # {metric: {compare_val: data}}
    
    for metric in metrics:
        if metric not in filtered_df.columns:
            continue
        
        all_group_data[metric] = {}
        
        for compare_val in compare_values:
            group_data = filtered_df[filtered_df[compare_by] == compare_val][metric].dropna()
            
            # Filter out infinite values and NaN
            group_data = group_data[np.isfinite(group_data)]
            
            if len(group_data) > 0:
                mean_val = float(group_data.mean())
                
                # Calculate 95% CI
                if len(group_data) > 1:
                    ci = stats.t.interval(
                        0.95,
                        len(group_data) - 1,
                        loc=mean_val,
                        scale=stats.sem(group_data)
                    )
                    lower = float(ci[0])
                    upper = float(ci[1])
                else:
                    lower = mean_val
                    upper = mean_val
                
                if compare_val not in all_group_stats:
                    all_group_stats[compare_val] = {}
                
                all_group_stats[compare_val][metric] = {
                    'mean': mean_val,
                    'lower': lower,
                    'upper': upper,
                    'n': len(group_data)
                }
                all_group_data[metric][compare_val] = group_data
    
    if not all_group_stats:
        return pd.DataFrame()
    
    # Create one row per group
    results = []
    for compare_val in compare_values:
        if compare_val not in all_group_stats:
            continue
        
        row = {column_name: compare_val}
        
        # Add columns for each metric
        for metric in metrics:
            metric_abbr = METRIC_ABBREVIATIONS.get(metric, metric)
            if metric not in all_group_stats[compare_val]:
                row[f"{metric_abbr} (N)"] = "N/A"
                row[metric_abbr] = "N/A"
            else:
                stats_dict = all_group_stats[compare_val][metric]
                format_str = ".0f" if metric == 'calinski_harabasz_index' else ".1f"
                row[f"{metric_abbr} (N)"] = stats_dict['n']
                row[metric_abbr] = f"{stats_dict['mean']:{format_str}} ({stats_dict['lower']:{format_str}}; {stats_dict['upper']:{format_str}})"
        
        results.append(row)
    
    # Add p-value row at the end
    p_value_row = {column_name: "p-value"}
    for metric in metrics:
        metric_abbr = METRIC_ABBREVIATIONS.get(metric, metric)
        if metric in all_group_data and len(all_group_data[metric]) >= 2:
            # Calculate overall p-value comparing all groups for this metric
            overall_p_value = calculate_all_groups_p_value(all_group_data[metric])
            
            # Format p-value
            if np.isnan(overall_p_value):
                p_value_str = "N/A"
            elif overall_p_value < 0.05:
                p_value_str = "<0.05"
            else:
                p_value_str = f"{overall_p_value:.4f}"
            
            p_value_row[f"{metric_abbr} (N)"] = ""
            p_value_row[metric_abbr] = p_value_str
        else:
            p_value_row[f"{metric_abbr} (N)"] = ""
            p_value_row[metric_abbr] = "N/A"
    
    results.append(p_value_row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Convert all columns to strings to avoid Arrow serialization issues
    # This ensures consistent types for Streamlit display
    for col in result_df.columns:
        result_df[col] = result_df[col].astype(str)
    
    return result_df


def create_heatmap_table(
    df: pd.DataFrame,
    metric: str,
    clustering_methods: List[str],
    dr_methods: List[str],
    variant_order: List[str],
    stratification_selection: str = "overall"
) -> go.Figure:
    """
    Create a heatmap table with statistics and jet colormap.
    
    Args:
        df: Filtered DataFrame
        metric: Metric column name
        clustering_methods: List of clustering methods
        dr_methods: List of DR methods
        variant_order: Order of variants
        
    Returns:
        Plotly Figure with heatmap table
    """
    # Calculate statistics
    stats_df = calculate_statistics_table(df, metric, clustering_methods, dr_methods, variant_order, stratification_selection)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for statistics table",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract mean values for coloring (from the formatted strings)
    # We'll need to recalculate means for coloring
    mean_values = {}
    for cluster_method in clustering_methods:
        for dr_method in dr_methods:
            if stratification_selection != "overall":
                # For stratified views, include stratification in the key
                if stratification_selection == "sex-stratified":
                    strat_groups = ['male', 'female']
                else:  # age-stratified
                    strat_groups = ['old', 'young']
                
                for strat_group in strat_groups:
                    key = (cluster_method, dr_method, strat_group)
                    mean_values[key] = {}
                    
                    for variant in variant_order:
                        variant_data = df[
                            (df['method'] == cluster_method) &
                            (df['dr_method'] == dr_method) &
                            (df['variant'] == variant) &
                            (df['stratification'] == strat_group)
                        ][metric].dropna()
                        
                        if len(variant_data) > 0:
                            mean_values[key][variant] = float(variant_data.mean())
                        else:
                            mean_values[key][variant] = np.nan
            else:
                # Overall view
                key = (cluster_method, dr_method)
                mean_values[key] = {}
                
                for variant in variant_order:
                    variant_data = df[
                        (df['method'] == cluster_method) &
                        (df['dr_method'] == dr_method) &
                        (df['variant'] == variant)
                    ][metric].dropna()
                    
                    if len(variant_data) > 0:
                        mean_values[key][variant] = float(variant_data.mean())
                    else:
                        mean_values[key][variant] = np.nan
    
    # Create matrix of mean values for heatmap
    heatmap_data = []
    row_keys = []  # Store keys in same order as rows
    for _, row in stats_df.iterrows():
        cluster = row['Clustering']
        dr = row['DR']
        if stratification_selection != "overall":
            strat_label = row.get('Stratification', '')
            # Map label back to group
            if stratification_selection == "sex-stratified":
                strat_map = {'Male': 'male', 'Female': 'female'}
            else:  # age-stratified
                strat_map = {'Old': 'old', 'Young': 'young'}
            strat_group = strat_map.get(strat_label, '')
            key = (cluster, dr, strat_group)
        else:
            key = (cluster, dr)
        row_keys.append(key)
        row_means = [mean_values.get(key, {}).get(v, np.nan) for v in variant_order]
        heatmap_data.append(row_means)
    
    heatmap_matrix = np.array(heatmap_data)
    
    # Get min/max for colorbar (excluding NaN)
    valid_values = heatmap_matrix[~np.isnan(heatmap_matrix)]
    if len(valid_values) > 0:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())
    else:
        vmin = 0
        vmax = 1
    
    # Normalize values for jet colormap (0-1 range)
    normalized = (heatmap_matrix - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(heatmap_matrix)
    normalized = np.clip(normalized, 0, 1)
    
    # Apply jet colormap (handle both old and new matplotlib APIs)
    try:
        import matplotlib
        import matplotlib.cm as cm
        try:
            jet_colormap = matplotlib.colormaps['jet']
        except (AttributeError, KeyError):
            # Fallback for older matplotlib versions
            jet_colormap = cm.get_cmap('jet')
    except ImportError:
        # Fallback if matplotlib not available - use a simple colormap
        def simple_jet(x):
            # Simple jet-like colormap approximation
            # Handle both scalar and array inputs
            x_arr = np.asarray(x)
            x_clipped = np.clip(x_arr, 0, 1)
            r = np.clip(1.5 - 4 * np.abs(x_clipped - 0.75), 0, 1)
            g = np.clip(1.5 - 4 * np.abs(x_clipped - 0.5), 0, 1)
            b = np.clip(1.5 - 4 * np.abs(x_clipped - 0.25), 0, 1)
            result = np.stack([r, g, b, np.ones_like(x_clipped)], axis=-1 if x_arr.ndim > 0 else -1)
            # Return scalar if input was scalar
            if np.isscalar(x):
                return result.flatten()[:4] if result.ndim > 1 else result
            return result
        jet_colormap = simple_jet
    
    # Apply colormap - handle 2D array properly
    rgba_colors = []
    for i in range(heatmap_matrix.shape[0]):
        rgba_row = []
        for j in range(heatmap_matrix.shape[1]):
            if np.isnan(heatmap_matrix[i, j]):
                rgba_row.append('rgba(200,200,200,0.3)')  # Gray for NaN
            else:
                norm_val = normalized[i, j]
                if callable(jet_colormap):
                    if hasattr(jet_colormap, '__call__'):
                        # Matplotlib colormap
                        rgba = jet_colormap(norm_val)
                    else:
                        # Fallback function
                        rgba = jet_colormap(norm_val)
                else:
                    rgba = jet_colormap(norm_val)
                
                # Convert to rgba string
                if isinstance(rgba, np.ndarray):
                    if len(rgba) >= 4:
                        rgba_row.append(f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})')
                    elif len(rgba) == 3:
                        rgba_row.append(f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},1.0)')
                    else:
                        rgba_row.append('white')
                elif isinstance(rgba, (list, tuple)) and len(rgba) >= 3:
                    rgba_row.append(f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},1.0)')
                else:
                    rgba_row.append('white')
        rgba_colors.append(rgba_row)
    
    # Create table header
    if stratification_selection != "overall":
        header_values = ['Clustering', 'DR', 'Stratification'] + variant_order
    else:
        header_values = ['Clustering', 'DR'] + variant_order
    cell_values = []
    cell_colors = []
    
    for row_idx, (_, row) in enumerate(stats_df.iterrows()):
        if stratification_selection != "overall":
            cell_row = [row['Clustering'], row['DR'], row.get('Stratification', '')] + [row[v] for v in variant_order]
            # Colors: white for first three columns (Clustering, DR, Stratification), heatmap for variant columns
            color_row = ['white', 'white', 'white'] + rgba_colors[row_idx]
        else:
            cell_row = [row['Clustering'], row['DR']] + [row[v] for v in variant_order]
            # Colors: white for first two columns, heatmap for variant columns
            color_row = ['white', 'white'] + rgba_colors[row_idx]
        cell_values.append(cell_row)
        cell_colors.append(color_row)
    
    # Transpose for table format (Plotly expects columns as lists)
    cell_values_t = list(map(list, zip(*cell_values)))
    cell_colors_t = list(map(list, zip(*cell_colors)))
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color='lightgray',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=cell_values_t,
            fill_color=cell_colors_t,
            align='center',
            font=dict(size=11, color='black'),
            height=30
        )
    )])
    
    # Create colorbar using a separate figure
    # We'll add it as a subplot or annotation
    # For now, add annotation with color scale info
    direction = METRIC_DIRECTIONS.get(metric, '')
    format_str = ".0f" if metric == 'calinski_harabasz_index' else ".3f"
    
    fig.add_annotation(
        text=f"<b>Color Scale</b><br>({METRIC_LABELS[metric]})<br>Min: {vmin:{format_str}}<br>Max: {vmax:{format_str}}<br><br>Jet Colormap",
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        xanchor="left", yanchor="middle",
        showarrow=False,
        bgcolor='white',
        bordercolor='gray',
        borderwidth=1,
        borderpad=5,
        font=dict(size=10)
    )
    
    # Create a simple colorbar visualization using shapes
    # Add gradient rectangles to show the colorbar
    n_steps = 20
    for i in range(n_steps):
        val = vmin + (vmax - vmin) * (i / (n_steps - 1))
        normalized_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        
        if callable(jet_colormap):
            color = jet_colormap(normalized_val)
        else:
            color = jet_colormap(normalized_val)
        
        # Convert color to rgba string
        if isinstance(color, np.ndarray):
            if len(color) >= 4:
                rgba_str = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
            elif len(color) >= 3:
                rgba_str = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1.0)'
            else:
                rgba_str = 'gray'
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            rgba_str = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1.0)'
        else:
            rgba_str = 'gray'
        
        # Add rectangle for colorbar
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=1.15, y0=0.3 + (i / n_steps) * 0.4,
            x1=1.17, y1=0.3 + ((i + 1) / n_steps) * 0.4,
            fillcolor=rgba_str,
            line=dict(width=0)
        )
    
    # Add colorbar labels
    fig.add_annotation(
        text=f"{vmax:{format_str}}",
        xref="paper", yref="paper",
        x=1.18, y=0.7,
        xanchor="left", yanchor="middle",
        showarrow=False,
        font=dict(size=9)
    )
    fig.add_annotation(
        text=f"{vmin:{format_str}}",
        xref="paper", yref="paper",
        x=1.18, y=0.3,
        xanchor="left", yanchor="middle",
        showarrow=False,
        font=dict(size=9)
    )
    
    fig.update_layout(
        title=dict(
            text=f"Statistics Table: {METRIC_LABELS[metric]} (Mean with 95% CI)",
            font=dict(size=18)
        ),
        height=max(400, len(stats_df) * 35 + 100),
        margin=dict(l=20, r=200, t=50, b=20)
    )
    
    return fig


def apply_filters(
    df: pd.DataFrame,
    data_selection: str,
    selected_clusters: List[int],
    selected_dims: List[int],
    stratification_selection: str = "overall"
) -> pd.DataFrame:
    """
    Apply filters to DataFrame.
    
    Args:
        df: Input DataFrame
        data_selection: Data selection ("all", "12", "13", "23", "rz", or "all" for all)
        selected_clusters: List of n_clusters values to include (empty = all)
        selected_dims: List of dr_components values to include (empty = all)
        stratification_selection: Stratification selection ("overall", "sex-stratified", "age-stratified")
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by panel
    if data_selection != "all":
        filtered_df = filtered_df[filtered_df['panel'] == data_selection]
    else:
        # For "all", include all panels (including None)
        pass
    
    # Filter by stratification
    if stratification_selection == "overall":
        # Only include rows where stratification is 'overall' (exclude male, female, old, young)
        filtered_df = filtered_df[filtered_df['stratification'] == 'overall']
    elif stratification_selection == "sex-stratified":
        # Include only male and female
        filtered_df = filtered_df[filtered_df['stratification'].isin(['male', 'female'])]
    elif stratification_selection == "age-stratified":
        # Include only old and young
        filtered_df = filtered_df[filtered_df['stratification'].isin(['old', 'young'])]
    
    # Filter by n_clusters
    if selected_clusters:
        filtered_df = filtered_df[filtered_df['n_clusters'].isin(selected_clusters)]
    
    # Filter by dr_components
    if selected_dims:
        # Convert to float for comparison (dr_components might be float)
        filtered_df = filtered_df[filtered_df['dr_components'].isin([float(d) for d in selected_dims])]
    
    return filtered_df


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Clustering Metrics Visualization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Interactive Clustering Metrics Visualization")
    st.markdown("Explore clustering metrics across different variants, methods, and configurations")
    
    # Sidebar controls
    st.sidebar.header("Filters")
    
    # Results path input
    # Default to parent directory's results folder when running from PROMIS_Dashboard
    from pathlib import Path
    parent_results = Path(__file__).parent.parent / "results"
    default_results = str(parent_results) if parent_results.exists() else "../results/"
    results_path = st.sidebar.text_input(
        "Results Path",
        value=default_results,
        help="Path to the results directory containing clustering CSV files"
    )
    
    # Check for optimized data file first (for lightweight deployment)
    optimized_data_path = Path(__file__).parent / "results" / "dashboard_data.parquet"
    use_optimized = optimized_data_path.exists()
    
    if use_optimized:
        st.sidebar.info("📦 Using optimized data file (fast loading)")
        if st.sidebar.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Using raw data loading (slower). Run prepare_dashboard_data.py to optimize.")
    
    # Load data (using cached functions for performance)
    cache_key = f"df_{results_path}_{use_optimized}"
    if cache_key not in st.session_state or (not use_optimized and st.sidebar.button("Reload Data")):
        # Show loading indicator
        with st.spinner("Loading data (this may take a moment on first load)..."):
            try:
                if use_optimized:
                    # Load from optimized Parquet file (fast!)
                    df = pd.read_parquet(optimized_data_path)
                    st.success(f"✓ Successfully loaded {len(df):,} rows from optimized file")
                else:
                    # Load from raw CSV files (slower, but works with original data)
                    combined_df = load_all_results(results_path)
                    
                    if combined_df.empty:
                        st.error("No data loaded. Please check the results path.")
                        st.stop()
                    
                    # Use cached prepare_data function
                    df = prepare_data(combined_df)
                    st.success(f"✓ Successfully loaded {len(df):,} rows from raw data")
                
                st.session_state[cache_key] = df
                st.session_state.df = df  # Keep for backward compatibility
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    else:
        # Use cached data from session state
        df = st.session_state[cache_key]
        st.session_state.df = df  # Keep for backward compatibility
    
    if df.empty:
        st.error("No data available. Please check the results path.")
        st.stop()
    
    # Get available options
    available_panels = sorted([p for p in df['panel'].dropna().unique() if p in ['all', '12', '13', '23', 'rz']])
    available_clusters = sorted([int(c) for c in df['n_clusters'].dropna().unique() if pd.notna(c)])
    available_dims = sorted([int(d) for d in df['dr_components'].dropna().unique() if pd.notna(d)])
    
    # Data selection dropdown
    data_options = ["all"] + available_panels
    # Set default to "rz" if available, otherwise "all"
    default_index = 0
    if "rz" in available_panels:
        default_index = data_options.index("rz")
    data_selection = st.sidebar.selectbox(
        "Data Selection",
        options=data_options,
        index=default_index,
        help="Select which data panel to include (all, 12, 13, 23, or rz)"
    )
    
    # Stratification selection dropdown
    available_stratifications = sorted([s for s in df['stratification'].dropna().unique() if pd.notna(s)])
    stratification_options = ["overall", "sex-stratified", "age-stratified"]
    # Only show options that are available in the data
    available_strat_options = [opt for opt in stratification_options if 
                              (opt == "overall" and "overall" in available_stratifications) or
                              (opt == "sex-stratified" and any(s in available_stratifications for s in ["male", "female"])) or
                              (opt == "age-stratified" and any(s in available_stratifications for s in ["old", "young"]))]
    
    if not available_strat_options:
        available_strat_options = ["overall"]  # Default fallback
    
    stratification_selection = st.sidebar.selectbox(
        "Stratification",
        options=available_strat_options,
        index=0,
        help="Select stratification: overall (no stratification), sex-stratified (male vs female), or age-stratified (old vs young)"
    )
    
    # Number of clusters multi-select
    selected_clusters = st.sidebar.multiselect(
        "Number of Clusters",
        options=available_clusters,
        default=[],
        help="Select which number of clusters to include (empty = all)"
    )
    
    # N dimensions multi-select
    selected_dims = st.sidebar.multiselect(
        "N Dimensions (DR Components)",
        options=available_dims,
        default=[],
        help="Select which number of dimensions to include (empty = all)"
    )
    
    # Get available methods from filtered data (before method filtering)
    # Apply basic filters first to get available methods
    temp_filtered = df.copy()
    if data_selection != "all":
        temp_filtered = temp_filtered[temp_filtered['panel'] == data_selection]
    if stratification_selection == "overall":
        temp_filtered = temp_filtered[temp_filtered['stratification'] == 'overall']
    elif stratification_selection == "sex-stratified":
        temp_filtered = temp_filtered[temp_filtered['stratification'].isin(['male', 'female'])]
    elif stratification_selection == "age-stratified":
        temp_filtered = temp_filtered[temp_filtered['stratification'].isin(['old', 'young'])]
    
    available_clustering_methods = sorted([m for m in temp_filtered['method'].dropna().unique() if pd.notna(m)])
    available_dr_methods = sorted([d for d in temp_filtered['dr_method'].dropna().unique() if pd.notna(d)])
    
    # Clustering methods multi-select
    selected_clustering_methods = st.sidebar.multiselect(
        "Clustering Methods",
        options=available_clustering_methods,
        default=[],
        help="Select which clustering methods to include (empty = all)"
    )
    
    # DR methods multi-select
    selected_dr_methods = st.sidebar.multiselect(
        "DR Methods",
        options=available_dr_methods,
        default=[],
        help="Select which dimensionality reduction methods to include (empty = all)"
    )
    
    # Metric selection (for performance - show only one at a time)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Visualization Options")
    available_metrics = [m for m in METRICS if m in df.columns and not df[m].isna().all()]
    metric_labels_list = [METRIC_LABELS[m] for m in available_metrics]
    
    # Find index of silhouette_score (default metric)
    default_metric = 'silhouette_score'
    default_index = 0
    if default_metric in available_metrics:
        default_index = available_metrics.index(default_metric)
    
    selected_metric_label = st.sidebar.selectbox(
        "Select Metric to Display",
        options=metric_labels_list,
        index=default_index if metric_labels_list else None,
        help="Select which metric to visualize (showing one at a time improves browser performance)"
    )
    
    # Get the metric key from the label
    selected_metric = None
    for metric_key, metric_label in METRIC_LABELS.items():
        if metric_label == selected_metric_label:
            selected_metric = metric_key
            break
    
    # Plot type selection
    plot_type = st.sidebar.selectbox(
        "Plot Type",
        options=["Box Plot", "Violin Plot", "Scatter Plot"],
        index=0,  # Box Plot as default
        help="Select the type of plot to display"
    )
    
    # Max points per trace setting (for performance)
    max_points = st.sidebar.slider(
        "Max Points per Trace",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Limit number of data points per trace to improve browser performance. Lower = faster rendering."
    )
    
    # Apply filters
    filtered_df = apply_filters(df, data_selection, selected_clusters, selected_dims, stratification_selection)
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        st.stop()
    
    # Y-axis controls (only show if metric is selected)
    if selected_metric and selected_metric in filtered_df.columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Y-Axis Settings")
        
        # Get metric values for computing defaults
        metric_values = filtered_df[selected_metric].dropna()
        
        if len(metric_values) > 0:
            # Get threshold defaults for this metric
            threshold_defaults = METRIC_THRESHOLDS.get(selected_metric, {'min': None, 'max': None})
            
            # Display and allow editing of default thresholds
            st.sidebar.markdown("**Default Thresholds (editable):**")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                threshold_min_default = threshold_defaults.get('min')
                if threshold_min_default is None:
                    # For WSS and metrics without min threshold, use 1st percentile
                    threshold_min_default = float(metric_values.quantile(0.01))
                    threshold_min_label = "Min (1st percentile)"
                else:
                    threshold_min_label = "Min Threshold"
                
                threshold_min = st.sidebar.number_input(
                    threshold_min_label,
                    value=float(threshold_min_default),
                    step=0.01 if selected_metric == 'silhouette_score' else (100.0 if selected_metric == 'calinski_harabasz_index' else 1000.0),
                    format="%.2f" if selected_metric == 'silhouette_score' else "%.0f",
                    help=f"Minimum threshold for {METRIC_LABELS[selected_metric]}"
                )
            
            with col2:
                threshold_max_default = threshold_defaults.get('max')
                if threshold_max_default is None:
                    # For metrics without max threshold, use 99th percentile
                    threshold_max_default = float(metric_values.quantile(0.99))
                    threshold_max_label = "Max (99th percentile)"
                else:
                    threshold_max_label = "Max Threshold"
                
                threshold_max = st.sidebar.number_input(
                    threshold_max_label,
                    value=float(threshold_max_default),
                    step=0.01 if selected_metric == 'davies_bouldin_index' else (100.0 if selected_metric == 'calinski_harabasz_index' else 1000.0),
                    format="%.2f" if selected_metric == 'davies_bouldin_index' else "%.0f",
                    help=f"Maximum threshold for {METRIC_LABELS[selected_metric]}"
                )
            
            # Use threshold values as y-axis defaults (user can override)
            # For WSS, use threshold_max as default y_max
            if selected_metric == 'wss':
                y_max_default = threshold_max
                y_min_default = threshold_min
            else:
                # For other metrics, use thresholds if available, otherwise use percentiles
                if threshold_min_default is not None:
                    y_min_default = threshold_min
                else:
                    y_min_default = float(metric_values.quantile(0.01))
                
                if threshold_max_default is not None:
                    y_max_default = threshold_max
                else:
                    y_max_default = float(metric_values.quantile(0.99))
            
            # Get actual min/max for slider bounds
            y_min_actual = float(metric_values.min())
            y_max_actual = float(metric_values.max())
            
            # Expand bounds slightly for user adjustment
            # Ensure bounds are finite and within reasonable limits
            y_range = y_max_actual - y_min_actual
            if y_range > 0 and np.isfinite(y_range):
                y_min_bound = max(y_min_actual - y_range * 0.1, -1e10)  # Prevent -inf
                y_max_bound = min(y_max_actual + y_range * 0.1, 1e10)   # Prevent +inf
            else:
                # Fallback if range is invalid
                y_min_bound = max(y_min_actual - abs(y_min_actual) * 0.1, -1e10) if np.isfinite(y_min_actual) else -1e10
                y_max_bound = min(y_max_actual + abs(y_max_actual) * 0.1, 1e10) if np.isfinite(y_max_actual) else 1e10
            
            # Ensure bounds are finite
            y_min_bound = max(y_min_bound, -1e10) if np.isfinite(y_min_bound) else -1e10
            y_max_bound = min(y_max_bound, 1e10) if np.isfinite(y_max_bound) else 1e10
            
            # Y-axis limits (use threshold values as defaults)
            st.sidebar.markdown("**Y-Axis Limits:**")
            use_custom_ylim = st.sidebar.checkbox(
                "Set Custom Y-Axis Limits",
                value=True,  # Default to enabled since we have threshold values
                help="Enable to set custom y-axis limits based on thresholds"
            )
            
            if use_custom_ylim:
                # Calculate step size safely
                if y_range > 0 and np.isfinite(y_range):
                    step_size = max(float(y_range) / 1000, 1e-6)  # Ensure minimum step
                else:
                    step_size = 0.1
                
                # Ensure bounds are within Streamlit's acceptable range
                y_min_bound_safe = max(float(y_min_bound), -1.797e308) if np.isfinite(y_min_bound) else -1.797e308
                y_max_bound_safe = min(float(y_max_bound), 1.797e308) if np.isfinite(y_max_bound) else 1.797e308
                
                y_min = st.sidebar.number_input(
                    "Y-Axis Minimum",
                    value=float(y_min_default),
                    min_value=y_min_bound_safe,
                    max_value=y_max_bound_safe,
                    step=step_size,
                    format="%.4f",
                    help="Minimum value for y-axis (defaults to threshold or percentile)"
                )
                
                y_max = st.sidebar.number_input(
                    "Y-Axis Maximum",
                    value=float(y_max_default),
                    min_value=y_min_bound_safe,
                    max_value=y_max_bound_safe,
                    step=step_size,
                    format="%.4f",
                    help="Maximum value for y-axis (defaults to threshold or percentile)"
                )
                
                # Validate min < max
                if y_min >= y_max:
                    st.sidebar.warning("⚠️ Y-axis minimum must be less than maximum. Using defaults.")
                    y_min = y_min_default
                    y_max = y_max_default
            else:
                y_min = None
                y_max = None
            
            # Log scale option
            y_log_scale = st.sidebar.checkbox(
                "Use Log Scale for Y-Axis",
                value=False,
                help="Use logarithmic scale for y-axis (useful for metrics with wide ranges)"
            )
            
            # Warn if log scale with negative/zero values
            if y_log_scale and (metric_values <= 0).any():
                st.sidebar.warning("⚠️ Log scale requires positive values. Some points may not be visible.")
        else:
            y_min = None
            y_max = None
            y_log_scale = False
    else:
        y_min = None
        y_max = None
        y_log_scale = False
    
    # Apply method filters to filtered_df
    if selected_clustering_methods:
        filtered_df = filtered_df[filtered_df['method'].isin(selected_clustering_methods)]
    if selected_dr_methods:
        filtered_df = filtered_df[filtered_df['dr_method'].isin(selected_dr_methods)]
    
    # Display summary statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Summary")
    st.sidebar.write(f"**Total rows:** {len(filtered_df)}")
    st.sidebar.write(f"**Variants:** {', '.join(filtered_df['variant'].unique())}")
    st.sidebar.write(f"**Clustering methods:** {len(filtered_df['method'].unique())}")
    st.sidebar.write(f"**DR methods:** {len(filtered_df['dr_method'].unique())}")
    
    # Get unique methods for plotting (after filtering)
    all_clustering_methods = sorted(filtered_df['method'].dropna().unique())
    all_dr_methods = sorted(filtered_df['dr_method'].dropna().unique())
    
    # Get variant order (only include variants present in filtered data)
    present_variants = [v for v in VARIANT_ORDER if v in filtered_df['variant'].values]
    
    # Create visualization for selected metric only (for browser performance)
    if selected_metric and selected_metric in filtered_df.columns:
        # Check if metric has any non-null values
        if not filtered_df[selected_metric].isna().all():
            st.subheader(METRIC_LABELS[selected_metric])
            
            # Use methods from sidebar (already filtered in filtered_df)
            # If no methods selected in sidebar, use all available
            plot_clustering_methods = selected_clustering_methods if selected_clustering_methods else all_clustering_methods
            plot_dr_methods = selected_dr_methods if selected_dr_methods else all_dr_methods
            
            # Filter the dataframe based on sidebar selections (already done, but ensure consistency)
            filtered_plot_df = filtered_df[
                (filtered_df['method'].isin(plot_clustering_methods)) &
                (filtered_df['dr_method'].isin(plot_dr_methods))
            ]
            
            # Create and display plot first
            if not filtered_plot_df.empty:
                # Handle stratified views
                if stratification_selection in ["sex-stratified", "age-stratified"]:
                    # Determine stratification groups
                    if stratification_selection == "sex-stratified":
                        strat_groups = ['male', 'female']
                        strat_labels = ['Male', 'Female']
                    else:  # age-stratified
                        strat_groups = ['old', 'young']
                        strat_labels = ['Old', 'Young']
                    
                    # Create stacked plots (one below the other)
                    for strat_idx, (strat_group, strat_label) in enumerate(zip(strat_groups, strat_labels)):
                        strat_df = filtered_plot_df[filtered_plot_df['stratification'] == strat_group]
                        
                        if not strat_df.empty:
                            # Add clear header with stratification info
                            st.markdown(f"### {strat_label} ({stratification_selection.replace('-', ' ').title()})")
                            st.markdown(f"*Showing results for {strat_label.lower()} patients*")
                            
                            # Create plot for this stratification group
                            if plot_type == "Scatter Plot":
                                fig = create_scatter_plot(
                                    strat_df,
                                    selected_metric,
                                    present_variants,
                                    plot_clustering_methods,
                                    plot_dr_methods,
                                    max_points_per_trace=max_points,
                                    y_min=y_min,
                                    y_max=y_max,
                                    y_log_scale=y_log_scale
                                )
                            elif plot_type == "Violin Plot":
                                fig = create_violin_plot(
                                    strat_df,
                                    selected_metric,
                                    present_variants,
                                    plot_clustering_methods,
                                    plot_dr_methods,
                                    y_min=y_min,
                                    y_max=y_max,
                                    y_log_scale=y_log_scale
                                )
                            else:  # Box Plot
                                fig = create_boxen_plot(
                                    strat_df,
                                    selected_metric,
                                    present_variants,
                                    plot_clustering_methods,
                                    plot_dr_methods,
                                    y_min=y_min,
                                    y_max=y_max,
                                    y_log_scale=y_log_scale
                                )
                            
                            # Update title to include stratification info
                            current_title = ""
                            if hasattr(fig.layout, 'title'):
                                if isinstance(fig.layout.title, dict):
                                    current_title = fig.layout.title.get('text', '')
                                elif hasattr(fig.layout.title, 'text'):
                                    current_title = fig.layout.title.text
                            
                            if current_title:
                                # Remove existing stratification suffix if present
                                current_title = current_title.split(' - ')[0]
                                fig.update_layout(
                                    title=dict(
                                        text=f"{current_title} - {strat_label}",
                                        x=0.5,
                                        xanchor='center',
                                        y=0.98,
                                        yanchor='top'
                                    )
                                )
                            
                            # Display plot
                            st.plotly_chart(
                                fig,
                                width='stretch',
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': f'{selected_metric}_{strat_group}_{plot_type.lower().replace(" ", "_")}',
                                        'height': 400,
                                        'width': 800,
                                        'scale': 1
                                    }
                                }
                            )
                            
                            # Add spacing between stratified plots
                            if strat_idx < len(strat_groups) - 1:
                                st.markdown("---")
                        else:
                            st.markdown(f"### {strat_label} ({stratification_selection.replace('-', ' ').title()})")
                            st.warning(f"No data available for {strat_label.lower()} patients.")
                            if strat_idx < len(strat_groups) - 1:
                                st.markdown("---")
                    
                    # Add statistics table below all stratified plots
                    if not filtered_plot_df.empty and plot_clustering_methods and plot_dr_methods:
                        st.markdown("---")
                        st.markdown("### Statistics Table")
                        st.markdown(f"Mean values with 95% confidence intervals for {METRIC_LABELS[selected_metric]}.")
                        
                        # Create and display heatmap table
                        table_fig = create_heatmap_table(
                            filtered_plot_df,
                            selected_metric,
                            plot_clustering_methods,
                            plot_dr_methods,
                            present_variants,
                            stratification_selection
                        )
                        
                        st.plotly_chart(
                            table_fig,
                            width='stretch',
                            config={
                                'displayModeBar': False,
                                'displaylogo': False
                            }
                        )
                        
                        # Add comparative tables for stratified view (pairwise comparisons)
                        st.markdown("---")
                        st.markdown("### Comparative Tables")
                        st.markdown(f"Pairwise comparisons between {strat_labels[0]} and {strat_labels[1]} for different dimensions. All metrics shown as columns.")
                        
                        # Get available metrics
                        available_metrics = [m for m in METRICS if m in filtered_plot_df.columns and not filtered_plot_df[m].isna().all()]
                        
                        # Compare clustering methods (pairwise: male vs female for each method)
                        if len(plot_clustering_methods) >= 1:
                            st.markdown("#### Clustering Methods Comparison")
                            comp_methods_df = create_stratified_pairwise_table(
                                filtered_plot_df,
                                available_metrics,
                                'method',
                                strat_groups,
                                strat_labels,
                                other_filters={
                                    'dr_method': plot_dr_methods if plot_dr_methods else None,
                                    'variant': present_variants if present_variants else None
                                }
                            )
                            if not comp_methods_df.empty:
                                st.dataframe(comp_methods_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_stratified_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'method',
                                    strat_groups,
                                    strat_labels,
                                    other_filters={
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for clustering methods comparison.")
                        
                        # Compare DR methods (pairwise: male vs female for each DR method)
                        if len(plot_dr_methods) >= 1:
                            st.markdown("#### DR Methods Comparison")
                            comp_dr_df = create_stratified_pairwise_table(
                                filtered_plot_df,
                                available_metrics,
                                'dr_method',
                                strat_groups,
                                strat_labels,
                                other_filters={
                                    'method': plot_clustering_methods if plot_clustering_methods else None,
                                    'variant': present_variants if present_variants else None
                                }
                            )
                            if not comp_dr_df.empty:
                                st.dataframe(comp_dr_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_stratified_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'dr_method',
                                    strat_groups,
                                    strat_labels,
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'variant': present_variants if present_variants else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for DR methods comparison.")
                        
                        # Compare variants (pairwise: male vs female for each variant)
                        if len(present_variants) >= 1:
                            st.markdown("#### Variants Comparison")
                            comp_variants_df = create_stratified_pairwise_table(
                                filtered_plot_df,
                                available_metrics,
                                'variant',
                                strat_groups,
                                strat_labels,
                                other_filters={
                                    'method': plot_clustering_methods if plot_clustering_methods else None,
                                    'dr_method': plot_dr_methods if plot_dr_methods else None
                                }
                            )
                            if not comp_variants_df.empty:
                                st.dataframe(comp_variants_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_stratified_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'variant',
                                    strat_groups,
                                    strat_labels,
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for variants comparison.")
                        
                        # Compare number of clusters (pairwise: male vs female for each n_clusters)
                        if 'n_clusters' in filtered_plot_df.columns:
                            n_clusters_values = sorted([v for v in filtered_plot_df['n_clusters'].dropna().unique() if pd.notna(v)])
                            if len(n_clusters_values) >= 1:
                                st.markdown("#### Number of Clusters Comparison")
                                comp_nclusters_df = create_stratified_pairwise_table(
                                    filtered_plot_df,
                                    available_metrics,
                                    'n_clusters',
                                    strat_groups,
                                    strat_labels,
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    }
                                )
                                if not comp_nclusters_df.empty:
                                    st.dataframe(comp_nclusters_df, width='stretch', hide_index=True)
                                    
                                    # Add visualization plot
                                    comp_plot = create_stratified_comparative_plot(
                                        filtered_plot_df,
                                        available_metrics,
                                        'n_clusters',
                                        strat_groups,
                                        strat_labels,
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None
                                        },
                                        selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                    )
                                    if comp_plot and len(comp_plot.data) > 0:
                                        st.plotly_chart(comp_plot, width='stretch')
                                else:
                                    st.info("Insufficient data for number of clusters comparison.")
                        
                        # Compare N dim (pairwise: male vs female for each dr_components)
                        if 'dr_components' in filtered_plot_df.columns:
                            n_dim_values = sorted([v for v in filtered_plot_df['dr_components'].dropna().unique() if pd.notna(v)])
                            if len(n_dim_values) >= 1:
                                st.markdown("#### N Dim Comparison")
                                comp_ndim_df = create_stratified_pairwise_table(
                                    filtered_plot_df,
                                    available_metrics,
                                    'dr_components',
                                    strat_groups,
                                    strat_labels,
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    }
                                )
                                if not comp_ndim_df.empty:
                                    st.dataframe(comp_ndim_df, width='stretch', hide_index=True)
                                    
                                    # Add visualization plot
                                    comp_plot = create_stratified_comparative_plot(
                                        filtered_plot_df,
                                        available_metrics,
                                        'dr_components',
                                        strat_groups,
                                        strat_labels,
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None
                                        },
                                        selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                    )
                                    if comp_plot and len(comp_plot.data) > 0:
                                        st.plotly_chart(comp_plot, width='stretch')
                                else:
                                    st.info("Insufficient data for N dim comparison.")
                        
                        # Feature Importance Comparison
                        st.markdown("---")
                        st.markdown("### Feature Importance Comparison")
                        st.markdown(f"Comparison of top-10 feature importances between {strat_labels[0]} and {strat_labels[1]}.")
                        
                        try:
                            fi_df = load_feature_importance_data(results_path)
                            
                            if fi_df.empty:
                                st.warning(f"No feature importance data found in the results directory: '{results_path}'")
                            else:
                                st.success(f"✓ Loaded {len(fi_df)} feature importance records.")
                                
                                # Create separate figure for each variant
                                if present_variants:
                                    for variant in present_variants:
                                        st.markdown(f"#### Variant: {variant}")
                                        
                                        fi_fig = create_feature_importance_comparison(
                                            fi_df,
                                            filtered_plot_df,
                                            strat_groups,
                                            strat_labels,
                                            other_filters={
                                                'method': plot_clustering_methods if plot_clustering_methods else None,
                                                'dr_method': plot_dr_methods if plot_dr_methods else None,
                                                'variant': [variant],
                                                'n_clusters': selected_clusters if selected_clusters else None,
                                                'dr_components': selected_dims if selected_dims else None,
                                                'panel': [data_selection] if data_selection != "all" else None
                                            },
                                            top_k=10
                                        )
                                        
                                        if fi_fig and len(fi_fig.data) > 0:
                                            st.plotly_chart(
                                                fi_fig,
                                                width='stretch',
                                                config={
                                                    'displayModeBar': True,
                                                    'displaylogo': False,
                                                    'toImageButtonOptions': {
                                                        'format': 'png',
                                                        'filename': f'feature_importance_{stratification_selection}_{variant}',
                                                        'height': 600,
                                                        'width': 1000,
                                                        'scale': 1
                                                    }
                                                }
                                            )
                                        else:
                                            st.info(f"No feature importance data available for variant {variant} with the selected filters.")
                                        
                                        if variant != present_variants[-1]:
                                            st.markdown("---")
                                else:
                                    # Fallback if no variants
                                    fi_fig = create_feature_importance_comparison(
                                        fi_df,
                                        filtered_plot_df,
                                        strat_groups,
                                        strat_labels,
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None,
                                            'n_clusters': selected_clusters if selected_clusters else None,
                                            'dr_components': selected_dims if selected_dims else None,
                                            'panel': [data_selection] if data_selection != "all" else None
                                        },
                                        top_k=10
                                    )
                                    
                                    if fi_fig and len(fi_fig.data) > 0:
                                        st.plotly_chart(
                                            fi_fig,
                                            width='stretch',
                                            config={
                                                'displayModeBar': True,
                                                'displaylogo': False,
                                                'toImageButtonOptions': {
                                                    'format': 'png',
                                                    'filename': f'feature_importance_{stratification_selection}',
                                                    'height': 600,
                                                    'width': 1000,
                                                    'scale': 1
                                                }
                                            }
                                        )
                                    else:
                                        st.warning("Could not generate the feature importance comparison plot.")
                                        st.info("This may be because no feature importance data was found for the selected filters, or there were no overlapping top features between the groups.")
                        except Exception as e:
                            import traceback
                            st.error(f"An error occurred while generating the feature importance plot: {e}")
                            st.code(traceback.format_exc())
                else:
                    # Overall view - single plot
                    if plot_type == "Scatter Plot":
                        fig = create_scatter_plot(
                            filtered_plot_df,
                            selected_metric,
                            present_variants,
                            plot_clustering_methods,
                            plot_dr_methods,
                            max_points_per_trace=max_points,
                            y_min=y_min,
                            y_max=y_max,
                            y_log_scale=y_log_scale
                        )
                    elif plot_type == "Violin Plot":
                        fig = create_violin_plot(
                            filtered_plot_df,
                            selected_metric,
                            present_variants,
                            plot_clustering_methods,
                            plot_dr_methods,
                            y_min=y_min,
                            y_max=y_max,
                            y_log_scale=y_log_scale
                        )
                    else:  # Box Plot
                        fig = create_boxen_plot(
                            filtered_plot_df,
                            selected_metric,
                            present_variants,
                            plot_clustering_methods,
                            plot_dr_methods,
                            y_min=y_min,
                            y_max=y_max,
                            y_log_scale=y_log_scale
                        )
                    
                    # Display plot
                    st.plotly_chart(
                        fig,
                        width='stretch',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{selected_metric}_{plot_type.lower().replace(" ", "_")}',
                                'height': 400,
                                'width': 800,
                                'scale': 1
                            }
                        }
                    )
                    
                    # Add statistics table below the figure
                    if not filtered_plot_df.empty and plot_clustering_methods and plot_dr_methods:
                        st.markdown("---")
                        st.markdown("### Statistics Table")
                        st.markdown(f"Mean values with 95% confidence intervals for {METRIC_LABELS[selected_metric]}.")
                        
                        # Create and display heatmap table
                        table_fig = create_heatmap_table(
                            filtered_plot_df,
                            selected_metric,
                            plot_clustering_methods,
                            plot_dr_methods,
                            present_variants,
                            stratification_selection
                        )
                        
                        st.plotly_chart(
                            table_fig,
                            width='stretch',
                            config={
                                'displayModeBar': False,
                                'displaylogo': False
                            }
                        )
                        
                        # Add comparative tables for overall view
                        st.markdown("---")
                        st.markdown("### Comparative Tables")
                        st.markdown("Comparisons with p-values for different dimensions. All metrics shown as columns.")
                        
                        # Get available metrics
                        available_metrics = [m for m in METRICS if m in filtered_plot_df.columns and not filtered_plot_df[m].isna().all()]
                        
                        # Compare clustering methods
                        if len(plot_clustering_methods) >= 2:
                            st.markdown("#### Clustering Methods Comparison")
                            comp_methods_df = create_comparative_table(
                                filtered_plot_df,
                                available_metrics,
                                'method',
                                other_filters={
                                    'dr_method': plot_dr_methods if plot_dr_methods else None,
                                    'variant': present_variants if present_variants else None
                                }
                            )
                            if not comp_methods_df.empty:
                                st.dataframe(comp_methods_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'method',
                                    other_filters={
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for clustering methods comparison.")
                        
                        # Compare DR methods
                        if len(plot_dr_methods) >= 2:
                            st.markdown("#### DR Methods Comparison")
                            comp_dr_df = create_comparative_table(
                                filtered_plot_df,
                                available_metrics,
                                'dr_method',
                                other_filters={
                                    'method': plot_clustering_methods if plot_clustering_methods else None,
                                    'variant': present_variants if present_variants else None
                                }
                            )
                            if not comp_dr_df.empty:
                                st.dataframe(comp_dr_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'dr_method',
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'variant': present_variants if present_variants else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for DR methods comparison.")
                        
                        # Compare variants
                        if len(present_variants) >= 2:
                            st.markdown("#### Variants Comparison")
                            comp_variants_df = create_comparative_table(
                                filtered_plot_df,
                                available_metrics,
                                'variant',
                                other_filters={
                                    'method': plot_clustering_methods if plot_clustering_methods else None,
                                    'dr_method': plot_dr_methods if plot_dr_methods else None
                                }
                            )
                            if not comp_variants_df.empty:
                                st.dataframe(comp_variants_df, width='stretch', hide_index=True)
                                
                                # Add visualization plot
                                comp_plot = create_comparative_plot(
                                    filtered_plot_df,
                                    available_metrics,
                                    'variant',
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None
                                    },
                                    selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                )
                                if comp_plot and len(comp_plot.data) > 0:
                                    st.plotly_chart(comp_plot, width='stretch')
                            else:
                                st.info("Insufficient data for variants comparison.")
                        
                        # Compare number of clusters
                        if 'n_clusters' in filtered_plot_df.columns:
                            n_clusters_values = sorted([v for v in filtered_plot_df['n_clusters'].dropna().unique() if pd.notna(v)])
                            if len(n_clusters_values) >= 2:
                                st.markdown("#### Number of Clusters Comparison")
                                comp_nclusters_df = create_comparative_table(
                                    filtered_plot_df,
                                    available_metrics,
                                    'n_clusters',
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    }
                                )
                                if not comp_nclusters_df.empty:
                                    st.dataframe(comp_nclusters_df, width='stretch', hide_index=True)
                                    
                                    # Add visualization plot
                                    comp_plot = create_comparative_plot(
                                        filtered_plot_df,
                                        available_metrics,
                                        'n_clusters',
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None
                                        },
                                        selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                    )
                                    if comp_plot and len(comp_plot.data) > 0:
                                        st.plotly_chart(comp_plot, width='stretch')
                                else:
                                    st.info("Insufficient data for number of clusters comparison.")
                        
                        # Compare N dim (dr_components)
                        if 'dr_components' in filtered_plot_df.columns:
                            n_dim_values = sorted([v for v in filtered_plot_df['dr_components'].dropna().unique() if pd.notna(v)])
                            if len(n_dim_values) >= 2:
                                st.markdown("#### N Dim Comparison")
                                comp_ndim_df = create_comparative_table(
                                    filtered_plot_df,
                                    available_metrics,
                                    'dr_components',
                                    other_filters={
                                        'method': plot_clustering_methods if plot_clustering_methods else None,
                                        'dr_method': plot_dr_methods if plot_dr_methods else None,
                                        'variant': present_variants if present_variants else None
                                    }
                                )
                                if not comp_ndim_df.empty:
                                    st.dataframe(comp_ndim_df, width='stretch', hide_index=True)
                                    
                                    # Add visualization plot
                                    comp_plot = create_comparative_plot(
                                        filtered_plot_df,
                                        available_metrics,
                                        'dr_components',
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None
                                        },
                                        selected_metrics=['silhouette_score', 'davies_bouldin_index'] if 'davies_bouldin_index' in available_metrics else ['silhouette_score']
                                    )
                                    if comp_plot and len(comp_plot.data) > 0:
                                        st.plotly_chart(comp_plot, width='stretch')
                                else:
                                    st.info("Insufficient data for N dim comparison.")
                        
                        # Feature Importance for Overall View
                        st.markdown("---")
                        st.markdown("### Feature Importance")
                        st.markdown("Overall feature importance across all selected configurations.")
                        
                        try:
                            fi_df = load_feature_importance_data(results_path)
                            
                            if fi_df.empty:
                                st.warning(f"No feature importance data found in the results directory: '{results_path}'")
                            else:
                                st.success(f"✓ Loaded {len(fi_df)} feature importance records.")
                                
                                # Create separate figure for each variant
                                if present_variants:
                                    for variant in present_variants:
                                        st.markdown(f"#### Variant: {variant}")
                                        
                                        fi_fig = create_overall_feature_importance(
                                            fi_df,
                                            filtered_plot_df,
                                            other_filters={
                                                'method': plot_clustering_methods if plot_clustering_methods else None,
                                                'dr_method': plot_dr_methods if plot_dr_methods else None,
                                                'variant': [variant],
                                                'n_clusters': selected_clusters if selected_clusters else None,
                                                'dr_components': selected_dims if selected_dims else None,
                                                'panel': [data_selection] if data_selection != "all" else None
                                            },
                                            top_k=10
                                        )
                                        
                                        if fi_fig and len(fi_fig.data) > 0:
                                            st.plotly_chart(
                                                fi_fig,
                                                width='stretch',
                                                config={
                                                    'displayModeBar': True,
                                                    'displaylogo': False,
                                                    'toImageButtonOptions': {
                                                        'format': 'png',
                                                        'filename': f'feature_importance_overall_{variant}',
                                                        'height': 600,
                                                        'width': 1000,
                                                        'scale': 1
                                                    }
                                                }
                                            )
                                        else:
                                            st.info(f"No feature importance data available for variant {variant} with the selected filters.")
                                        
                                        if variant != present_variants[-1]:
                                            st.markdown("---")
                                else:
                                    # Fallback if no variants
                                    fi_fig = create_overall_feature_importance(
                                        fi_df,
                                        filtered_plot_df,
                                        other_filters={
                                            'method': plot_clustering_methods if plot_clustering_methods else None,
                                            'dr_method': plot_dr_methods if plot_dr_methods else None,
                                            'variant': present_variants if present_variants else None,
                                            'n_clusters': selected_clusters if selected_clusters else None,
                                            'dr_components': selected_dims if selected_dims else None,
                                            'panel': [data_selection] if data_selection != "all" else None
                                        },
                                        top_k=10
                                    )
                                    
                                    if fi_fig and len(fi_fig.data) > 0:
                                        st.plotly_chart(
                                            fi_fig,
                                            width='stretch',
                                            config={
                                                'displayModeBar': True,
                                                'displaylogo': False,
                                                'toImageButtonOptions': {
                                                    'format': 'png',
                                                    'filename': 'feature_importance_overall',
                                                    'height': 600,
                                                    'width': 1000,
                                                    'scale': 1
                                                }
                                            }
                                        )
                                    else:
                                        st.warning("Could not generate the feature importance plot.")
                                        st.info("This may be because no feature importance data was found for the selected filters.")
                        except Exception as e:
                            import traceback
                            st.error(f"An error occurred while generating the feature importance plot: {e}")
                            st.code(traceback.format_exc())
            else:
                st.warning("No data available for the selected method combinations.")
        else:
            st.warning(f"Metric '{METRIC_LABELS[selected_metric]}' has no valid values for the selected filters.")
    else:
        st.warning("Please select a metric to display.")


if __name__ == "__main__":
    main()


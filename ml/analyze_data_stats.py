#!/usr/bin/env python3
"""
Analyze data statistics for normalization from train subset.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, Tuple


def load_results_data(run_path: str) -> pd.DataFrame:
    """Load results_xy.csv from a run directory."""
    results_file = os.path.join(run_path, 'results_xy.csv')
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return pd.DataFrame()


def analyze_data_statistics():
    """Analyze statistics from train subset (seeds 1-18) for normalization."""
    
    # Load seed metadata
    seed_df = pd.read_csv('../seed1-30.csv')
    
    # Filter for train subset using split column
    train_seeds = seed_df[seed_df['split'] == 'train']
    
    print(f"Found {len(train_seeds)} train runs")
    
    # Collect all results data from train runs
    all_results = []
    
    for _, row in train_seeds.iterrows():
        run_path = f"../runs/{row['path']}"
        if not os.path.exists(run_path):
            # Try alternative path structure
            run_path = f"../runs/{row['path']}"
        
        if os.path.exists(run_path):
            results_df = load_results_data(run_path)
            if not results_df.empty:
                # Filter out root node (mote 1) - it doesn't receive messages
                results_df = results_df[results_df['mote'] != 1]
                
                # Add graph-level features
                results_df['spacing'] = row['spacing']
                results_df['N'] = row['N']
                results_df['coverage_30'] = row['coverage_30']
                all_results.append(results_df)
                print(f"Loaded data from {row['path']}: {len(results_df)} nodes (excluding root)")
    
    if not all_results:
        print("No results data found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Total nodes in train set: {len(combined_df)}")
    
    # Calculate statistics for normalization
    stats = {}
    
    # Node-level features
    node_features = ['x', 'y', 'initial_battery']
    for feature in node_features:
        if feature in combined_df.columns:
            values = combined_df[feature].dropna()
            stats[f'node_{feature}'] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            }
            print(f"{feature}: mean={stats[f'node_{feature}']['mean']:.2f}, "
                  f"std={stats[f'node_{feature}']['std']:.2f}, "
                  f"range=[{stats[f'node_{feature}']['min']:.2f}, {stats[f'node_{feature}']['max']:.2f}]")
    
    # Graph-level features
    graph_features = ['spacing', 'N', 'coverage_30']
    for feature in graph_features:
        if feature in combined_df.columns:
            values = combined_df[feature].dropna()
            stats[f'graph_{feature}'] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            }
            print(f"{feature}: mean={stats[f'graph_{feature}']['mean']:.2f}, "
                  f"std={stats[f'graph_{feature}']['std']:.2f}, "
                  f"range=[{stats[f'graph_{feature}']['min']:.2f}, {stats[f'graph_{feature}']['max']:.2f}]")
    
    # Node-level outputs
    node_outputs = ['last_msg_recv_by_root', 'uptime']
    for output in node_outputs:
        if output in combined_df.columns:
            values = combined_df[output].dropna()
            stats[f'node_output_{output}'] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max())
            }
            print(f"{output}: mean={stats[f'node_output_{output}']['mean']:.2f}, "
                  f"std={stats[f'node_output_{output}']['std']:.2f}, "
                  f"range=[{stats[f'node_output_{output}']['min']:.2f}, {stats[f'node_output_{output}']['max']:.2f}]")
    
    # Print hardcoded normalization constants
    print("\n" + "="*50)
    print("HARDCODED NORMALIZATION CONSTANTS:")
    print("="*50)
    
    print("\n# Node-level input features")
    for feature in node_features:
        if f'node_{feature}' in stats:
            s = stats[f'node_{feature}']
            print(f"    '{feature}': {{'mean': {s['mean']:.6f}, 'std': {s['std']:.6f}}},")
    
    print("\n# Graph-level input features")
    for feature in graph_features:
        if f'graph_{feature}' in stats:
            s = stats[f'graph_{feature}']
            print(f"    '{feature}': {{'mean': {s['mean']:.6f}, 'std': {s['std']:.6f}}},")
    
    print("\n# Node-level outputs")
    for output in node_outputs:
        if f'node_output_{output}' in stats:
            s = stats[f'node_output_{output}']
            print(f"    '{output}': {{'mean': {s['mean']:.6f}, 'std': {s['std']:.6f}}},")
    
    print("\n# Graph-level outputs")
    if 'graph_coverage_30' in stats:
        s = stats['graph_coverage_30']
        print(f"    'coverage_30': {{'mean': {s['mean']:.6f}, 'std': {s['std']:.6f}}},")


if __name__ == "__main__":
    analyze_data_statistics()

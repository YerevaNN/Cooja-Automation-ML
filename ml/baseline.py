#!/usr/bin/env python3
"""
Baseline prediction script for Battery RPL Simulation.
Provides simple heuristic baselines for uptime and last_msg_recv_by_root prediction.
"""

import argparse
import pandas as pd
import os
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import cv2


def load_results_data(run_path: str) -> pd.DataFrame:
    """Load results_xy.csv from a run directory."""
    results_file = os.path.join(run_path, 'results_xy.csv')
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return pd.DataFrame()


def compute_uptime_baseline(initial_battery: float) -> float:
    """
    Compute baseline uptime prediction based on initial battery.
    Formula: min(295, initial_battery_level*5.4+25)
    """
    return min(295.0, initial_battery * 5.4 + 25.0)


def build_lastmsg_mean_lookup(seed_file: str, runs_dir: str, num_bins: int = 50) -> Dict:
    """
    Build a lookup table for mean-based last_msg prediction from training data.
    
    Args:
        seed_file: Path to seed CSV file
        runs_dir: Directory containing run results
        num_bins: Number of bins for battery levels
    
    Returns:
        Dictionary with bins, bin_centers, and bin_means
    """
    # Load training data
    seed_df = pd.read_csv(seed_file)
    train_df = seed_df[seed_df['split'] == 'train']
    
    battery_data = []
    lastmsg_data = []
    
    # Collect training data
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building lookup table"):
        results_file = os.path.join(runs_dir, row['path'], 'results_xy.csv')
        if os.path.exists(results_file):
            try:
                df_run = pd.read_csv(results_file)
                df_run = df_run[df_run['mote'] != 1]  # Filter out root
                
                battery_data.extend(df_run['initial_battery'].values)
                lastmsg_data.extend(df_run['last_msg_recv_by_root'].values)
            except Exception:
                pass
    
    # Clean data
    battery_data = np.array(battery_data)
    lastmsg_data = np.array(lastmsg_data)
    valid_mask = ~(np.isnan(battery_data) | np.isnan(lastmsg_data))
    battery_clean = battery_data[valid_mask]
    lastmsg_clean = lastmsg_data[valid_mask]
    
    # Create bins and compute means
    bins = np.linspace(battery_clean.min(), battery_clean.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    
    for i in range(len(bins) - 1):
        mask = (battery_clean >= bins[i]) & (battery_clean < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(lastmsg_clean[mask].mean())
        else:
            bin_means.append(np.nan)
    
    return {
        'bins': bins,
        'bin_centers': bin_centers,
        'bin_means': np.array(bin_means)
    }


def compute_lastmsg_mean_baseline(initial_battery: float, lookup: Dict) -> float:
    """
    Compute last_msg baseline using mean lookup table.
    
    Args:
        initial_battery: Initial battery level
        lookup: Lookup table from build_lastmsg_mean_lookup
    
    Returns:
        Predicted last_msg value
    """
    bins = lookup['bins']
    bin_means = lookup['bin_means']
    
    # Find which bin this battery falls into
    bin_idx = np.digitize(initial_battery, bins) - 1
    
    # Clip to valid range
    bin_idx = max(0, min(bin_idx, len(bin_means) - 1))
    
    # Return mean for this bin (or NaN if no data)
    return bin_means[bin_idx] if not np.isnan(bin_means[bin_idx]) else 0.0


def compute_coverage_baseline(N: int) -> float:
    """
    Compute baseline coverage prediction based on number of nodes.
    Formula: 27295.06 * log(N) - 55685.44 (from log fit)
    """
    return 27295.06 * np.log(N) - 55685.44


def calculate_sensing_area_opencv_all_nodes(df, radius=30.0, resolution=10):
    """
    Calculate sensing area using OpenCV for all nodes (ignoring activity status).
    This is the exact same code from extract_simulation_data.py but uses ALL nodes.
    
    Args:
        df: DataFrame with x, y coordinates
        radius: Coverage radius in meters (default: 30.0)
        resolution: Resolution in pixels per meter (default: 10)
    
    Returns:
        Sensing area in square meters
    """
    if df is None or len(df) == 0:
        return 0.0
    
    x_coords = df['x'].values
    y_coords = df['y'].values
    
    # Calculate bounding box
    x_min, x_max = x_coords.min() - radius, x_coords.max() + radius
    y_min, y_max = y_coords.min() - radius, y_coords.max() + radius
    
    # Create image dimensions
    width = int((x_max - x_min) * resolution)
    height = int((y_max - y_min) * resolution)
    
    # Create white image (uncovered area)
    img = np.full((height, width), 255, dtype=np.uint8)
    
    # Convert coordinates to pixel coordinates
    pixel_x = ((x_coords - x_min) * resolution).astype(int)
    pixel_y = ((y_coords - y_min) * resolution).astype(int)
    radius_pixels = int(radius * resolution)
    
    # Draw circles using OpenCV (much faster than PIL)
    for px, py in zip(pixel_x, pixel_y):
        cv2.circle(img, (px, py), radius_pixels, 0, -1)  # -1 = filled circle
    
    # Calculate sensing area
    covered_pixels = np.sum(img == 0)  # Count black pixels
    pixel_area_m2 = 1.0 / (resolution ** 2)
    sensing_area = covered_pixels * pixel_area_m2
    
    return sensing_area


def compute_coverage_xy_baseline(results_df: pd.DataFrame, battery_threshold: float = 0.0) -> float:
    """
    Compute coverage baseline using x,y coordinates of nodes with battery >= threshold.
    
    Note: Many nodes have NaN for last_msg (never joined network). Ground truth only
    counts nodes with last_msg > 0. Battery threshold is a heuristic to predict which
    nodes will successfully send messages.
    
    Args:
        results_df: DataFrame with node data
        battery_threshold: Minimum battery to include (default: 0 = all nodes with valid battery)
    """
    # Filter out root node (mote 1)
    client_nodes = results_df[results_df['mote'] != 1].copy()
    
    # Filter by battery: only nodes with valid battery >= threshold
    client_nodes = client_nodes[client_nodes['initial_battery'].notna()]
    if battery_threshold > 0:
        client_nodes = client_nodes[client_nodes['initial_battery'] >= battery_threshold]
    
    if len(client_nodes) == 0:
        return 0.0
    
    # Calculate coverage using filtered nodes
    coverage = calculate_sensing_area_opencv_all_nodes(client_nodes, radius=30.0, resolution=10)
    
    return round(coverage, 2)


def run_baseline(seed_file: str, runs_dir: str, split: str, task: str, battery_threshold: float = 0.0, 
                 lastmsg_method: str = 'uptime') -> pd.DataFrame:
    """Run baseline prediction for the specified task and split.
    
    Args:
        lastmsg_method: For last_msg_recv_by_root task, use 'uptime' or 'mean' method
    """

    # Load seed metadata
    seed_df = pd.read_csv(seed_file)
    seed_df = seed_df[seed_df['split'] == split]

    print(f"Computing baseline for {len(seed_df)} runs on {split} split")
    print(f"Task: {task}")
    if battery_threshold > 0:
        print(f"Battery threshold: {battery_threshold}")
    if task == 'last_msg_recv_by_root':
        print(f"Last_msg method: {lastmsg_method}")
    
    # Build lookup table for mean-based method if needed
    lookup = None
    if task == 'last_msg_recv_by_root' and lastmsg_method == 'mean':
        print("Building mean lookup table from training data...")
        lookup = build_lastmsg_mean_lookup(seed_file, runs_dir, num_bins=50)
        print(f"Lookup table built with {len(lookup['bin_means'])} bins")

    results = []

    for _, row in tqdm(seed_df.iterrows(), total=len(seed_df), desc="Processing runs"):
        run_path = os.path.join(runs_dir, row['path'])

        # Try the expected path
        possible_paths = [run_path]

        results_file = None
        for path in possible_paths:
            results_path = os.path.join(path, 'results_xy.csv')
            if os.path.exists(results_path):
                results_file = results_path
                break

        if results_file is None:
            print(f"Warning: Could not find results_xy.csv for {row['path']}")
            continue

        try:
            # Load results data
            results_df = pd.read_csv(results_file)

            # Filter out root node (mote 1) as it has special properties
            results_df = results_df[results_df['mote'] != 1]

            if len(results_df) < 1:  # Need at least 1 node
                continue

            # Handle different task types
            if task == 'coverage_30':
                # For graph-level task, process once per graph
                predicted = compute_coverage_baseline(row['N'])
                ground_truth = row['coverage_30']
                
                results.append({
                    'path': row['path'],
                    'N': row['N'],
                    'spacing': row['spacing'],
                    'x': 0.0,  # Not applicable for graph-level
                    'y': 0.0,  # Not applicable for graph-level
                    'initial_battery': 0.0,  # Not applicable for graph-level
                    'ground_truth': ground_truth,
                    'predicted': predicted
                })
            elif task == 'coverage_xy':
                # For graph-level task using x,y coordinates (stronger baseline)
                predicted = compute_coverage_xy_baseline(results_df, battery_threshold=battery_threshold)
                ground_truth = row['coverage_30']
                
                results.append({
                    'path': row['path'],
                    'N': row['N'],
                    'spacing': row['spacing'],
                    'x': 0.0,  # Not applicable for graph-level
                    'y': 0.0,  # Not applicable for graph-level
                    'initial_battery': 0.0,  # Not applicable for graph-level
                    'ground_truth': ground_truth,
                    'predicted': predicted
                })
            else:
                # For node-level tasks, process each node in the graph
                for _, node_row in results_df.iterrows():
                    node_id = node_row['mote']
                    x = node_row['x']
                    y = node_row['y']
                    initial_battery = node_row['initial_battery']

                    # Compute baseline prediction based on task
                    if task == 'uptime':
                        predicted = compute_uptime_baseline(initial_battery)
                        ground_truth = node_row['uptime']
                    elif task == 'last_msg_recv_by_root':
                        if lastmsg_method == 'mean':
                            predicted = compute_lastmsg_mean_baseline(initial_battery, lookup)
                        else:  # uptime method
                            predicted = compute_uptime_baseline(initial_battery) / 1.5
                        ground_truth = node_row['last_msg_recv_by_root']
                    else:
                        raise ValueError(f"Unknown task: {task}")

                    results.append({
                        'path': row['path'],
                        'N': row['N'],
                        'spacing': row['spacing'],
                        'x': x,
                        'y': y,
                        'initial_battery': initial_battery,
                        'ground_truth': ground_truth,
                        'predicted': predicted
                    })

        except Exception as e:
            print(f"Error processing {row['path']}: {e}")
            continue

    return pd.DataFrame(results)


def compute_statistics(results_df: pd.DataFrame) -> Dict:
    """Compute prediction statistics."""
    if len(results_df) == 0:
        return {}

    errors = results_df['predicted'] - results_df['ground_truth']
    abs_errors = np.abs(errors)

    stats = {
        'MAE': abs_errors.mean(),
        'MSE': (errors**2).mean(),
        'RMSE': np.sqrt((errors**2).mean()),
        'Count': len(results_df),
        'Mean_Predicted': results_df['predicted'].mean(),
        'Std_Predicted': results_df['predicted'].std(),
        'Mean_Ground_Truth': results_df['ground_truth'].mean(),
        'Std_Ground_Truth': results_df['ground_truth'].std()
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute baseline predictions for Battery RPL Simulation')

    # Required arguments
    parser.add_argument('--seed-file', type=str, required=True,
                        help='Path to seed CSV file (e.g., seed1-30.csv)')
    parser.add_argument('--runs-dir', type=str, required=True,
                        help='Directory containing run results')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Data split to compute baseline on')
    parser.add_argument('--task', type=str, required=True, choices=['uptime', 'last_msg_recv_by_root', 'coverage_30', 'coverage_xy'],
                        help='Task to compute baseline for')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Output CSV file path for baseline predictions')
    parser.add_argument('--battery-threshold', type=float, default=0.0,
                        help='Minimum battery level to include node in coverage_xy (default: 0.0 = all nodes)')
    parser.add_argument('--lastmsg-method', type=str, default='uptime', choices=['uptime', 'mean'],
                        help='Method for last_msg_recv_by_root baseline: uptime (formula-based) or mean (training data average)')

    args = parser.parse_args()

    print(f"Computing baseline for task: {args.task}")
    print(f"Dataset: {args.seed_file}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_csv}")
    if args.battery_threshold > 0:
        print(f"Battery threshold: {args.battery_threshold}")

    # Run baseline prediction
    results_df = run_baseline(args.seed_file, args.runs_dir, args.split, args.task, 
                               args.battery_threshold, args.lastmsg_method)

    if len(results_df) == 0:
        print("No results generated!")
        return

    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"Baseline predictions saved to: {args.output_csv}")
    print(f"Total predictions: {len(results_df)}")

    # Compute and display statistics
    stats = compute_statistics(results_df)
    print("\nBaseline Statistics:")
    print(f"MAE: {stats['MAE']:.4f}")
    print(f"MSE: {stats['MSE']:.4f}")
    print(f"RMSE: {stats['RMSE']:.4f}")
    print(f"Mean Predicted: {stats['Mean_Predicted']:.2f}")
    print(f"Std Predicted: {stats['Std_Predicted']:.2f}")
    print(f"Mean Ground Truth: {stats['Mean_Ground_Truth']:.2f}")
    print(f"Std Ground Truth: {stats['Std_Ground_Truth']:.2f}")

    print(f"\nBaseline Description:")
    if args.task == 'uptime':
        print("Uptime baseline: min(295, initial_battery*5.4+25)")
        print("This assumes ~5.4 units of uptime per battery unit, with max of 295")
    elif args.task == 'last_msg_recv_by_root':
        if args.lastmsg_method == 'uptime':
            print("Last_msg_recv_by_root baseline: uptime_baseline/1.5")
            print("Formula: min(295, initial_battery*5.4+25) / 1.5")
            print("This assumes a scaled relationship between battery and message reception time")
        else:  # mean
            print("Last_msg_recv_by_root baseline: mean-based lookup")
            print("Uses average last_msg values from training data, binned by battery level")
            print("This provides an empirical baseline based on observed behavior")
    elif args.task == 'coverage_30':
        print("Coverage baseline: 27295.06 * log(N) - 55685.44")
        print("This uses only the number of nodes N to predict coverage")
    elif args.task == 'coverage_xy':
        if args.battery_threshold > 0:
            print(f"Coverage XY baseline: uses x,y coordinates of nodes with battery >= {args.battery_threshold}")
            print(f"This filters out {args.battery_threshold}% of low-battery nodes")
        else:
            print("Coverage XY baseline: uses actual x,y coordinates of ALL nodes")
            print("This assumes all nodes remain active (perfect information baseline)")


if __name__ == "__main__":
    main()

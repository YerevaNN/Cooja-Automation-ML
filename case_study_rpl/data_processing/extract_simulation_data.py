#!/usr/bin/env python3
"""
Script to extract simulation data from runs folder and create a CSV dataset.

Each row corresponds to one simulation (subfolder), with columns:
- path: subfolder name
- N: number of nodes (extracted from folder name)
- spacing: spacing parameter (extracted from folder name)
- seed: random seed (extracted from folder name)
- slurm_job_id: SLURM job ID (extracted from folder name)
- split: empty for now
- coverage_30: raw sensing area in square meters using OpenCV-based calculation
- active_nodes: number of nodes with last_msg_recv_by_root >= 295
"""

import os
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np
import cv2


def parse_folder_name(folder_name: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Parse folder name to extract N, spacing, seed, and job_id.
    
    Expected format: sparse_grid_n{N}_s{spacing}_seed{seed}_job{job_id}
    
    Args:
        folder_name: Name of the simulation folder
        
    Returns:
        Tuple of (N, spacing, seed, job_id) or (None, None, None, None) if parsing fails
    """
    pattern = r'sparse_grid_n(\d+)_s(\d+)_seed(\d+)_job(\d+)'
    match = re.match(pattern, folder_name)
    
    if match:
        N = int(match.group(1))
        spacing = int(match.group(2))
        seed = int(match.group(3))
        job_id = int(match.group(4))
        return N, spacing, seed, job_id
    else:
        print(f"Warning: Could not parse folder name: {folder_name}")
        return None, None, None, None


def calculate_active_nodes(results_xy_path: str) -> int:
    """
    Calculate the number of active nodes by counting rows where last_msg_recv_by_root >= 295.
    
    Args:
        results_xy_path: Path to the results_xy.csv file
        
    Returns:
        Number of active nodes
    """
    try:
        df = pd.read_csv(results_xy_path)
        
        # Check if the required column exists
        if 'last_msg_recv_by_root' not in df.columns:
            print(f"Warning: 'last_msg_recv_by_root' column not found in {results_xy_path}")
            return 0
        
        # Count nodes with last_msg_recv_by_root >= 295
        # Handle NaN values by treating them as 0 (inactive)
        active_nodes = df['last_msg_recv_by_root'].fillna(0).ge(295).sum()
        return int(active_nodes)
        
    except Exception as e:
        print(f"Error reading {results_xy_path}: {e}")
        return 0


def calculate_sensing_area_opencv(df, radius, resolution=10):
    """
    Calculate sensing area using OpenCV for maximum performance.
    Only draws circles around nodes with last_msg_recv_by_root > 0.
    Avoids linspace and meshgrid operations.
    """
    if df is None or len(df) == 0:
        return 0.0, None
    
    # Filter nodes that have received messages from root
    if 'last_msg_recv_by_root' in df.columns:
        active_nodes = df[df['last_msg_recv_by_root'] > 0]
    else:
        # If column doesn't exist, use all nodes
        active_nodes = df
    
    if len(active_nodes) == 0:
        return 0.0, None
    
    x_coords = active_nodes['x'].values
    y_coords = active_nodes['y'].values
    n_nodes = len(x_coords)
    
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
    
    return sensing_area, img


def calculate_coverage_30(results_xy_path: str, radius: float = 30.0, resolution: int = 10) -> float:
    """
    Calculate raw sensing area using OpenCV-based calculation.
    
    Args:
        results_xy_path: Path to the results_xy.csv file
        radius: Coverage radius in meters (default: 30.0)
        resolution: Resolution in pixels per meter (default: 10)
        
    Returns:
        Raw sensing area in square meters
    """
    try:
        df = pd.read_csv(results_xy_path)
        
        # Check if required columns exist
        if 'x' not in df.columns or 'y' not in df.columns:
            print(f"Warning: 'x' or 'y' columns not found in {results_xy_path}")
            return 0.0
        
        # Calculate sensing area using OpenCV
        sensing_area, _ = calculate_sensing_area_opencv(df, radius, resolution)
        
        # Round to nearest integer
        return round(sensing_area)
        
    except Exception as e:
        print(f"Error calculating coverage for {results_xy_path}: {e}")
        return 0.0


def extract_simulation_data(runs_folder: str, output_csv: str = "simulation_dataset.csv") -> None:
    """
    Extract simulation data from runs folder and save to CSV.
    
    Args:
        runs_folder: Path to the runs folder containing simulation subfolders
        output_csv: Output CSV file path
    """
    runs_path = Path(runs_folder)
    
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs folder not found: {runs_folder}")
    
    # List to store all simulation data
    simulation_data = []
    
    # Get all subdirectories (simulation folders)
    simulation_folders = [f for f in runs_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(simulation_folders)} simulation folders")
    
    for folder in tqdm(simulation_folders, desc="Processing simulations"):
        folder_name = folder.name
        
        # Parse folder name to extract parameters
        N, spacing, seed, job_id = parse_folder_name(folder_name)
        
        if N is None:
            print(f"Skipping folder due to parsing error: {folder_name}")
            continue
        
        # Check if results_xy.csv exists
        results_xy_path = folder / "results_xy.csv"
        if not results_xy_path.exists():
            print(f"Warning: results_xy.csv not found in {folder_name}")
            active_nodes = 0
            coverage_30 = 0.0
        else:
            # Calculate active nodes
            active_nodes = calculate_active_nodes(str(results_xy_path))
            
            # Calculate coverage using OpenCV method
            coverage_30 = calculate_coverage_30(str(results_xy_path), radius=30.0, resolution=10)
        
        # Add simulation data
        simulation_data.append({
            'path': folder_name,
            'N': N,
            'spacing': spacing,
            'seed': seed,
            'slurm_job_id': job_id,
            'split': '',  # Empty for now
            'coverage_30': coverage_30,
            'active_nodes': active_nodes
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(simulation_data)
    
    # Sort by N, spacing, seed for better organization
    df = df.sort_values(['N', 'spacing', 'seed'])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nDataset saved to: {output_csv}")
    print(f"Total simulations processed: {len(simulation_data)}")
    print(f"Dataset shape: {df.shape}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(f"Number of nodes (N): {df['N'].unique()}")
    print(f"Spacing values: {sorted(df['spacing'].unique())}")
    print(f"Seed range: {df['seed'].min()} - {df['seed'].max()}")
    print(f"Active nodes range: {df['active_nodes'].min()} - {df['active_nodes'].max()}")
    print(f"Average active nodes: {df['active_nodes'].mean():.2f}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Extract simulation data from runs folder')
    parser.add_argument('runs_folder', help='Path to the runs folder containing simulation subfolders')
    parser.add_argument('-o', '--output', default='simulation_dataset.csv', 
                       help='Output CSV file path (default: simulation_dataset.csv)')
    
    args = parser.parse_args()
    
    try:
        extract_simulation_data(args.runs_folder, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

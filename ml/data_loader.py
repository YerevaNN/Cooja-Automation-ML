#!/usr/bin/env python3
"""
Data loader for GNN training on battery RPL simulation data.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm


class BatteryRPLDataset(Dataset):
    """
    Dataset for loading battery RPL simulation data for GNN training.
    """
    
    def __init__(self, seed_file: str, runs_dir: str, split: str = None, 
                 node_features: list = ['x', 'y', 'initial_battery'],
                 graph_features: list = [],
                 target_type: str = 'node',
                 target_feature: str = 'last_msg_recv_by_root'):
        """
        Initialize dataset.
        
        Args:
            seed_file: Path to seed CSV file (e.g., seed1-30.csv)
            runs_dir: Directory containing run results
            split: Data split ('train', 'val', 'test') or None for all data
            node_features: List of node features to include
            graph_features: List of graph features to include
            target_type: 'node' or 'graph'
            target_feature: Target feature name
        """
        self.seed_file = seed_file
        self.runs_dir = runs_dir
        self.split = split
        self.node_features = node_features
        self.graph_features = graph_features
        self.target_type = target_type
        self.target_feature = target_feature
        
        # Load seed metadata
        self.seed_df = pd.read_csv(seed_file)
        
        # Filter by split if specified
        if split is not None:
            self.seed_df = self.seed_df[self.seed_df['split'] == split]
        
        print(f"Loaded {len(self.seed_df)} runs for split: {split or 'all'}")
        
        # Load all data
        self.data_list = self._load_all_data()
        print(f"Total graphs loaded: {len(self.data_list)}")
    
    def _load_all_data(self) -> List[Data]:
        """Load all simulation data into memory."""
        data_list = []
        
        for _, row in tqdm(self.seed_df.iterrows(), total=len(self.seed_df), desc="Loading data"):
            run_path = os.path.join(self.runs_dir, row['path'])
            
            # Use the expected path
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
                
                # Keep root node in the graph (needed for topology), but mark it
                # We'll exclude it from loss calculation later
                
                if len(results_df) < 2:  # Need at least 2 nodes for a graph (including root)
                    continue
                
                # Create graph features
                graph_features = {
                    'spacing': row['spacing'],
                    'N': row['N'],
                    'coverage_30': row['coverage_30']
                }
                
                # Create PyTorch Geometric Data object
                data = self._create_data_object(results_df, graph_features)
                if data is not None:
                    data_list.append(data)
                    
            except Exception as e:
                print(f"Error loading {row['path']}: {e}")
                continue
        
        return data_list
    
    def _create_data_object(self, results_df: pd.DataFrame, graph_features: Dict[str, float]) -> Data:
        """Create PyTorch Geometric Data object from results DataFrame."""
        try:
            # Node features: selected features only
            node_features_data = results_df[self.node_features].values
            # Handle NaN values in features
            node_features_data = np.nan_to_num(node_features_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            node_features = torch.tensor(
                node_features_data, 
                dtype=torch.float32
            )
            
            # Node targets: all available targets (will be filtered in model)
            # Handle NaN values: NaN means node never sent a message, use -1 as special token
            last_msg = results_df['last_msg_recv_by_root'].fillna(-1.0).values
            uptime = results_df['uptime'].fillna(-1.0).values
            
            # Ensure no remaining NaN or infinite values
            last_msg = np.nan_to_num(last_msg, nan=-1.0, posinf=0.0, neginf=0.0)
            uptime = np.nan_to_num(uptime, nan=-1.0, posinf=0.0, neginf=0.0)
            
            node_targets = torch.tensor(
                np.column_stack([last_msg, uptime]), 
                dtype=torch.float32
            )
            
            # Create mask for nodes to include in loss (exclude root node)
            # Root is mote==1, so it's at index 0 after sorting by mote ID
            train_mask = torch.tensor(results_df['mote'].values != 1, dtype=torch.bool)
            
            # Graph features: selected features only
            if self.graph_features:
                graph_features_tensor = torch.tensor(
                    [graph_features[feat] for feat in self.graph_features], 
                    dtype=torch.float32
                )
            else:
                graph_features_tensor = torch.tensor([[]], dtype=torch.float32)  # 2D empty tensor
            
            # Graph target: coverage_30
            graph_target = torch.tensor(
                [graph_features['coverage_30']], 
                dtype=torch.float32
            )
            
            # Create edge index based on distance (nodes within 50m)
            num_nodes = len(results_df)
            if num_nodes < 2:
                return None
            
            # Get coordinates
            coords = results_df[['x', 'y']].values
            
            # Calculate pairwise distances and create edges
            edge_list = []
            edge_distances = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):  # Only check upper triangle
                    # Calculate Euclidean distance
                    dist = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                    
                    # Connect if within 50m
                    if dist <= 50.0:
                        edge_list.append([i, j])
                        edge_distances.append(dist)
                        edge_list.append([j, i])  # Add both directions for undirected graph
                        edge_distances.append(dist)  # Same distance for both directions
            
            if len(edge_list) == 0:
                # If no edges found, create a minimal connected graph
                # Connect each node to its nearest neighbor
                for i in range(num_nodes):
                    if i < num_nodes - 1:
                        dist = np.sqrt((coords[i, 0] - coords[i+1, 0])**2 + (coords[i, 1] - coords[i+1, 1])**2)
                        edge_list.append([i, i + 1])
                        edge_distances.append(dist)
                        edge_list.append([i + 1, i])
                        edge_distances.append(dist)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)  # Shape: [num_edges, 1]
            
            # Create Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,  # Distance between connected nodes
                node_targets=node_targets,
                train_mask=train_mask,  # Mask to exclude root from loss
                graph_features=graph_features_tensor,
                graph_targets=graph_target.unsqueeze(0)
            )
            
            return data
            
        except Exception as e:
            print(f"Error creating data object: {e}")
            return None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def create_data_loaders(
    seed_file: str,
    runs_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: str = 'train',
    val_split: str = 'val',
    test_split: str = 'test',
    node_features: list = ['x', 'y', 'initial_battery'],
    graph_features: list = [],
    target_type: str = 'node',
    target_feature: str = 'last_msg_recv_by_root'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        seed_file: Path to seed CSV file
        runs_dir: Directory containing run results
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_split: Name of train split column value
        val_split: Name of validation split column value
        test_split: Name of test split column value
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = BatteryRPLDataset(seed_file, runs_dir, split=train_split, 
                                    node_features=node_features, graph_features=graph_features,
                                    target_type=target_type, target_feature=target_feature)
    val_dataset = BatteryRPLDataset(seed_file, runs_dir, split=val_split,
                                  node_features=node_features, graph_features=graph_features,
                                  target_type=target_type, target_feature=target_feature)
    test_dataset = BatteryRPLDataset(seed_file, runs_dir, split=test_split,
                                   node_features=node_features, graph_features=graph_features,
                                   target_type=target_type, target_feature=target_feature)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Data]) -> Batch:
    """Custom collate function for batching graphs."""
    return Batch.from_data_list(batch)


if __name__ == "__main__":
    # Test data loading
    seed_file = "../seed1-30.csv"
    runs_dir = "../runs"
    
    # Create datasets
    train_dataset = BatteryRPLDataset(seed_file, runs_dir, split='train')
    val_dataset = BatteryRPLDataset(seed_file, runs_dir, split='val')
    test_dataset = BatteryRPLDataset(seed_file, runs_dir, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loading
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample graph: {sample}")
        print(f"Node features shape: {sample.x.shape}")
        print(f"Edge index shape: {sample.edge_index.shape}")
        print(f"Node targets shape: {sample.node_targets.shape}")
        print(f"Graph features shape: {sample.graph_features.shape}")
        print(f"Graph targets shape: {sample.graph_targets.shape}")

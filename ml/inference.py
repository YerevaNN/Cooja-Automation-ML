#!/usr/bin/env python3
"""
Inference script for GNN model predictions.
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from gnn_model import GraphSAGEModel
from data_loader import BatteryRPLDataset
import numpy as np
from tqdm import tqdm
import time


def load_model(checkpoint_path: str, target_type: str, target_feature: str, 
               node_features: list, graph_features: list, hyperparams: dict = None) -> GraphSAGEModel:
    """Load a trained model from checkpoint."""
    
    # Use provided hyperparameters or defaults
    if hyperparams is None:
        hyperparams = {}
    
    default_params = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'hidden_dim': 128,
        'num_layers': 16,
        'dropout': 0.1,
        'max_epochs': 100,
        'warmup_epochs': 10
    }
    
    # Merge provided hyperparameters with defaults
    model_params = {**default_params, **hyperparams}
    
    # Try multiple loading strategies
    try:
        # Strategy 1: Try loading with map_location and strict=False
        print("Trying to load checkpoint with strict=False...")
        model = GraphSAGEModel.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False,  # Allow missing keys
            target_type=target_type,
            target_feature=target_feature,
            node_features=node_features,
            graph_features=graph_features,
            **model_params
        )
        print("Successfully loaded model from checkpoint!")
        
    except Exception as e1:
        print(f"First loading attempt failed: {e1}")
        
        try:
            # Strategy 2: Try loading without strict=False but with hyperparameters
            print("Trying to load checkpoint with explicit hyperparameters...")
            model = GraphSAGEModel.load_from_checkpoint(
                checkpoint_path,
                map_location='cpu',
                target_type=target_type,
                target_feature=target_feature,
                node_features=node_features,
                graph_features=graph_features,
                **model_params
            )
            print("Successfully loaded model from checkpoint!")
            
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            print("Trying manual checkpoint loading...")
            
            # Strategy 3: Manual checkpoint loading
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Determine number of layers
            if hyperparams and hyperparams.get('num_layers') is not None:
                num_layers = hyperparams['num_layers']
                print(f"Using specified number of layers: {num_layers}")
            else:
                # Auto-detect from checkpoint
                conv_keys = [k for k in checkpoint.keys() if k.startswith('convs.')]
                num_layers = max([int(k.split('.')[1]) for k in conv_keys]) + 1 if conv_keys else 3
                print(f"Auto-detected {num_layers} layers from checkpoint")

            # Override num_layers with the actual value
            actual_params = model_params.copy()
            actual_params['num_layers'] = num_layers


            # Create model with correct number of layers
            model = GraphSAGEModel(
                target_type=target_type,
                target_feature=target_feature,
                node_features=node_features,
                graph_features=graph_features,
                **actual_params
            )
            
            # Load state dict if available
            if 'state_dict' in checkpoint:
                # Extract number of layers from state_dict
                state_dict = checkpoint['state_dict']
                conv_keys = [k for k in state_dict.keys() if k.startswith('convs.')]
                if conv_keys:
                    num_layers_from_checkpoint = max([int(k.split('.')[1]) for k in conv_keys]) + 1
                    # Determine number of layers
                    if hyperparams and hyperparams.get('num_layers') is not None:
                        num_layers = hyperparams['num_layers']
                        print(f"Using specified number of layers: {num_layers}")
                    else:
                        num_layers = num_layers_from_checkpoint
                        print(f"Auto-detected {num_layers} layers from state_dict")

                    # Recreate model with correct number of layers if needed
                    if num_layers != model_params.get('num_layers', 16):
                        actual_params = model_params.copy()
                        actual_params['num_layers'] = num_layers

                        model = GraphSAGEModel(
                            target_type=target_type,
                            target_feature=target_feature,
                            node_features=node_features,
                            graph_features=graph_features,
                            **actual_params
                        )

                model.load_state_dict(state_dict)
                print("Successfully loaded state dict manually!")
            elif isinstance(checkpoint, dict) and any(key.startswith(('node_encoder', 'graph_encoder', 'convs', 'predictor')) for key in checkpoint.keys()):
                # Direct PyTorch save (weights at root level)
                print("Detected direct PyTorch save, loading weights directly...")
                model.load_state_dict(checkpoint)
                print("Successfully loaded direct PyTorch save!")
            else:
                print("Warning: No state_dict or direct weights found in checkpoint")
    
    # Set to evaluation mode
    model.eval()
    model.freeze()
    
    return model


def get_denormalization_stats():
    """Get hardcoded normalization statistics for denormalization.
    These MUST match the stats in gnn_model.py exactly!
    """
    stats = {
        # Node-level input features (from gnn_model.py)
        'x': {'mean': 100.186616, 'std': 70.712870},
        'y': {'mean': 99.960732, 'std': 70.283916},
        'initial_battery': {'mean': 63.045146, 'std': 21.638183},
        
        # Graph-level input features
        'spacing': {'mean': 24.980241, 'std': 3.423190},
        'N': {'mean': 52.005644, 'std': 16.119981},
        
        # Node-level outputs (targets - NOT normalized during training)
        'last_msg_recv_by_root': {'mean': 186.199127, 'std': 64.326435},
        'uptime': {'mean': 273.415317, 'std': 41.106432},
        
        # Graph-level outputs
        'coverage_30': {'mean': 50436.935508, 'std': 14089.192902},
    }
    return stats


def denormalize_value(value: float, feature_name: str) -> float:
    """Denormalize a single value using hardcoded statistics."""
    stats = get_denormalization_stats()
    
    if feature_name not in stats:
        raise ValueError(f"Unknown feature for denormalization: {feature_name}")
    
    mean = stats[feature_name]['mean']
    std = stats[feature_name]['std']
    
    # Denormalize: original = normalized * std + mean
    return value * std + mean


def run_inference(model, data_loader: DataLoader, 
                 seed_df: pd.DataFrame, target_type: str, target_feature: str, batch_size: int, node_features_list: list):
    """Run inference and collect predictions.
    
    For node-level tasks, predictions for all nodes (including root) are returned,
    but metrics are calculated only on non-root nodes.
    
    Returns:
        tuple: (results_df, total_time, num_graphs)
    """
    
    metadata = []
    num_graphs = 0
    
    model.eval()
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running inference")):
            # Get predictions using the correct forward signature
            pred = model(batch)
            
            # Get ground truth for the target feature
            if target_type == 'node':
                # Map target feature to column index in node_targets
                target_mapping = {'last_msg_recv_by_root': 0, 'uptime': 1}
                target_idx = target_mapping[target_feature]
                gt = batch.node_targets[:, target_idx]
                
                # Get train mask (False for root nodes)
                train_mask = batch.train_mask if hasattr(batch, 'train_mask') else torch.ones(len(gt), dtype=torch.bool)
                
            elif target_type == 'graph':
                gt = batch.graph_targets.squeeze()
                
                if target_feature != 'coverage_30':
                    raise ValueError(f"Unknown graph target: {target_feature}")
            else:
                raise ValueError(f"Unknown target type: {target_type}")
            
            # Count graphs in this batch
            if target_type == 'graph':
                num_graphs += len(pred)
            else:
                num_graphs += 1  # For node-level tasks, each batch is typically one graph
            
            # Handle batched predictions and ground truth
            if target_type == 'node':
                # For node-level predictions, handle each node in the batch
                num_nodes = len(pred)
                for i in range(num_nodes):
                    # Denormalize predictions only (ground truth is already in original scale)
                    node_pred_denorm = denormalize_value(pred[i].cpu().item(), target_feature)
                    node_gt_original = gt[i].cpu().item()  # Ground truth is already in original scale
                    is_root = not train_mask[i].cpu().item()  # Root nodes have train_mask=False

                    # Get input features for this node (denormalize them)
                    # Map features based on the selected node features
                    feature_mapping = {'x': 0, 'y': 1, 'initial_battery': 2}
                    node_feat_dict = {}
                    
                    for j, feature in enumerate(node_features_list):
                        if feature in feature_mapping:
                            feature_idx = feature_mapping[feature]
                            if feature_idx < batch.x.size(1):
                                node_feat_dict[feature] = batch.x[i, feature_idx].cpu().item()
                                node_feat_dict[f"{feature}_denorm"] = denormalize_value(node_feat_dict[feature], feature)
                    
                    # Set defaults for missing features
                    node_x_denorm = node_feat_dict.get('x_denorm', 0.0)
                    node_y_denorm = node_feat_dict.get('y_denorm', 0.0)
                    node_battery_denorm = node_feat_dict.get('initial_battery_denorm', 0.0)

                    # Note: We can't directly map back to seed_df rows for node-level data
                    # Each graph has multiple nodes, so we need to track this differently
                    metadata.append({
                        'batch_idx': batch_idx,
                        'node_idx': i,
                        'is_root': is_root,
                        'x': node_x_denorm,
                        'y': node_y_denorm,
                        'initial_battery': node_battery_denorm,
                        'ground_truth': node_gt_original,
                        'predicted': node_pred_denorm
                    })
                        
            else:  # graph-level
                # For graph-level predictions, handle each graph in the batch
                batch_size = len(pred)
                for i in range(batch_size):
                    # Denormalize predictions only (ground truth is already in original scale)
                    graph_pred_denorm = denormalize_value(pred[i].cpu().item(), target_feature)
                    graph_gt_original = gt[i].cpu().item()  # Ground truth is already in original scale

                    # Get metadata for this graph
                    meta_idx = batch_idx * batch_size + i
                    if meta_idx < len(seed_df):
                        row = seed_df.iloc[meta_idx]
                        
                        # Get graph features for this graph (if available)
                        if (batch.graph_features is not None and 
                            batch.graph_features.numel() > 0 and 
                            batch.graph_features.size(1) > 0):
                            graph_spacing = batch.graph_features[i, 0].cpu().item()
                            graph_N = batch.graph_features[i, 1].cpu().item()
                            # Denormalize graph features
                            graph_spacing_denorm = denormalize_value(graph_spacing, 'spacing')
                            graph_N_denorm = denormalize_value(graph_N, 'N')
                        else:
                            # Use metadata values if graph features not available
                            graph_spacing_denorm = row['spacing']
                            graph_N_denorm = row['N']
                        metadata.append({
                            'path': row['path'],
                            'N': row['N'],
                            'spacing': row['spacing'],
                            'x': 0.0,  # Not applicable for graph-level
                            'y': 0.0,  # Not applicable for graph-level
                            'initial_battery': 0.0,  # Not applicable for graph-level
                            'ground_truth': graph_gt_original,
                            'predicted': graph_pred_denorm
                        })
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    return pd.DataFrame(metadata), total_time, num_graphs


def main():
    parser = argparse.ArgumentParser(description='Run GNN inference and save predictions')
    
    # Required arguments
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--seed-file', type=str, required=True,
                        help='Path to seed CSV file (e.g., seed1-30.csv)')
    parser.add_argument('--runs-dir', type=str, required=True,
                        help='Directory containing run results')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Data split to run inference on')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Output CSV file path for predictions')
    
    # Model configuration (should match training)
    parser.add_argument('--target-type', type=str, required=True, choices=['node', 'graph'],
                        help='Target type (node or graph)')
    parser.add_argument('--target-feature', type=str, required=True,
                        help='Target feature name')
    parser.add_argument('--node-features', nargs='+', default=['x', 'y', 'initial_battery'],
                        help='Node features to use')
    parser.add_argument('--graph-features', nargs='*', default=[],
                        help='Graph features to use (optional)')
    
    # Data loading arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes')
    
    # Model hyperparameters (fallback if not in checkpoint)
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (fallback if not in checkpoint)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (fallback if not in checkpoint)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (fallback if not in checkpoint)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers (auto-detected from checkpoint if not specified)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (fallback if not in checkpoint)')
    
    args = parser.parse_args()
    
    # Validate target feature based on target type
    if args.target_type == 'node':
        valid_node_targets = ['last_msg_recv_by_root', 'uptime']
        if args.target_feature not in valid_node_targets:
            raise ValueError(f"Invalid node target: {args.target_feature}. Must be one of {valid_node_targets}")
    elif args.target_type == 'graph':
        valid_graph_targets = ['coverage_30']
        if args.target_feature not in valid_graph_targets:
            raise ValueError(f"Invalid graph target: {args.target_feature}. Must be one of {valid_graph_targets}")
    
    print(f"Loading model from: {args.checkpoint_path}")
    print(f"Target: {args.target_type} - {args.target_feature}")
    print(f"Node features: {args.node_features}")
    print(f"Graph features: {args.graph_features}")
    print(f"Split: {args.split}")
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        target_type=args.target_type,
        target_feature=args.target_feature,
        node_features=args.node_features,
        graph_features=args.graph_features,
        hyperparams={
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    )
    
    # Create dataset for the specified split
    dataset = BatteryRPLDataset(
        seed_file=args.seed_file,
        runs_dir=args.runs_dir,
        split=args.split,
        node_features=args.node_features,
        graph_features=args.graph_features,
        target_type=args.target_type,
        target_feature=args.target_feature
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0] if len(batch) == 1 else batch,  # Handle single batch
        pin_memory=True
    )
    
    # Load seed metadata for the split
    seed_df = pd.read_csv(args.seed_file)
    seed_df = seed_df[seed_df['split'] == args.split]
    
    print(f"Running inference on {len(dataset)} graphs...")
    
    # Run inference
    results_df, total_time, num_graphs = run_inference(model, data_loader, seed_df, args.target_type, args.target_feature, args.batch_size, args.node_features)
    
    # Print timing information
    avg_time_per_graph = total_time / num_graphs if num_graphs > 0 else 0
    print(f"\n⏱️  Inference Timing:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Number of graphs: {num_graphs}")
    print(f"  Average time per graph: {avg_time_per_graph*1000:.2f} ms")
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to: {args.output_csv}")
    print(f"Total predictions: {len(results_df)}")
    
    # Print statistics
    if len(results_df) > 0:
        # For node-level tasks, calculate metrics only on non-root nodes
        if args.target_type == 'node' and 'is_root' in results_df.columns:
            non_root_df = results_df[~results_df['is_root']]
            root_df = results_df[results_df['is_root']]
            
            print(f"\nDataset composition:")
            print(f"  Total nodes: {len(results_df)}")
            print(f"  Non-root nodes: {len(non_root_df)}")
            print(f"  Root nodes: {len(root_df)}")
            
            if len(non_root_df) > 0:
                mae = np.mean(np.abs(non_root_df['predicted'] - non_root_df['ground_truth']))
                mse = np.mean((non_root_df['predicted'] - non_root_df['ground_truth'])**2)
                rmse = np.sqrt(mse)
                
                print(f"\nPrediction Statistics (non-root nodes only):")
                print(f"  MAE: {mae:.4f}")
                print(f"  MSE: {mse:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                
                # Additional stats
                print(f"\n  Ground truth stats:")
                print(f"    Mean: {non_root_df['ground_truth'].mean():.4f}")
                print(f"    Std: {non_root_df['ground_truth'].std():.4f}")
                print(f"    Min: {non_root_df['ground_truth'].min():.4f}")
                print(f"    Max: {non_root_df['ground_truth'].max():.4f}")
        else:
            # Graph-level or no root tracking
            mae = np.mean(np.abs(results_df['predicted'] - results_df['ground_truth']))
            mse = np.mean((results_df['predicted'] - results_df['ground_truth'])**2)
            rmse = np.sqrt(mse)
            
            print(f"\nPrediction Statistics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()

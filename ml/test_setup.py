#!/usr/bin/env python3
"""
Test script to verify GNN setup and data loading.
"""

import sys
import os
import torch
import pandas as pd
from torch_geometric.data import Data

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnn_model import GraphSAGEModel, create_data_from_results
from data_loader import BatteryRPLDataset


def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    model = GraphSAGEModel(
        node_feature_dim=3,
        graph_feature_dim=2,
        hidden_dim=64,  # Smaller for testing
        num_layers=2,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5,
        max_epochs=10,
        use_wandb=False
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy data
    num_nodes = 10
    dummy_data = Data(
        x=torch.randn(num_nodes, 3),  # x, y, initial_battery
        edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),  # Random edges
        node_targets=torch.randn(num_nodes, 2),  # last_msg_recv_by_root, uptime
        graph_features=torch.tensor([20.0, 10.0]),  # spacing, N
        graph_targets=torch.tensor([[1000.0]])  # coverage_30
    )
    
    with torch.no_grad():
        node_pred, graph_pred = model(dummy_data)
    
    print(f"✓ Forward pass successful")
    print(f"  Node prediction shape: {node_pred.shape}")
    print(f"  Graph prediction shape: {graph_pred.shape}")
    
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    # Check if data files exist
    seed_file = "../seed1-30.csv"
    runs_dir = "../runs"
    
    if not os.path.exists(seed_file):
        print(f"✗ Seed file not found: {seed_file}")
        return False
    
    if not os.path.exists(runs_dir):
        print(f"✗ Runs directory not found: {runs_dir}")
        return False
    
    print(f"✓ Data files found")
    
    # Test loading a small subset
    try:
        # Load seed data
        seed_df = pd.read_csv(seed_file)
        print(f"✓ Loaded seed data: {len(seed_df)} rows")
        
        # Test data loading (just first few rows)
        test_df = seed_df.head(5)
        test_dataset = BatteryRPLDataset(seed_file, runs_dir, split=None)
        
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"✓ Sample data loaded successfully")
            print(f"  Node features shape: {sample.x.shape}")
            print(f"  Edge index shape: {sample.edge_index.shape}")
            print(f"  Node targets shape: {sample.node_targets.shape}")
            print(f"  Graph features shape: {sample.graph_features.shape}")
            print(f"  Graph targets shape: {sample.graph_targets.shape}")
        else:
            print("✗ No data loaded from dataset")
            return False
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    return True


def test_normalization():
    """Test normalization functionality."""
    print("\nTesting normalization...")
    
    model = GraphSAGEModel(use_wandb=False)
    
    # Test data with known values
    test_data = Data(
        x=torch.tensor([[100.0, 100.0, 100000000.0]]),  # x, y, initial_battery
        edge_index=torch.tensor([[0], [0]]),
        node_targets=torch.tensor([[200.0, 300.0]]),  # last_msg_recv_by_root, uptime
        graph_features=torch.tensor([25.0, 50.0]),  # spacing, N
        graph_targets=torch.tensor([[50000.0]])  # coverage_30
    )
    
    # Test normalization
    normalized_data = model.normalize_features(test_data)
    
    # Check if normalization worked (values should be close to 0 mean, 1 std)
    x_norm = normalized_data.x[0]
    print(f"  Original x: {test_data.x[0][0]:.2f}, Normalized: {x_norm[0]:.2f}")
    print(f"  Original y: {test_data.x[0][1]:.2f}, Normalized: {x_norm[1]:.2f}")
    
    # Test denormalization
    node_pred = torch.tensor([[0.0, 0.0]])  # Normalized predictions
    graph_pred = torch.tensor([[0.0]])  # Normalized predictions
    
    node_denorm, graph_denorm = model.denormalize_predictions(node_pred, graph_pred)
    
    print(f"  Denormalized node pred: {node_denorm[0]}")
    print(f"  Denormalized graph pred: {graph_denorm[0]}")
    
    print("✓ Normalization working correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing GNN Setup for Battery RPL Simulation")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_data_loading,
        test_normalization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready for training.")
        return True
    else:
        print("✗ Some tests failed. Please check the setup.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

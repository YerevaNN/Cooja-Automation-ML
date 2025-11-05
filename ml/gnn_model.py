#!/usr/bin/env python3
"""
Graph Neural Network for Battery RPL Simulation Prediction
Uses PyTorch Lightning with GraphSAGE architecture for node and graph-level predictions.

Based on:
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. 
  Advances in neural information processing systems, 30.
- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. 
  arXiv preprint arXiv:1903.02428.

GraphSAGE Citation:
@inproceedings{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Rex and Leskovec, Jure},
  booktitle={Advances in neural information processing systems},
  pages={1024--1034},
  year={2017}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
from typing import Dict, Any, Tuple, Optional
import numpy as np


class GraphSAGEModel(pl.LightningModule):
    """
    GraphSAGE-based model for predicting node and graph-level properties.
    
    Architecture:
    - Node-level features: x, y coordinates, initial_battery
    - Graph-level features: spacing, N (number of nodes)
    - Node-level outputs: last_msg_recv_by_root, uptime
    - Graph-level outputs: coverage_30
    """
    
    def __init__(
        self,
        node_features: list = ['x', 'y', 'initial_battery'],  # Selected node features
        graph_features: list = [],  # Disabled graph features
        target_type: str = 'node',  # 'node' or 'graph'
        target_feature: str = 'last_msg_recv_by_root',  # For node: 'last_msg_recv_by_root' or 'uptime', for graph: 'coverage_30'
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 25,
        use_wandb: bool = True,
        project_name: str = "battery-rpl-gnn",
        run_name: str = None  # For display purposes
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store feature selections
        self.node_features = node_features
        self.graph_features = graph_features
        self.target_type = target_type
        self.target_feature = target_feature
        
        # Validate target selection
        if target_type == 'node':
            if target_feature not in ['last_msg_recv_by_root', 'uptime']:
                raise ValueError(f"Invalid node target: {target_feature}. Must be 'last_msg_recv_by_root' or 'uptime'")
        elif target_type == 'graph':
            if target_feature != 'coverage_30':
                raise ValueError(f"Invalid graph target: {target_feature}. Must be 'coverage_30'")
        else:
            raise ValueError(f"Invalid target_type: {target_type}. Must be 'node' or 'graph'")
        
        # Normalization constants (hardcoded from train subset analysis - excluding root nodes)
        self.normalization_stats = {
            # Node-level input features
            'x': {'mean': 100.186616, 'std': 70.712870},
            'y': {'mean': 99.960732, 'std': 70.283916},
            'initial_battery': {'mean': 63.045146, 'std': 21.638183},
            
            # Graph-level input features
            'spacing': {'mean': 24.980241, 'std': 3.423190},
            'N': {'mean': 52.005644, 'std': 16.119981},
            
            # Node-level outputs
            'last_msg_recv_by_root': {'mean': 186.199127, 'std': 64.326435},
            'uptime': {'mean': 273.415317, 'std': 41.106432},
            
            # Graph-level outputs
            'coverage_30': {'mean': 50436.935508, 'std': 14089.192902},
        }
        
        # Calculate feature dimensions based on selections
        self.node_feature_dim = len(node_features)
        self.graph_feature_dim = len(graph_features)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Output dimension is always 1 since we predict one target at a time
        self.output_dim = 1
        
        # Node feature encoder
        self.node_encoder = nn.Linear(self.node_feature_dim, hidden_dim)
        
        # Graph feature encoder (disabled)
        if self.graph_feature_dim > 0:
            self.graph_encoder = nn.Linear(self.graph_feature_dim, hidden_dim)
        else:
            self.graph_encoder = None
        
        # Initialize weights properly to avoid NaN
        self._init_weights()
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
        
        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        # Node-level prediction head
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )
        
        # Graph-level prediction head
        graph_input_dim = hidden_dim * 3  # Only pooling features (no graph features)
        self.graph_predictor = nn.Sequential(
            nn.Linear(graph_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )
        
        # Loss functions
        self.node_loss_fn = nn.MSELoss()
        self.graph_loss_fn = nn.MSELoss()
        
        # Metrics tracking
        self.train_node_losses = []
        self.train_graph_losses = []
        self.val_node_losses = []
        self.val_graph_losses = []
    
    def normalize_features(self, data: Data) -> Data:
        """Normalize input features using hardcoded statistics."""
        # Normalize node features based on selected features
        if hasattr(data, 'x') and data.x is not None:
            x_norm = data.x.clone()
            for i, feature in enumerate(self.node_features):
                if i < x_norm.shape[1] and feature in self.normalization_stats:
                    x_norm[:, i] = (x_norm[:, i] - self.normalization_stats[feature]['mean']) / self.normalization_stats[feature]['std']
            data.x = x_norm
        
        # Normalize graph features based on selected features
        if hasattr(data, 'graph_features') and data.graph_features is not None:
            graph_norm = data.graph_features.clone()
            for i, feature in enumerate(self.graph_features):
                if i < graph_norm.shape[0] and feature in self.normalization_stats:
                    graph_norm[i] = (graph_norm[i] - self.normalization_stats[feature]['mean']) / self.normalization_stats[feature]['std']
            data.graph_features = graph_norm
        
        return data
    
    def denormalize_predictions(self, pred: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions back to original scale."""
        denorm = pred.clone()
        if self.target_feature in self.normalization_stats:
            denorm = denorm * self.normalization_stats[self.target_feature]['std'] + self.normalization_stats[self.target_feature]['mean']
        return denorm
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the model."""
        # Normalize input features
        data = self.normalize_features(data)
        
        # Encode node features
        x = self.node_encoder(data.x)
        
        # GraphSAGE layers (note: SAGEConv doesn't support edge weights directly)
        # Edge distances are stored in data.edge_attr for future use if needed
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = dropout(x)
        
        if self.target_type == 'node':
            # Node-level predictions
            pred = self.node_predictor(x)
        else:  # graph
            # Graph-level pooling
            mean_pool = global_mean_pool(x, data.batch)
            max_pool = global_max_pool(x, data.batch)
            add_pool = global_add_pool(x, data.batch)
            
            # Graph features are disabled, use only pooling
            graph_input = torch.cat([mean_pool, max_pool, add_pool], dim=1)
            
            # Graph-level predictions
            pred = self.graph_predictor(graph_input)
        
        return pred
    
    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step."""
        pred = self(batch)
        
        # Get targets based on target type
        if self.target_type == 'node':
            # Map target feature to column index in node_targets
            target_mapping = {'last_msg_recv_by_root': 0, 'uptime': 1}
            target_idx = target_mapping[self.target_feature]
            targets = batch.node_targets[:, target_idx].unsqueeze(1)
            
            # Apply mask to exclude root nodes from loss
            if hasattr(batch, 'train_mask'):
                pred = pred[batch.train_mask]
                targets = targets[batch.train_mask]
        else:  # graph
            targets = batch.graph_targets
        
        # Check for NaN or infinite values in targets
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print(f"Warning: NaN or inf values in targets at batch {batch_idx}")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for NaN or infinite values in predictions
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"Warning: NaN or inf values in predictions at batch {batch_idx}")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

        # Denormalize predictions to compute loss in original scale
        pred_denorm = self.denormalize_predictions(pred)
        
        # Compute loss on denormalized predictions vs original targets (no target normalization)
        loss = self.node_loss_fn(pred_denorm, targets) if self.target_type == 'node' else self.graph_loss_fn(pred_denorm, targets)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or inf loss at batch {batch_idx}")
            loss = torch.tensor(1e6, device=loss.device, requires_grad=True)
        
        # Log loss
        self.log(f'train/{self.target_type}_{self.target_feature}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=targets.size(0))
        
        # Store for epoch-end logging
        if self.target_type == 'node':
            self.train_node_losses.append(loss.detach().cpu())
        else:
            self.train_graph_losses.append(loss.detach().cpu())
        
        return loss
    
    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        pred = self(batch)
        
        # Get targets based on target type
        if self.target_type == 'node':
            # Map target feature to column index in node_targets
            target_mapping = {'last_msg_recv_by_root': 0, 'uptime': 1}
            target_idx = target_mapping[self.target_feature]
            targets = batch.node_targets[:, target_idx].unsqueeze(1)
            
            # Apply mask to exclude root nodes from loss
            if hasattr(batch, 'train_mask'):
                pred = pred[batch.train_mask]
                targets = targets[batch.train_mask]
        else:  # graph
            targets = batch.graph_targets
        
        # Denormalize predictions to compute loss in original scale
        pred_denorm = self.denormalize_predictions(pred)

        # Compute loss on denormalized predictions vs original targets
        loss = self.node_loss_fn(pred_denorm, targets) if self.target_type == 'node' else self.graph_loss_fn(pred_denorm, targets)
        
        # Log loss
        self.log(f'val/{self.target_type}_{self.target_feature}_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=targets.size(0))
        
        # Store for epoch-end logging
        if self.target_type == 'node':
            self.val_node_losses.append(loss.detach().cpu())
        else:
            self.val_graph_losses.append(loss.detach().cpu())
        
        return loss
    
    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Test step."""
        pred = self(batch)
        
        # Get targets based on target type
        if self.target_type == 'node':
            # Map target feature to column index in node_targets
            target_mapping = {'last_msg_recv_by_root': 0, 'uptime': 1}
            target_idx = target_mapping[self.target_feature]
            targets = batch.node_targets[:, target_idx].unsqueeze(1)
            
            # Apply mask to exclude root nodes from loss
            if hasattr(batch, 'train_mask'):
                pred = pred[batch.train_mask]
                targets = targets[batch.train_mask]
        else:  # graph
            targets = batch.graph_targets
        
        # Denormalize predictions for evaluation
        pred_denorm = self.denormalize_predictions(pred)
        
        # Compute loss on denormalized predictions
        loss = self.node_loss_fn(pred_denorm, targets) if self.target_type == 'node' else self.graph_loss_fn(pred_denorm, targets)
        
        # Log loss
        self.log(f'test/{self.target_type}_{self.target_feature}_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=targets.size(0))
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.use_wandb and self.logger:
            # Log epoch-level metrics
            if self.train_node_losses:
                avg_node_loss = torch.stack(self.train_node_losses).mean()
                self.log('train/epoch_node_loss', avg_node_loss, on_epoch=True, logger=True)
            if self.train_graph_losses:
                avg_graph_loss = torch.stack(self.train_graph_losses).mean()
                self.log('train/epoch_graph_loss', avg_graph_loss, on_epoch=True, logger=True)
        
        # Clear stored losses
        self.train_node_losses.clear()
        self.train_graph_losses.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.use_wandb and self.logger:
            # Log epoch-level metrics
            if self.val_node_losses:
                avg_node_loss = torch.stack(self.val_node_losses).mean()
                self.log('val/epoch_node_loss', avg_node_loss, on_epoch=True, logger=True)
            if self.val_graph_losses:
                avg_graph_loss = torch.stack(self.val_graph_losses).mean()
                self.log('val/epoch_graph_loss', avg_graph_loss, on_epoch=True, logger=True)
        
        # Clear stored losses
        self.val_node_losses.clear()
        self.val_graph_losses.clear()
    
    def _init_weights(self):
        """Initialize model weights to prevent NaN values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Calculate total steps for step-level scheduling
        # Get estimated total steps from trainer, with better fallback
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # More reasonable fallback: assume 100 steps per epoch
            total_steps = self.trainer.max_epochs * 100
        warmup_steps = int(total_steps * self.warmup_epochs / self.max_epochs)
        
        # Calculate stable phase: 30% of total training time
        stable_epochs = int(0.3 * self.max_epochs)
        stable_steps = int(total_steps * stable_epochs / self.max_epochs)
        
        # Warmup-stable-decay learning rate schedule (step-level)
        # Phase 1: Warmup - linear increase from 0.1 to 1.0
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        
        # Phase 2: Stable - constant learning rate at 1.0
        stable_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=stable_steps
        )
        
        # Phase 3: Decay - cosine annealing
        decay_steps = total_steps - warmup_steps - stable_steps
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler],
            milestones=[warmup_steps, warmup_steps + stable_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def setup_wandb(self):
        """Setup wandb logging."""
        if self.use_wandb:
            # Get hyperparameters from hparams
            config = {
                'node_features': self.node_features,
                'graph_features': self.graph_features,
                'target_type': self.target_type,
                'target_feature': self.target_feature,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_epochs': self.warmup_epochs,
                'max_epochs': self.max_epochs
            }
            wandb.init(
                project=self.project_name,
                config=config,
                name=f"gnn_experiment_{wandb.util.generate_id()}"
            )
    
    def on_train_start(self):
        """Called when training starts."""
        if self.use_wandb:
            self.setup_wandb()


def create_data_from_results(results_df, graph_features):
    """
    Create PyTorch Geometric Data object from results DataFrame.
    
    Args:
        results_df: DataFrame with columns [mote, x, y, initial_battery, last_msg_recv_by_root, uptime]
        graph_features: Dict with spacing and N values
    
    Returns:
        Data object for PyTorch Geometric
    """
    # Node features: x, y, initial_battery
    node_features = torch.tensor(results_df[['x', 'y', 'initial_battery']].values, dtype=torch.float32)
    
    # Node targets: last_msg_recv_by_root, uptime
    node_targets = torch.tensor(results_df[['last_msg_recv_by_root', 'uptime']].values, dtype=torch.float32)
    
    # Graph features: spacing, N
    graph_features_tensor = torch.tensor([graph_features['spacing'], graph_features['N']], dtype=torch.float32)
    
    # Graph target: coverage_30
    graph_target = torch.tensor([graph_features['coverage_30']], dtype=torch.float32)
    
    # Create edge index (fully connected graph for now)
    num_nodes = len(results_df)
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        node_targets=node_targets,
        graph_features=graph_features_tensor,
        graph_targets=graph_target.unsqueeze(0)
    )
    
    return data


if __name__ == "__main__":
    # Example usage
    model = GraphSAGEModel(
        node_feature_dim=3,
        graph_feature_dim=2,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=10,
        max_epochs=100
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Model architecture:")
    print(model)

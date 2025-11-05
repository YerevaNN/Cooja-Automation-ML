#!/usr/bin/env python3
"""
Training script for Graph Neural Network on Battery RPL Simulation Data.
"""

import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb

from gnn_model import GraphSAGEModel
from data_loader import create_data_loaders


def create_run_name(args):
    """Create descriptive run name from all hyperparameters"""
    name_parts = []

    # Add task info
    name_parts.append(args.target_type)
    name_parts.append(args.target_feature)

    # Add model architecture
    name_parts.append(f"hd{args.hidden_dim}")
    name_parts.append(f"l{args.num_layers}")
    name_parts.append(f"do{args.dropout}")

    # Add training parameters
    name_parts.append(f"lr{args.learning_rate}")
    name_parts.append(f"bs{args.batch_size}")
    name_parts.append(f"wd{args.weight_decay}")
    name_parts.append(f"wu{args.warmup_epochs}")

    return "_".join(name_parts)

def main():
    parser = argparse.ArgumentParser(description='Train GNN on Battery RPL Simulation Data')
    
    # Data arguments
    parser.add_argument('--seed-file', type=str, default='../seed1-30.csv',
                        help='Path to seed CSV file')
    parser.add_argument('--runs-dir', type=str, default='../runs',
                        help='Directory containing run results')
    
    # Required arguments
    parser.add_argument('--run-name', type=str, required=True,
                        help='Name for this training run (used in checkpoint naming)')
    parser.add_argument('--checkpoint-dir', type=str, default='/nfs/np/mnt/ten/cooja-checkpoints/',
                        help='Directory to save checkpoints')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GraphSAGE layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Feature selection arguments
    parser.add_argument('--node-features', nargs='+', 
                        choices=['x', 'y', 'initial_battery'],
                        default=['x', 'y', 'initial_battery'],
                        help='Node features to include')
    parser.add_argument('--graph-features', nargs='*',
                        choices=['spacing', 'N'],
                        default=[],
                        help='Graph features to include (optional)')
    parser.add_argument('--target-type', type=str, choices=['node', 'graph'],
                        required=True, help='Target type: node or graph')
    parser.add_argument('--target-feature', type=str, required=True,
                        help='Target feature name (required)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--max-epochs', type=int, default=25,
                        help='Maximum number of epochs')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32-true',
                        help='Training precision (16-mixed, 32-true, bf16-mixed)')
    
    # Logging arguments
    parser.add_argument('--project-name', type=str, default='battery-rpl-gnn',
                        help='Wandb project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')

    # Wandb arguments
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='Use wandb logging')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                        help='Disable wandb logging')
    
    # Other arguments
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='Run a quick development run')
    
    args = parser.parse_args()
    
    # Validate target feature based on target type
    if args.target_type == 'node':
        if args.target_feature not in ['last_msg_recv_by_root', 'uptime']:
            parser.error("For node target type, --target-feature must be 'last_msg_recv_by_root' or 'uptime'")
    elif args.target_type == 'graph':
        if args.target_feature != 'coverage_30':
            parser.error("For graph target type, --target-feature must be 'coverage_30'")

    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = create_run_name(args)
    print(f"Run name: {args.run_name}")

    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        seed_file=args.seed_file,
        runs_dir=args.runs_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        node_features=args.node_features,
        graph_features=args.graph_features,
        target_type=args.target_type,
        target_feature=args.target_feature
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = GraphSAGEModel(
        node_features=args.node_features,
        graph_features=args.graph_features,
        target_type=args.target_type,
        target_feature=args.target_feature,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup logging
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            log_model=True
        )
    
    # Setup callbacks
    callbacks = []
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Model checkpointing - save only final checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.run_name}_final',
        save_top_k=0,  # Don't save best checkpoints
        save_last=True,  # Save the last checkpoint as last.ckpt
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=0.5,  # Gradient clipping
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        deterministic=True,
        benchmark=False  # Set to True for better performance if input sizes are consistent
    )
    
    # Train the model
    print("Starting training...")
    if args.ckpt_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # Save final checkpoint with correct name
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.run_name}_final.ckpt')
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Final model saved as checkpoint: {final_checkpoint_path}")
    
    # Test the model
    print("Testing model...")
    trainer.test(model, test_loader)
    
    # Close wandb if used
    if args.use_wandb and wandb.run is not None:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()

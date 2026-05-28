import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from axon.config import Config, DataConfig, TrainingConfig, ModelConfig, OutputConfig
from axon.model import build_model, AXON
from axon.dataset import calculate_class_weights, create_data_loaders
from axon.trainer import Trainer
from axon.utils import seed_everything


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AXON Training")
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--label_column', type=str, default='ACTION1_ID',
                        help='Column name for labels')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Distillation loss weight')
    parser.add_argument('--distillation_scale', type=float, default=0.05,
                        help='Scale for distillation loss in total loss')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to Hyperformer weights')
    parser.add_argument('--arch', type=str, default='ViT-B/16',
                        help='CLIP architecture')
    parser.add_argument('--pose_model', type=str, 
                        default='trainers.Hyperformer.Hyperformer_Model',
                        help='Hyperformer model class')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    
    # Config file (optional, for compatibility with original codebase)
    parser.add_argument('--config', '-cfg', type=str, default=None,
                        help='Path to config YAML file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AXON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  CSV Path: {args.csv_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Distillation Scale: {args.distillation_scale}")
    print(f"  Hyperformer Weights: {args.weights}")
    
    # Class names for action recognition
    class_names = [
        'Augmented Communication', 'Body', 'Face',
        'Gestures', 'Looking', 'Vocalization'
    ]
    num_classes = len(class_names)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.csv_path)
    
    # Build data loaders
    # NOTE: You need to import and use your actual build_dataloader function
    # This is a placeholder - replace with your actual data loading code
    try:
        from datasets.build import build_dataloader
        train_loader, val_loader = build_dataloader(
            df, batch_size=args.batch_size,
            label=args.label_column, seed=args.seed
        )
    except ImportError:
        print("ERROR: Could not import build_dataloader from datasets.build_mine")
        print("Please ensure your data loading module is available.")
        return
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Build model
    print("\nBuilding model...")
    
    # Load external config if provided
    external_config = None
    
    model = build_model(
        classnames=class_names,
        arch=args.arch,
        use_prompt=False,
        hyperformer_class=args.pose_model,
        hyperformer_weights=args.weights,
        config=external_config
    )
    model = model.cuda()
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(
        train_loader, num_classes,
        class_names=class_names, device='cuda'
    )
    

    print("Using default optimizer setup...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device='cuda',
        class_names=class_names,
        output_dir=output_dir
    )
    
    # Setup training
    trainer.setup_training(
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        class_weights=class_weights
    )
    
    # Train
    print("\nStarting training...")
    best_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        distillation_scale=args.distillation_scale,
        patience=args.patience,
        save_best=True
    )
    
    # Save results
    results = {
        'seed': args.seed,
        'best_accuracy': best_metrics.accuracy,
        'best_f1': best_metrics.f1,
        'best_uar': best_metrics.uar,
        'best_ece': best_metrics.ece,
        'best_epoch': best_metrics.epoch,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'distillation_scale': args.distillation_scale,
            'patience': args.patience
        }
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Accuracy: {best_metrics.accuracy:.2f}%")
    print(f"Best F1-Score: {best_metrics.f1:.2f}%")
    print(f"Best UAR: {best_metrics.uar:.2f}%")
    print(f"ECE at best: {best_metrics.ece:.4f}")
    print(f"Best Epoch: {best_metrics.epoch}")


if __name__ == "__main__":
    main()

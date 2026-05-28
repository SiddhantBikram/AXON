import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass

from .model import logsum_distance
from .utils import (
    AverageMeter, compute_topk_accuracy, compute_metrics,
    calibration_analysis, print_calibration_results, EarlyStopping
)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    accuracy: float = 0.0
    f1: float = 0.0
    uar: float = 0.0
    ece: float = 0.0
    epoch: int = 0


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        device: str,
        class_names: List[str],
        output_dir: str = "outputs",
        logger: Any = None
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.output_dir = output_dir
        self.logger = logger
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.distillation_loss_fn = None
        self.best_metrics = TrainingMetrics()
    
    def log(self, message: str):
        """Log message to logger if available, else print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def setup_training(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Setup training components.
        
        Args:
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            class_weights: Optional class weights for loss
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = criterion
        
        self.distillation_loss_fn = nn.MSELoss()
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        accumulation_steps: int = 1,
        distillation_scale: float = 0.05,
        log_interval: int = 50
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            accumulation_steps: Gradient accumulation steps
            distillation_scale: Scale for distillation loss
            log_interval: Steps between logging
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        num_steps = len(train_loader)
        batch_time = AverageMeter("Batch Time")
        cls_loss_meter = AverageMeter("Classification Loss")
        dist_loss_meter = AverageMeter("Distillation Loss")
        total_loss_meter = AverageMeter("Total Loss")
        
        start = time.time()
        end = time.time()
        
        pbar = tqdm(enumerate(train_loader), total=num_steps, 
                    desc=f"Epoch {epoch+1} Training", leave=False)
        
        for idx, batch_data in pbar:
            pose_data = batch_data[0]["imgs"].cuda(non_blocking=True)
            labels = batch_data[0]["label"].cuda(non_blocking=True)
            labels = labels.reshape(-1).long()
            
            # Forward pass
            pose_embeddings_distilled, pose_logits, text_embeddings = self.model(pose_data)
            
            # Classification loss
            cls_loss = self.criterion(pose_logits, labels)
            
            # Distillation loss
            selected_text_embeddings = text_embeddings[labels]
            dist_loss = logsum_distance(pose_embeddings_distilled, selected_text_embeddings)
            
            # Combined loss
            total_loss = cls_loss + distillation_scale * dist_loss
            
            # Handle gradient accumulation
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            
            if (idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update meters
            batch_size = labels.size(0)
            cls_loss_meter.update(cls_loss.item(), batch_size)
            dist_loss_meter.update(dist_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item() * accumulation_steps, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            lr = self.optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            
            pbar.set_postfix({
                'loss': f'{total_loss_meter.avg:.4f}',
                'cls': f'{cls_loss_meter.avg:.4f}',
                'dist': f'{dist_loss_meter.avg:.4f}',
                'lr': f'{lr:.6f}',
                'mem': f'{memory_used:.0f}MB'
            })
            
            if idx % log_interval == 0:
                etas = batch_time.avg * (num_steps - idx)
                self.log(
                    f'Train: [Epoch {epoch+1}][{idx}/{num_steps}] '
                    f'ETA {datetime.timedelta(seconds=int(etas))} '
                    f'LR {lr:.9f} '
                    f'Loss {total_loss_meter.avg:.4f} '
                    f'Cls {cls_loss_meter.avg:.4f} '
                    f'Dist {dist_loss_meter.avg:.4f} '
                    f'Mem {memory_used:.0f}MB'
                )
        
        epoch_time = time.time() - start
        self.log(f"Epoch {epoch+1} training took {datetime.timedelta(seconds=int(epoch_time))}")
        
        return {
            'total_loss': total_loss_meter.avg,
            'cls_loss': cls_loss_meter.avg,
            'dist_loss': dist_loss_meter.avg
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader,
        epoch: int = -1,
        save_calibration: bool = True
    ) -> Tuple[Dict[str, float], List[int], List[int], np.ndarray]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch (-1 for test mode)
            save_calibration: Whether to save calibration plot
            
        Returns:
            Tuple of (metrics_dict, y_true, y_pred, all_probs)
        """
        self.model.eval()
        
        acc1_meter = AverageMeter("Top1")
        acc2_meter = AverageMeter("Top2")
        acc3_meter = AverageMeter("Top3")
        
        y_pred = []
        y_true = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batch_data in pbar:
            pose_data = batch_data[0]["imgs"].cuda(non_blocking=True)
            labels = batch_data[0]["label"].cuda(non_blocking=True)
            labels = labels.reshape(-1)
            
            batch_size = labels.size(0)
            
            # Forward pass
            _, pose_logits, _ = self.model(pose_data)
            
            # Get probabilities
            probs = F.softmax(pose_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
            # Predictions
            _, preds = pose_logits.data.max(1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            
            # Top-k accuracy
            acc1, acc2, acc3 = compute_topk_accuracy(pose_logits, labels)
            acc1_meter.update(acc1, batch_size)
            acc2_meter.update(acc2, batch_size)
            acc3_meter.update(acc3, batch_size)
            
            pbar.set_postfix({
                'Top1': f'{acc1_meter.avg:.2f}%',
                'Top2': f'{acc2_meter.avg:.2f}%',
                'Top3': f'{acc3_meter.avg:.2f}%'
            })
        
        # Concatenate probabilities
        all_probs = np.vstack(all_probs)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, self.class_names, print_report=True)
        metrics['top1'] = acc1_meter.avg
        metrics['top2'] = acc2_meter.avg
        metrics['top3'] = acc3_meter.avg
        
        # Calibration analysis
        calibration_path = None
        if save_calibration:
            calibration_path = os.path.join(self.output_dir, f'calibration_epoch_{epoch+1}.png')
        
        ece, calibration_data = calibration_analysis(
            np.array(y_true), all_probs,
            self.class_names, calibration_path
        )
        metrics['ece'] = ece
        print_calibration_results(ece, calibration_data)
        
        # Print summary
        epoch_str = f"Epoch {epoch+1}" if epoch >= 0 else "Test"
        print(f'\n{"="*80}')
        print(f'RESULTS - {epoch_str}')
        print(f'{"="*80}')
        print(f'Top-1 Accuracy: {metrics["top1"]:.2f}%')
        print(f'Top-2 Accuracy: {metrics["top2"]:.2f}%')
        print(f'Top-3 Accuracy: {metrics["top3"]:.2f}%')
        print(f'Weighted F1-Score: {metrics["f1"]:.2f}%')
        print(f'UAR: {metrics["uar"]:.2f}%')
        print(f'ECE: {metrics["ece"]:.4f}')
        
        self.log(
            f'{epoch_str} | Top-1: {metrics["top1"]:.2f}% | '
            f'F1: {metrics["f1"]:.2f}% | UAR: {metrics["uar"]:.2f}% | '
            f'ECE: {ece:.4f}'
        )
        
        return metrics, y_true, y_pred, all_probs
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        accumulation_steps: int = 1,
        distillation_scale: float = 0.05,
        patience: int = 20,
        save_best: bool = True
    ) -> TrainingMetrics:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            accumulation_steps: Gradient accumulation steps
            distillation_scale: Scale for distillation loss
            patience: Early stopping patience
            save_best: Whether to save best model
            
        Returns:
            Best training metrics
        """
        early_stopping = EarlyStopping(patience=patience, mode='max')
        
        print(f"\n{'='*80}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Accumulation steps: {accumulation_steps}")
        print(f"Distillation scale: {distillation_scale}")
        print(f"Early stopping patience: {patience}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, epoch,
                accumulation_steps=accumulation_steps,
                distillation_scale=distillation_scale
            )
            
            # Validate
            val_metrics, y_true, y_pred, all_probs = self.validate(
                val_loader, epoch,
                save_calibration=True
            )
            
            # Check for improvement
            current_acc = val_metrics['accuracy']
            if current_acc > self.best_metrics.accuracy:
                self.best_metrics = TrainingMetrics(
                    accuracy=val_metrics['accuracy'],
                    f1=val_metrics['f1'],
                    uar=val_metrics['uar'],
                    ece=val_metrics['ece'],
                    epoch=epoch + 1
                )
                
                if save_best:
                    save_path = os.path.join(self.output_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
            
            # Early stopping
            if early_stopping(current_acc, epoch + 1):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Print final results
        print(f"\n{'#'*80}")
        print("TRAINING COMPLETE")
        print(f"{'#'*80}")
        print(f"Best Accuracy: {self.best_metrics.accuracy:.2f}% (Epoch {self.best_metrics.epoch})")
        print(f"Best F1-Score: {self.best_metrics.f1:.2f}%")
        print(f"Best UAR: {self.best_metrics.uar:.2f}%")
        print(f"ECE at best: {self.best_metrics.ece:.4f}")
        
        return self.best_metrics

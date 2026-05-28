import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    classification_report
)


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking training statistics.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 3
) -> Tuple[float, float, float]:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model predictions [B, num_classes]
        labels: Ground truth labels [B]
        k: Maximum k for top-k accuracy
        
    Returns:
        Tuple of (top1_acc, top2_acc, top3_acc)
    """
    k = min(k, logits.size(1))
    top_predictions = logits.topk(k, dim=1)[1]
    
    batch_size = labels.size(0)
    top1_correct = (top_predictions[:, 0] == labels).sum().item()
    top2_correct = ((top_predictions[:, :min(2, k)] == labels.unsqueeze(1)).any(dim=1)).sum().item()
    top3_correct = ((top_predictions[:, :min(3, k)] == labels.unsqueeze(1)).any(dim=1)).sum().item()
    
    return (
        top1_correct / batch_size * 100,
        top2_correct / batch_size * 100,
        top3_correct / batch_size * 100
    )


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    print_report: bool = True
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        print_report: Whether to print classification report
        
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    
    # Unweighted Average Recall (UAR)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    uar = np.mean(per_class_recall) * 100
    
    if print_report and class_names:
        print("\nClassification Report:")
        print("=" * 80)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        
        print("\nConfusion Matrix:")
        print("=" * 80)
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'uar': uar,
        'per_class_recall': per_class_recall.tolist()
    }


def calibration_analysis(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Perform calibration analysis.
    
    Computes Expected Calibration Error (ECE) and creates reliability diagram.
    
    Args:
        y_true: Ground truth labels
        probs: Predicted probabilities [N, num_classes]
        class_names: Optional class names
        output_path: Path to save calibration plot
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE, calibration_data)
    """
    # Ensure numpy arrays
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    
    # Handle list of arrays
    if probs.ndim == 1 and probs.dtype == object:
        try:
            probs = np.vstack(probs)
        except Exception:
            probs = np.asarray(list(probs))
    
    # Get predictions and confidences
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Calculate ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    calibration_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        samples_in_bin = np.sum(in_bin)
        prop_in_bin = float(samples_in_bin) / len(confidences) if len(confidences) > 0 else 0.0
        
        if samples_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_pred[in_bin] == y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_data.append({
                'bin': i,
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'gap': avg_confidence_in_bin - accuracy_in_bin,
                'samples': int(samples_in_bin)
            })
    
    # Create calibration plot
    if output_path:
        plt.figure(figsize=(10, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration
        if calibration_data:
            confs = [d['confidence'] for d in calibration_data]
            accs = [d['accuracy'] for d in calibration_data]
            plt.plot(confs, accs, 'o-', label=f'Model (ECE={ece:.3f})')
        
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Calibration Plot', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return ece, calibration_data


def print_calibration_results(ece: float, calibration_data: List[Dict]):
    """
    Print calibration analysis results.
    
    Args:
        ece: Expected Calibration Error
        calibration_data: Per-bin calibration data
    """
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f'Expected Calibration Error (ECE): {ece:.4f}')
    
    if calibration_data:
        print('\nPer-bin calibration:')
        print(f'{"Bin":<5} {"Confidence":<12} {"Accuracy":<10} {"Gap":<10} {"Samples":<10}')
        for data in calibration_data:
            print(f'{data["bin"]:<5} {data["confidence"]:<12.3f} {data["accuracy"]:<10.3f} '
                  f'{data["gap"]:<10.3f} {data["samples"]:<10}')


class EarlyStopping:
    """
    Early stopping utility.
    
    Args:
        patience: Number of epochs to wait before stopping
        mode: 'max' for metrics where higher is better, 'min' for lower
        delta: Minimum change to qualify as improvement
    """
    
    def __init__(self, patience: int = 20, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

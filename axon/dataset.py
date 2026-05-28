import numpy as np
import torch
from collections import Counter
from typing import List, Optional, Tuple, Any
from tqdm import tqdm


def calculate_class_weights(
    train_loader,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Calculate class weights for weighted cross entropy loss.
    
    Uses inverse frequency weighting to handle class imbalance.
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        class_names: Optional list of class names for logging
        device: Device to place the weights tensor
        
    Returns:
        Class weights tensor
    """
    class_counts = Counter()
    
    # Count samples per class
    print("Calculating class distribution...")
    for batch_data in tqdm(train_loader, desc="Counting classes"):
        labels = batch_data[0]["label"].cpu().numpy()
        labels = labels.reshape(-1)
        class_counts.update(labels)
    
    # Calculate weights (inverse frequency)
    counts = np.array([class_counts[i] for i in range(num_classes)])
    total_samples = sum(counts)
    
    # Inverse frequency weighting
    weights = total_samples / (num_classes * counts + 1e-8)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    # Convert to tensor
    weights = torch.FloatTensor(weights).to(device)
    
    # Print class distribution and weights
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    print("\nClass Distribution and Weights:")
    print("-" * 80)
    
    for i, (name, count, weight) in enumerate(zip(class_names, counts, weights)):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"Class {i} ({name}): {count} samples ({percentage:.2f}%), Weight: {weight:.4f}")
    
    print("-" * 80)
    print(f"Total samples: {total_samples}")
    
    return weights


def get_class_distribution(
    data_loader,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> dict:
    """
    Get class distribution from data loader.
    
    Args:
        data_loader: Data loader
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary with class distribution info
    """
    class_counts = Counter()
    
    for batch_data in data_loader:
        labels = batch_data[0]["label"].cpu().numpy()
        labels = labels.reshape(-1)
        class_counts.update(labels)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    total = sum(class_counts.values())
    distribution = {}
    
    for i in range(num_classes):
        count = class_counts[i]
        distribution[class_names[i]] = {
            'count': count,
            'percentage': (count / total * 100) if total > 0 else 0
        }
    
    return {
        'distribution': distribution,
        'total': total,
        'class_counts': dict(class_counts)
    }


def create_data_loaders(
    df,
    batch_size: int,
    label_column: str = 'ACTION1_ID',
    seed: int = 42,
    build_dataloader_fn: Any = None
) -> Tuple[Any, Any]:
    """
    Create train and validation data loaders.
    
    This is a wrapper that calls the actual build_dataloader function
    from your datasets module.
    
    Args:
        df: Pandas DataFrame with data
        batch_size: Batch size
        label_column: Column name for labels
        seed: Random seed
        build_dataloader_fn: Function to build data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if build_dataloader_fn is None:
        raise ValueError("build_dataloader_fn must be provided")
    
    return build_dataloader_fn(df, batch_size=batch_size, label=label_column, seed=seed)

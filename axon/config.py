import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # CLIP settings
    arch: str = "ViT-B/16"
    
    # Pose model
    pose_model: str = "trainers.Hyperformer.Hyperformer_Model"
    pose_weights_path: Optional[str] = None
    
    # Feature dimensions
    feature_dim: int = 512
    num_classes: int = 6
    
    # Prompt learning settings
    use_prompt_model: bool = False
    prompt_depth_vision: int = 0
    prompt_depth_text: int = 0
    n_ctx_vision: int = 0
    n_ctx_text: int = 4
    ctx_init: str = "a photo of a"
    zero_shot_eval: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training
    epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 1
    
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    
    # Loss weights
    alpha: float = 2.0  # Distillation loss weight
    distillation_scale: float = 0.05  # Scale for distillation in total loss
    
    # Early stopping
    patience: int = 20
    
    # Random seed
    seed: int = 42


@dataclass  
class DataConfig:
    """Data configuration."""
    
    csv_path: str = ""
    label_column: str = "ACTION1_ID"
    num_classes: int = 6
    test_size: float = 0.2
    
    # Class names for action recognition
    class_names: List[str] = field(default_factory=lambda: [
        'Augmented Communication', 'Body', 'Face', 
        'Gestures', 'Looking', 'Vocalization'
    ])


@dataclass
class OutputConfig:
    """Output configuration."""
    
    output_dir: str = "outputs"
    save_model: bool = True
    save_calibration_plots: bool = True
    log_interval: int = 50


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Device
    device: str = "cuda"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from command line arguments."""
        config = cls()
        
        # Update from args
        if hasattr(args, 'batch_size') and args.batch_size:
            config.training.batch_size = args.batch_size
        if hasattr(args, 'alpha'):
            config.training.alpha = args.alpha
        if hasattr(args, 'epochs'):
            config.training.epochs = args.epochs
        if hasattr(args, 'seed'):
            config.training.seed = args.seed
        if hasattr(args, 'output'):
            config.output.output_dir = args.output
        if hasattr(args, 'weights'):
            config.model.pose_weights_path = args.weights
        if hasattr(args, 'csv_path'):
            config.data.csv_path = args.csv_path
            
        return config


def parse_args() -> Tuple[argparse.Namespace, Config]:
    """Parse command line arguments and create config."""
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
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Distillation loss weight')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to Hyperformer weights')
    parser.add_argument('--arch', type=str, default='ViT-B/16',
                        help='CLIP architecture')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    args = parser.parse_args()
    config = Config.from_args(args)
    
    return args, config


# Label mappings for visualization
LABEL_NAMES = ['AAC', 'Body', 'Face', 'Gestures', 'Look', 'Vocalization']

# Upper body keypoint indices for visualization
UPPER_BODY_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]

# Body keypoint positions for visualization
BODY_KEYPOINT_POSITIONS = {
    0: (0.0, -0.6), 1: (0.0, -0.2), 2: (0.0, 0.5), 3: (0.0, 0.7),
    4: (-0.2, 0.5), 5: (-0.5, 0.3), 6: (-0.7, 0.0), 7: (-0.8, -0.1),
    8: (0.2, 0.5), 9: (0.5, 0.3), 10: (0.7, 0.0), 11: (0.8, -0.1),
    12: (-0.15, -0.6), 13: (-0.2, -1.0), 14: (-0.25, -1.4), 15: (-0.3, -1.5),
    16: (0.15, -0.6), 17: (0.2, -1.0), 18: (0.25, -1.4), 19: (0.3, -1.5),
    20: (0.0, 0.15), 21: (-0.9, -0.2), 22: (-0.7, 0.1), 23: (0.9, -0.2), 24: (0.7, 0.1)
}

# Upper body joint pairs for visualization
UPPER_BODY_JOINT_PAIRS = [
    [1, 20], [20, 2], [2, 3],  # Central spine/neck
    [2, 4], [4, 5], [5, 6], [6, 7],  # Left arm
    [6, 22], [7, 21],  # Left hand connections
    [2, 8], [8, 9], [9, 10], [10, 11],  # Right arm
    [10, 24], [11, 23]  # Right hand connections
]

from dataclasses import dataclass
from pathlib import Path
from typing import List
from src.models.model_original import GoogLeNet
from src.models.model_modified_v1 import GoogLeNetModifiedV1
from src.models.model_modified_v3 import GoogLeNetModifiedV3
from src.models.model_modified_v4 import GoogLeNetModifiedV4

@dataclass
class TrainingConfig:
    # Device
    device: str = "cuda"
    
    # Hyperparameters
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Dataset
    data_dir: str = "data/train"
    train_split: int = 80000
    val_split: int = 10000
    num_workers: int = 4
    
    # Model
    num_classes: int = 1000 
    aux_logits: bool = False
    
    # Model and save checkpoint
    Model = GoogLeNetModifiedV4
    checkpoint_path: Path = Path("checkpoint_googlenet_modified_v4.pth.tar")
    log_file: Path = Path("results/training_log_v4.csv")
    
    # mean and std for normalization
    mean: List[float] = None
    std: List[float] = None

    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]

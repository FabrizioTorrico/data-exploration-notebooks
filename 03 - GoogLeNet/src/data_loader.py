from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.config import TrainingConfig

def get_transforms(config: TrainingConfig) -> Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]),
    }

def get_dataloaders(config: TrainingConfig) -> Tuple[Dict[str, DataLoader], DataLoader]:
    data_transforms = get_transforms(config)
    
    full_dataset = datasets.ImageFolder(config.data_dir, transform=data_transforms["train"])
    
    test_split = len(full_dataset) - config.train_split - config.val_split
    
    train_data, val_data, test_data = random_split(
        full_dataset, [config.train_split, config.val_split, test_split]
    )
    
    common_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True
    }
    
    train_loader = DataLoader(train_data, shuffle=True, **common_args)
    val_loader = DataLoader(val_data, shuffle=False, **common_args)
    test_loader = DataLoader(test_data, shuffle=False, **common_args)
    
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    
    return dataloaders, test_loader, len(full_dataset.classes)

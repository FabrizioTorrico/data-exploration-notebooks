import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.config import TrainingConfig
from src.data_loader import get_dataloaders
from src.trainer import Trainer, test_model

def main():
    config = TrainingConfig()
    
    print(f"Using device: {config.device}")
    
    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # LOAD DATA
    dataloaders, test_loader, num_classes = get_dataloaders(config)
    print(f"{num_classes} classes.")
    
    # SETUP MODEL
    model = config.Model(in_channels=3, num_classes=num_classes, aux_logits=config.aux_logits)
    model.to(config.device)
    
    # OPTIMIZER & SCHEDULER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.learning_rate, 
        momentum=config.momentum, 
        weight_decay=config.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    
    # CHECKPOINT LOADING
    start_epoch = 0
    if config.checkpoint_path.exists():
        print(f"=> Loading checkpoint '{config.checkpoint_path}'...")
        checkpoint = torch.load(config.checkpoint_path)
        
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler and checkpoint["scheduler"]:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            
        print(f"=> Resumed from epoch {start_epoch}, Best Acc: {checkpoint['best_acc']:.4f}")
    else:
        print("=> No checkpoint found. Starting from scratch.")

    # TRAIN
    if start_epoch < config.epochs:
        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            lr_scheduler=lr_scheduler,
            start_epoch=start_epoch
        )
        trained_model, _ = trainer.train()
    else:
        print("Training already completed.")
        trained_model = model

    # TEST
    test_model(trained_model, test_loader, config.device)

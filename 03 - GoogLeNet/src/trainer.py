import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from src.config import TrainingConfig
import datetime
import csv
import os

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        lr_scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
        start_epoch: int = 0
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.device = torch.device(config.device)
        self.log_file = config.log_file
        
        # Initialize CSV
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "batch", "phase", "loss", "accuracy", "duration_sec"])

    def train(self) -> Tuple[nn.Module, List[float]]:
        val_acc_history: List[float] = []
        
        print(f"Starting training on {self.device} for {self.config.epochs} epochs.")

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                print(f"\nEpoch {epoch}/{self.config.epochs - 1}")
                print("-" * 10)

                train_loss, train_acc = self._train_one_epoch(epoch)
                val_loss, val_acc = self._validate_one_epoch()
                
                # Log Validation Stats (batch -1 to signify end of epoch summary)
                self._log_to_csv(epoch, -1, "val", val_loss, val_acc, 0.0)

                print(f"Epoch Summary -> Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

                val_acc_history.append(val_acc)
                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)

                self._save_checkpoint(epoch, val_acc)

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current best model...")
        
        self.model.load_state_dict(self.best_model_wts)
        return self.model, val_acc_history

    def _train_one_epoch(self, epoch_index: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        loader = self.dataloaders["train"]
        num_batches = len(loader)
        
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(loader):
            batch_start = time.time()
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if self.config.aux_logits:
                    outputs, aux1, aux2 = self.model(inputs)
                    loss1 = self.criterion(outputs, labels)
                    loss2 = self.criterion(aux1, labels)
                    loss3 = self.criterion(aux2, labels)
                    loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

            batch_loss = loss.item()
            batch_corrects = torch.sum(preds == labels.data).item()
            batch_size = inputs.size(0)
            batch_acc = batch_corrects / batch_size

            running_loss += batch_loss * batch_size
            running_corrects += batch_corrects
            total_samples += batch_size
            
            batch_duration = time.time() - batch_start
            

            
            # Detailed Logging per batch
            if batch_idx % 10 == 0:  # Log every 10 batches to avoid spamming too much, or make it configurable
                print(
                    f"[Epoch {epoch_index}][Batch {batch_idx}/{num_batches}] "
                    f"Time: {batch_duration:.3f}s | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Train Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples
        
        total_time = time.time() - start_time
        print(f"Train Phase Finished. Total Time: {total_time:.2f}s")
        # Log Training Stats (batch -1 to signify end of epoch summary)
        self._log_to_csv(epoch_index, -1, "train", epoch_loss, epoch_acc, total_time)
        
        return epoch_loss, epoch_acc

    def _validate_one_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        loader = self.dataloaders["val"]
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples
        
        return epoch_loss, epoch_acc
        
    def _log_to_csv(self, epoch, batch, phase, loss, acc, duration):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                batch,
                phase,
                f"{loss:.4f}",
                f"{acc:.4f}",
                f"{duration:.4f}"
            ])

    def _save_checkpoint(self, epoch: int, current_acc: float):
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "best_acc": self.best_acc,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        torch.save(checkpoint, self.config.checkpoint_path)
        print(f"Checkpoint saved to {self.config.checkpoint_path}")

def test_model(model: nn.Module, test_loader: DataLoader, device_str: str) -> float:
    device = torch.device(device_str)
    model.eval()
    correct = 0
    total = 0
    print("\nStarting Evaluation on Test Set...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Test Set Accuracy: {acc:.2f}%")
    return acc

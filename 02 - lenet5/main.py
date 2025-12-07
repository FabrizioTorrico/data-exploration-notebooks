# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
from model import LeNet5, LeNet5_Wide
from actions import get_training_data, train_model, test_model


batch_size = 64
num_classes = 10
learning_rate = 0.0008
num_epochs = 10


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

if __name__ == "__main__":
    train_loader, test_loader = get_training_data(batch_size)
    model = LeNet5_Wide(num_classes).to("cuda")
    cost = nn.CrossEntropyLoss()

    train_model(model, train_loader, num_epochs, learning_rate)

    test_model(model, test_loader)

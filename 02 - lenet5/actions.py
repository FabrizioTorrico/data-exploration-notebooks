import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


def get_training_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader


def train_model(model, train_loader, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    cost = nn.CrossEntropyLoss()

    times = []
    for epoch in range(num_epochs):
        torch.cuda.synchronize()  # Synchronize before starting the epoch
        start_epoch = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to("cuda")
            labels = labels.to("cuda")

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
        torch.cuda.synchronize()  # Synchronize after the epoch finishes
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    avg_time_per_epoch = sum(times) / num_epochs
    print(f"Average time per epoch: {avg_time_per_epoch:.4f} seconds")


def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

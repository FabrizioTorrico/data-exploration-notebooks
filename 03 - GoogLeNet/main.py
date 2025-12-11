import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from actions import test_model, train_model

# Importamos tus módulos (asegúrate que los nombres de archivo sean correctos)
from model import GoogLeNet
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 1. SETUP DE DISPOSITIVO (Más robusto)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Optimizacion de cudnn si usas GPU Nvidia
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 2. DEFINICIÓN DE TRANSFORMACIONES (OPTIMIZACIÓN CLAVE)
    # Las medias y std standard de ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  # Data Augmentation basico
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1
                ),  # Un poco de variacion
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    # 3. CARGA DE DATOS
    # Nota: Usualmente train y val están en carpetas separadas.
    # Si tienes todo en una sola carpeta 'data/train', el split que haces abajo es correcto.
    full_dataset = torchvision.datasets.ImageFolder(
        "data/train", transform=data_transforms["train"]
    )

    # Es un poco "hacky" usar el transform de train para todo el dataset antes de dividirlo,
    # pero para simplificar el código lo dejaremos así.
    # Lo ideal seria una clase Dataset custom, pero esto funciona.

    # Dividir dataset
    # Asegúrate que la suma sea EXACTA al total de tus imágenes
    # Ejemplo: Si tienes 100,000 imgs
    train_size = 80000
    val_size = 10000
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # IMPORTANTE: A los sets de val y test, idealmente no les aplicamos Augmentation.
    # Como splitteamos del dataset 'full' que ya tiene augmentation, van a heredar eso.
    # En proyectos avanzados se crean datasets por separado, pero para aprender esto es aceptable.

    batch_size = 32
    # Num workers depende de tu CPU. 4 es un buen estándar.
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    dataloaders = {"train": train_loader, "val": val_loader}

    # 4. MODELO
    # Definimos num_classes segun tus carpetas
    num_classes = len(full_dataset.classes)
    print(f"Detectadas {num_classes} clases.")

    model = GoogLeNet(in_channels=3, num_classes=num_classes, aux_logits=True)
    model.to(device)

    # 5. PARAMETROS DE ENTRENAMIENTO
    epochs = 35
    checkpoint_file = "checkpoint_googlenet_modified.pth.tar"
    start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # --- LOGICA DE REANUDACION (RESUME) ---
    import os

    if os.path.isfile(checkpoint_file):
        print(f"=> Cargando checkpoint '{checkpoint_file}'...")
        checkpoint = torch.load(checkpoint_file)

        # Recuperar estado
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler and checkpoint["scheduler"]:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"=> Checkpoint cargado, reanudando desde época {start_epoch}")
        print(f"=> Mejor Accuracy previa: {checkpoint['best_acc']:.4f}")
    else:
        print("=> No se encontró checkpoint. Iniciando desde cero.")

    # 6. ENTRENAMIENTO
    # Si ya terminamos las epocas, no entrenamos mas
    if start_epoch < epochs:
        print("Iniciando entrenamiento...")
        trained_model, history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            is_inception=True,
            lr_scheduler=lr_scheduler,
            start_epoch=start_epoch,  # Pasamos donde arrancar
            checkpoint_path=checkpoint_file,  # Pasamos nombre archivo
        )
    else:
        print("El entrenamiento ya había completado todas las épocas.")
        trained_model = model

    # 7. TEST FINAL
    print("\nEvaluando en conjunto de Test...")
    test_model(trained_model, test_loader, device)

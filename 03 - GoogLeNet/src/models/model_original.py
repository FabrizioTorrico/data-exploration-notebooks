import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) # BatchNorm no estaba en el paper 2014 (usaban LRN), pero es el estándar moderno.
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,        # Rama 1
        red_3x3, out_3x3, # Rama 2 (Reducción -> 3x3)
        red_5x5, out_5x5, # Rama 3 (Reducción -> 5x5)
        out_pool_proj   # Rama 4 (Pool -> Proyección)
    ):
        super(InceptionBlock, self).__init__()

        # Rama 1: 1x1 conv
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        # Rama 2: 1x1 conv (reducción) -> 3x3 conv
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # Rama 3: 1x1 conv (reducción) -> 5x5 conv
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # Rama 4: 3x3 MaxPool -> 1x1 conv (proyección)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1)
        )

    def forward(self, x):
        # Concatenamos a lo largo de la dimensión de canales (dim=1)
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 
            dim=1
        )

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # Paper: Average Pooling 5x5 stride 3
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        # Paper: 1x1 Conv con 128 filtros para reducción de dimensión + ReLU
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        
        # FC Layers
        # Input: 4x4 x 128 = 2048
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1) # Aplanado más limpio que reshape
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        # --- BLOQUE INICIAL (STEM) ---
        # Conv 7x7 (s=2) -> MaxPool 3x3 (s=2) -> Conv 1x1 (Red) -> Conv 3x3 -> MaxPool 3x3 (s=2)
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2 = ConvBlock(64, 64, kernel_size=1)  # Reducción 1x1 (El paso olvidado)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # --- INCEPTION BLOCKS (3a, 3b) ---
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # --- INCEPTION BLOCKS (4a - 4e) ---
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # --- INCEPTION BLOCKS (5a, 5b) ---
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # --- CLASIFICADOR AUXILIAR ---
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes) # Salida de 4a
            self.aux2 = InceptionAux(528, num_classes) # Salida de 4d
        else:
            self.aux1 = self.aux2 = None

        # --- CLASIFICADOR FINAL ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Se adapta a cualquier tamaño de entrada
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Stem
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))

        # Bloque 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Bloque 4
        x = self.inception4a(x)
        
        # Auxiliar 1 (Solo durante entrenamiento)
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliar 2 (Solo durante entrenamiento)
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Bloque 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Salida Final
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x

if __name__ == "__main__":
    model = GoogLeNet(num_classes=200)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parámetros GoogLeNet: {total_params:,}")
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- ARQUITECTURA "SPLIT + BOTTLENECK" ---

class SplitInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,        
        red_3x3, out_3x3, 
        red_5x5, out_5x5, 
        out_pool_proj,   
    ):
        super(SplitInceptionBlock, self).__init__()

        # División de proyección del pooling
        pool_proj_3x3 = out_pool_proj // 2
        pool_proj_5x5 = out_pool_proj - pool_proj_3x3

        # --- GRUPO 1: DETALLE (Small) ---
        self.branch_1x1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch_3x3 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch_pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj_3x3, kernel_size=1)
        )

        # OPTIMIZACIÓN: Bottleneck Fusion
        # Sumamos canales de las ramas 3x3 y Pool3
        channels_small = out_3x3 + pool_proj_3x3
        # Comprimimos a la mitad (Factor 0.5)
        self.fusion_small = ConvBlock(channels_small, channels_small // 2, kernel_size=1)


        # --- GRUPO 2: CONTEXTO (Large) ---
        self.branch_5x5 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch_pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            ConvBlock(in_channels, pool_proj_5x5, kernel_size=1)
        )

        channels_large = out_5x5 + pool_proj_5x5
        # Comprimimos a la mitad (Factor 0.5)
        self.fusion_large = ConvBlock(channels_large, channels_large // 2, kernel_size=1)

    def forward(self, x):
        # 1. Grupo Pequeño: 3x3 + Pool3 -> Fusión (Compresión)
        # Nota: branch_1x1 la dejamos fuera de la fusión para mantener una conexión pura
        small_out = torch.cat([self.branch_3x3(x), self.branch_pool3(x)], dim=1)
        small_out = self.fusion_small(small_out)

        # 2. Grupo Grande: 5x5 + Pool5 -> Fusión (Compresión)
        large_out = torch.cat([self.branch_5x5(x), self.branch_pool5(x)], dim=1)
        large_out = self.fusion_large(large_out)

        # 3. Concatenación Final
        return torch.cat([self.branch_1x1(x), small_out, large_out], dim=1)


class GoogLeNetModifiedV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=True):
        super(GoogLeNetModifiedV1, self).__init__()
        self.aux_logits = aux_logits

        # --- STEM ---
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # --- BLOQUES INCEPTION (Recalculados para Bottleneck) ---
        # La entrada (in_channels) de cada bloque debe coincidir con la salida comprimida del anterior.
        # Salida = out_1x1 + (out_3x3+pool3)/2 + (out_5x5+pool5)/2
        
        # 3a: In 192. Out = 64 + (128+16)/2 + (32+16)/2 = 64 + 72 + 24 = 160
        self.inception3a = SplitInceptionBlock(192, 64, 96, 128, 16, 32, 32)
        
        # 3b: In 160. Out = 128 + (192+32)/2 + (96+32)/2 = 128 + 112 + 64 = 304
        self.inception3b = SplitInceptionBlock(160, 128, 128, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 4a: In 304. Out = 192 + (208+32)/2 + (48+32)/2 = 192 + 120 + 40 = 352
        self.inception4a = SplitInceptionBlock(304, 192, 96, 208, 16, 48, 64)
        
        # 4b: In 352. Out = 160 + (224+32)/2 + (64+32)/2 = 160 + 128 + 48 = 336
        self.inception4b = SplitInceptionBlock(352, 160, 112, 224, 24, 64, 64)
        
        # 4c: In 336. Out = 128 + (256+32)/2 + (64+32)/2 = 128 + 144 + 48 = 320
        self.inception4c = SplitInceptionBlock(336, 128, 128, 256, 24, 64, 64)
        
        # 4d: In 320. Out = 112 + (288+32)/2 + (64+32)/2 = 112 + 160 + 48 = 320
        self.inception4d = SplitInceptionBlock(320, 112, 144, 288, 32, 64, 64)
        
        # 4e: In 320. Out = 256 + (320+64)/2 + (128+64)/2 = 256 + 192 + 96 = 544
        self.inception4e = SplitInceptionBlock(320, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 5a: In 544. Out = 256 + (320+64)/2 + (128+64)/2 = 256 + 192 + 96 = 544
        self.inception5a = SplitInceptionBlock(544, 256, 160, 320, 32, 128, 128)
        
        # 5b: In 544. Out = 384 + (384+64)/2 + (128+64)/2 = 384 + 224 + 96 = 704
        self.inception5b = SplitInceptionBlock(544, 384, 192, 384, 48, 128, 128)

        # --- CLASIFICADORES ---
        if self.aux_logits:
            self.aux1 = InceptionAux(352, num_classes) # Salida de 4a
            self.aux2 = InceptionAux(320, num_classes) # Salida de 4d
        else:
            self.aux1 = self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # Linear final recibe 704 en vez de 1024 (Ahorro masivo de params aqui)
        self.fc = nn.Linear(704, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x

# Verificación
if __name__ == "__main__":
    model = GoogLeNetModifiedV1(num_classes=200)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parámetros GoogLeNetModifiedV1: {total_params:,}")
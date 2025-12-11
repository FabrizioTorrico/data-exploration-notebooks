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

# --- BLOQUE DE REDUCCIÓN (Sustituye al MaxPool) ---
# Evita la destrucción de información usando conv con stride
class ReductionBlock(nn.Module):
    def __init__(self, in_channels, out_conv_channels):
        super(ReductionBlock, self).__init__()
        
        # Rama 1: MaxPool tradicional (Conserva picos de activación)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Rama 2: Convolución con Stride 2 (Aprende a reducir)
        self.branch_conv = ConvBlock(in_channels, out_conv_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Concatenamos los canales originales (pooleados) con los nuevos (convolucionados)
        # Salida Channels = In_Channels + Out_Conv_Channels
        return torch.cat([self.branch_pool(x), self.branch_conv(x)], dim=1)

# --- BLOQUE PARALELO (Inicio) ---
class ParallelResidualBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool_proj):
        super(ParallelResidualBlock, self).__init__()
        
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1)
        )

        total_out = out_1x1 + out_3x3 + out_5x5 + out_pool_proj
        self.shortcut = ConvBlock(in_channels, total_out, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        concat = torch.cat([b1, b2, b3, b4], 1)
        res = self.shortcut(x)
        return self.relu(concat + res)

# --- BLOQUE ESPACIAL (Cuerpo) ---
class SpatialResidualBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, out_pool_proj):
        super(SpatialResidualBlock, self).__init__()
        
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1)
        )

        total_out = out_1x1 + out_3x3 + out_pool_proj
        self.shortcut = ConvBlock(in_channels, total_out, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        pool = self.branch_pool(x)
        concat = torch.cat([b1, b2, pool], 1)
        res = self.shortcut(x)
        return self.relu(concat + res)


class GoogLeNetModifiedV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=False):
        super(GoogLeNetModifiedV3, self).__init__()

        # --- STEM ---
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # Salida: 192

        # --- FASE 1: PARALLEL (3x3 + 5x5) ---
        
        # 3a: In 192. Out = 64+128+32+32 = 256.
        self.inception3a = ParallelResidualBlock(192, 64, 96, 128, 16, 32, 32)
        
        # 3b: In 256. Out = 128+192+48+64 = 432.
        self.inception3b = ParallelResidualBlock(256, 128, 128, 192, 32, 48, 64)
        
        # REDUCCIÓN 1 (Reemplaza MaxPool3)
        # In: 432. 
        # Branch Pool: 432 channels. Branch Conv: 160 channels (Stride 2).
        # Total Out: 432 + 160 = 592.
        self.reduction3 = ReductionBlock(432, 160)

        # --- FASE 2: (Solo 3x3) ---
 
        self.inception4a = SpatialResidualBlock(592, 192, 144, 288, 64) 

        # In 592. Out = 192+288+64 = 544
        self.inception4b = SpatialResidualBlock(544, 192, 144, 288, 64)        
        self.inception4c = SpatialResidualBlock(544, 192, 144, 288, 64)       
        self.inception4d = SpatialResidualBlock(544, 192, 144, 288, 64)        
        self.inception4e = SpatialResidualBlock(544, 224, 160, 320, 96)        
        
        # REDUCCIÓN 2 
        self.reduction4 = ReductionBlock(640, 160)
        
        # --- FASE 3: FINAL ---
        self.inception5a = SpatialResidualBlock(800, 256, 160, 320, 96)        
        self.inception5b = SpatialResidualBlock(672, 320, 192, 384, 128)


        # --- CLASIFICADOR ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(832, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.reduction3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.reduction4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = GoogLeNetModifiedV3(num_classes=200)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parámetros GoogLeNet Modified (Grid Reduction): {total_params:,}")
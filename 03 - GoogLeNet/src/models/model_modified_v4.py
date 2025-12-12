"""
Modificaciones v4:
1. Se modifica el bloque ParallelStateBlock, se vuelve al paso de
la modificacion v1, con 2 ramas paralelas separadas, una de 3x3 y otra de 5x5.
Cada una de estas ramas tiene un maxpool, a diferencia de la v1, ahora se hace un pasaje
de parametros entre un bloque y el siguiente paralelo para evitar perdidad de 
informacion en el bottleneck reduction.
2. Se implementa el bloque SpatialResidualBlock en todos los maxpool
3. La rama paralela se utiliza unicamente al inicio del modelo,
al final del modelo se utiliza exclusivamente el bloque 3x3 con maxpool y pasaje 
residual.
"""
import torch
import torch.nn as nn

# --- BLOQUES AUXILIARES ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ReductionBlock(nn.Module):
    """
    Sustituye al MaxPool.
    """
    def __init__(self, in_channels, out_conv_channels):
        super(ReductionBlock, self).__init__()
        # Rama 1: MaxPool tradicional (Conserva picos)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Rama 2: Conv Stride 2 (Aprende reducción)
        self.branch_conv = ConvBlock(in_channels, out_conv_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch_pool(x), self.branch_conv(x)], dim=1)

class ParallelStateBlock(nn.Module):
    """
    Lineas paralelas 3x3 y 5x5 con maxpool y triple pasaje residual.
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool_proj, 
                 prev_state_3x3_ch=None, prev_state_5x5_ch=None):
        super(ParallelStateBlock, self).__init__()
        
        # --- Rama 1 ---
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        # --- Rama 3x3 ---
        self.branch3x3_conv = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3x3_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1)
        )
        
        dim_concat_3x3 = out_3x3 + out_pool_proj 

        # condicional para el estado anterior
        in_rec_3x3 = prev_state_3x3_ch if prev_state_3x3_ch is not None else dim_concat_3x3
        
        # Proyecta de (Estado Anterior) -> (Estado Actual)
        self.recurrence_3x3 = nn.Conv2d(in_rec_3x3, dim_concat_3x3, kernel_size=1)
        
        self.branch3x3_out = ConvBlock(dim_concat_3x3, out_3x3, kernel_size=1)

        # --- Rama 5x5 ---
        self.branch5x5_conv = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch5x5_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1)
        )
        
        # LÓGICA DE PROYECCIÓN DE ESTADO 5x5
        dim_concat_5x5 = out_5x5 + out_pool_proj
        in_rec_5x5 = prev_state_5x5_ch if prev_state_5x5_ch is not None else dim_concat_5x5
        
        self.recurrence_5x5 = nn.Conv2d(in_rec_5x5, dim_concat_5x5, kernel_size=1)
        
        self.branch5x5_out = ConvBlock(dim_concat_5x5, out_5x5, kernel_size=1)

        # --- Shortcut de bloque completo ---
        self.global_shortcut = ConvBlock(in_channels, in_channels, kernel_size=1)

    def forward(self, x, prev_state_3x3=None, prev_state_5x5=None):
        out_b1 = self.branch1(x)

        # Sistema 3x3
        b2_conv = self.branch3x3_conv(x)
        b2_pool = self.branch3x3_pool(x)
        concat2 = torch.cat([b2_conv, b2_pool], dim=1)
        
        if prev_state_3x3 is not None:
            concat2 = concat2 + self.recurrence_3x3(prev_state_3x3)
        
        current_state_3x3 = concat2
        out_b2 = self.branch3x3_out(concat2)

        # Sistema 5x5
        b3_conv = self.branch5x5_conv(x)
        b3_pool = self.branch5x5_pool(x)
        concat3 = torch.cat([b3_conv, b3_pool], dim=1)

        if prev_state_5x5 is not None:
            concat3 = concat3 + self.recurrence_5x5(prev_state_5x5)

        current_state_5x5 = concat3
        out_b3 = self.branch5x5_out(concat3)

        out_shortcut = self.global_shortcut(x)

        final_output = torch.cat([out_b1, out_b2, out_b3, out_shortcut], dim=1)
        return final_output, (current_state_3x3, current_state_5x5)


class SpatialResidualBlock(nn.Module):
    """
    Bloque estándar para fases posteriores y unicamente 3x3
    """
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

        # pasaje de parametros del estado anterior al actual mediante suma
        return self.relu(concat + res)

class GoogLeNetModifiedV4(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits: bool = False):
        super(GoogLeNetModifiedV4, self).__init__()

      
        self.stem_conv1 = ConvBlock(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.stem_conv2 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.stem_reduce1 = ReductionBlock(64, 64)   # Out: 128
        self.stem_reduce2 = ReductionBlock(128, 96)  # Out: 224

        # --- FASE 1: PARALLEL STATE ---
        
        # 3a: In 224.
        self.inception3a = ParallelStateBlock(224, out_1x1=64, red_3x3=64, out_3x3=96, red_5x5=32, out_5x5=32, out_pool_proj=32,
                                              prev_state_3x3_ch=None, prev_state_5x5_ch=None)
        # Out 3a: 416

        # 3b: In 416. 
        self.inception3b = ParallelStateBlock(416, out_1x1=96, red_3x3=96, out_3x3=128, red_5x5=48, out_5x5=64, out_pool_proj=64,
                                              prev_state_3x3_ch=128, prev_state_5x5_ch=64)
        # Out 3b: 704

        # --- REDUCCIÓN 3 ---
        # In: 704. Out: 704(pool) + 160(conv) = 864
        self.reduction3 = ReductionBlock(704, 160)


        # --- FASE 2: SPATIAL ---
        # Entrada 864 (antes era 960, gran ahorro en la matriz de pesos inicial de 4a)
        
        # 4a: Bajamos un poco la salida total
        self.inception4a = SpatialResidualBlock(864, out_1x1=128, red_3x3=96, out_3x3=192, out_pool_proj=96)
        # Out = 128 + 192 + 96 = 416
        
        # 4b
        self.inception4b = SpatialResidualBlock(416, out_1x1=144, red_3x3=112, out_3x3=224, out_pool_proj=96)
        # Out = 464

        # 4c
        self.inception4c = SpatialResidualBlock(464, out_1x1=144, red_3x3=112, out_3x3=224, out_pool_proj=96)
        # Out = 464
        
        # 4d
        self.inception4d = SpatialResidualBlock(464, out_1x1=160, red_3x3=128, out_3x3=256, out_pool_proj=96)
        # Out = 512
        
        # --- REDUCCIÓN 4 ---
        # In: 512. Out: 704
        self.reduction4 = ReductionBlock(512, 192)


        # --- FASE 3: FINAL ---        
        # 5a: In 704.
        self.inception5a = SpatialResidualBlock(704, out_1x1=192, red_3x3=160, out_3x3=320, out_pool_proj=128)
        # Out = 640

        # 5b: In 640.
        self.inception5b = SpatialResidualBlock(640, out_1x1=256, red_3x3=192, out_3x3=384, out_pool_proj=128)
        # Out = 256 + 384 + 128 = 768


        # --- CLASIFICADOR ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        # Entrada 768
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.stem_conv1(x)
        x = self.stem_conv2(x)
        x = self.stem_reduce1(x)
        x = self.stem_reduce2(x)
        
        # Fase 1
        x, (state_3x3, state_5x5) = self.inception3a(x, prev_state_3x3=None, prev_state_5x5=None)
        x, _ = self.inception3b(x, prev_state_3x3=state_3x3, prev_state_5x5=state_5x5)
        
        # Reduccion
        x = self.reduction3(x)

        # Fase 2
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        # Reduccion
        x = self.reduction4(x)

        # Fase 3
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Salida
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Test rápido de dimensiones
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224) 
    model = GoogLeNetModifiedV4(num_classes=1000)
    
    # Cálculo aproximado de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    output = model(dummy_input)
    print("Output shape:", output.shape)
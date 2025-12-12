# 🚗 Traffic Monitor & Speed Detection

Este proyecto implementa un sistema de monitoreo de tráfico capaz de detectar vehículos, rastrear su trayectoria y estimar su velocidad en tiempo real utilizando visión por computadora.

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green)
![Supervision](https://img.shields.io/badge/Supervision-Latest-orange)

## 📋 Descripción

El sistema procesa videos de tráfico para identificar vehículos (autos, motos, autobuses, camiones) dentro de una zona delimitada. Utilizando técnicas de transformación de perspectiva, calcula la velocidad de cada vehículo y detecta posibles infracciones de velocidad.

## ✨ Características Principales

*   **Detección de Vehículos**: Utiliza modelos YOLO (You Only Look Once) para identificar múltiples clases de vehículos.
*   **Seguimiento (Tracking)**: Mantiene la identidad de los vehículos a través de los frames usando ByteTrack.
*   **Estimación de Velocidad**: Calcula la velocidad en km/h mediante transformación de perspectiva (homografía) para mitigar la distorsión de la cámara.
*   **Detección de Infracciones**: Identifica vehículos que superan el límite de velocidad configurado.
*   **Visualización**: Genera un video de salida con cajas delimitadoras, etiquetas de velocidad y alertas visuales.

## 🛠️ Requisitos e Instalación

Este proyecto utiliza `uv` para la gestión de dependencias, pero también puede instalarse con `pip`.

### Prerrequisitos
- Python 3.14 o superior
- Soporte para GPU (Recomendado para inferencia rápida)

### Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd 04-car-detection
    ```

2.  **Instalar dependencias:**
    
    Si usas `uv` (recomendado):
    ```bash
    uv sync
    ```

    O usando `pip` estándar:
    ```bash
    pip install .
    ```

## ⚙️ Configuración

El archivo `config.py` contiene todas las variables ajustables del sistema:

*   **Paths**: Rutas de entrada (`SOURCE_VIDEO_PATH`) y salida (`TARGET_VIDEO_PATH`).
*   **Modelo**: Ruta al modelo YOLO (`MODEL_PATH`) y umbrales de confianza (`CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`).
*   **Clases**: IDs de las clases a detectar (2=auto, 3=moto, 5=bus, 7=camión).
*   **Velocidad**: 
    *   `SPEED_LIMIT_KMH`: Límite de velocidad para marcar infracciones.
    *   `TARGET_WIDTH`, `TARGET_HEIGHT`: Dimensiones del mundo real (en metros o unidades relativas) para la transformación de perspectiva.
*   **Zona de Detección**: `SOURCE_POLYGON` define el área de interés en el video.

## 🚀 Uso

1.  Asegúrate de tener el video de entrada en la ruta especificada en `config.py` (por defecto `data/vehicles.mp4`).
    > **Tip:** Si no tienes el video, puedes descargarlo ejecutando:
    > ```bash
    > python src/video_downloader.py
    > ```
2.  Ejecuta el script principal:

    ```bash
    python main.py
    ```
    
    O si usas `uv`:
    ```bash
    uv run main.py
    ```

3.  El sistema mostrará una ventana con el procesamiento en tiempo real y guardará el resultado en `data/vehicles-result-5.mp4` (o lo que hayas configurado). Presiona `q` para detener.

## 📂 Estructura del Proyecto

```
.
├── config.py               # Configuración global
├── main.py                 # Punto de entrada principal
├── pyproject.toml          # Dependencias y metadatos
├── src/
│   ├── vehicle_detector.py # Lógica de detección y tracking
│   ├── speed_system.py     # Cálculo de velocidad y transformación de perspectiva
│   ├── visualizer.py       # Dibujado de anotaciones en el frame
│   ├── infractors_manager.py # Gestión de infracciones y guardado de capturas
│   └── ...
└── data/                   # Directorio para videos y modelos
```

## � Funcionamiento

### 1. Transformación de Perspectiva
La cámara ve la carretera en perspectiva, lo que hace que las distancias lejanas parezcan más pequeñas que las cercanas (distorsión de perspectiva). Medir la velocidad directamente en píxeles sería incorrecto.

Para solucionar esto, utilizamos una **Transformación de Perspectiva (Inverse Perspective Mapping)**:

*   **Entrada (`SOURCE_POLYGON`)**: Definimos manualmente un área trapezoidal en el video que corresponde a una sección rectangular de la carretera.
*   **Destino (`TARGET_RECT`)**: Definimos las dimensiones de ese rectángulo en "unidades lógicas" o metros (ej. 25x200 unidades).
*   **Matriz de Homografía**: El sistema calcula una matriz que mapea los puntos del trapezoide a una vista superior ("bird's-eye view").

### 2. Cálculo de Velocidad
Una vez que podemos transformar cualquier punto del video a esta vista superior plana:

1.  **Tracking**: Seguiemos el centro inferior (pies) de cada vehículo frame a frame.
2.  **Transformación**: Convertimos las coordenadas $(x, y)$ del video a coordenadas $(x', y')$ en la vista superior.
3.  **Delta**: Calculamos la distancia recorrida entre frames en este nuevo espacio.
4.  **Fórmula**:
    ```
    Velocidad (km/h) = (Distancia / Tiempo) * 3.6
    ```
    Donde `tiempo = cuadros_transcurridos / fps`.

## �🤖 Tecnologías

*   **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: Detección de objetos de última generación.
*   **[Supervision](https://github.com/roboflow/supervision)**: Utilidades visuales y de procesamiento para visión por computadora.
*   **OpenCV**: Manipulación de imágenes y video.
*   **NumPy**: Cálculos matemáticos y transformación de matrices.

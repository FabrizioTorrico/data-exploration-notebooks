**Author:** Fabrizio Torrico — Diciembre 2025
# 🧪 Arquitecturas experimentales: GoogLeNet (Inception v1) y mas

Este repositorio es un registro de **investigación experimental**. Tomando como base la arquitectura **GoogleNet (Inception v1)** —ganadora del ILSVRC 2014—, el objetivo fue deconstruir sus principios, probar nuevas hipótesis sobre el procesamiento espacial y reconstruir un modelo más eficiente.

> **Objetivo del Proyecto:** Exploración empírica de arquitecturas CNN, gestión de flujo de información y eficiencia de parámetros.

**Restricción del Proyecto:**
Todos los modelos experimentales debían mantenerse en el rango de **~9M a 11M de parámetros** (igual o menor al original) para asegurar una eficiencia de entrenamiento superior.

---

## 💡 Motivación y Filosofía

Durante el desarrollo, surgieron preguntas fundamentales que guiaron las iteraciones:

1.  **¿Es el MaxPool un error?** Inspirado por la crítica de Geoffrey Hinton (*"The pooling operation... is a big mistake"*), se buscó alternativas a la destrucción de información espacial.
2.  **Espacialidad:** ¿Deberían procesarse los detalles finos (3x3) y las formas globales (5x5) juntos o por separado?
3.  **Densidad:** ¿Dónde necesita "cerebro" la red? ¿Al inicio para extraer o al final para abstraer?

---

## 📉 Baseline: GoogleNet Original (Inception v1)

Este fue el punto de partida y el estándar a vencer. GoogleNet marcó un antes y un después en la visión por computadora al alejarse de la simple "profundidad secuencial" (como VGG) para apostar por la "anchura" y la multiescalaridad. 

### 🏛️ La Arquitectura
La genialidad del módulo Inception radica en su filosofía *"Network In Network"*. Ejecuta convoluciones de $1\times1$, $3\times3$ y $5\times5$ en paralelo y concatena los resultados, permitiendo a la red decidir qué tamaño de filtro es mejor para cada característica.

### 🔍 Diagnóstico del Entrenamiento
Inicialmente, se observó un fuerte overfitting. Sin embargo, tras ajustar el *Learning Rate Scheduler* y permitir un entrenamiento extendido (40 épocas), el modelo original demostró su robustez:

* **Train Accuracy:** ~86.8%
* **Val Accuracy:** ~88.7%
* **Params:** ~10.9M

### 🤔 El Fenómeno "Val > Train"
Un hallazgo interesante durante este laboratorio fue observar consistentemente una precisión de validación **mayor** que la de entrenamiento.

Esto ocurre debido a la fuerte regularización aplicada durante el entrenamiento (**Data Augmentation** agresivo y **Dropout**). La red se entrena con imagenes dificiles de reconocer a proposito (imágenes rotadas, recortadas, con ruido), pero se evalúa con imágenes limpias. Esto por el momento es un indicador de generalización, aunque podrian tomarse medidas no tan agresivas. 

---

## 📊 Resumen de Resultados

El objetivo final fue vencer en precision y mejorar eficiencia:

| Modelo | Arquitectura Clave | Params | Train Acc | Val/Test Acc | Estado |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | GoogleNet Original | ~10.9M | 86.8% | **88.7%** | ✅ Benchmark |
| **V1 (Split)** | Parallel Streams + Fusion | ~10.5M | 81.6% | 59.2% | ❌ Convergencia Lenta |
| **V2 (Dual)** | Dual ResNet + "Fat Start" | ~8.6M | 57.9% | 47.6% | ❌ Fallo Estructural |
| **V3 (Hybrid)** | **Hybrid + Grid Reduction** | **~9.0M** | 83.6% | **88.3%** | ✅ **Eficiencia SOTA** |

> **Conclusión:** El modelo V3 logró empatar técnicamente al Baseline (diferencia < 0.4%) utilizando casi **2 Millones de parámetros menos (-18%)**.

---

## 🔄 Iteración 1: Hipotesis de Streaming paralelo

### 📝 La Teoría
La hipótesis fue que mezclar convoluciones de 3x3 y 5x5 en el mismo bloque crea "ruido". Se propuso **desacoplar las frecuencias espaciales**.

* **Arquitectura:** Los bloques Inception se dividieron en dos ramas internas separadas que se procesaban independientemente y solo se fusionaban mediante un cuello de botella al final.
* **Análisis del Fallo:** Aunque la teoría era sólida, la fusión tardía impedía que las capas siguientes aprovecharan correlaciones complejas entre lo local y lo global lo suficientemente rápido, ralentizando el aprendizaje.

---

## 📉 Iteración 2: Doble stream con mayor densidad inicial

### 📝 La Teoría
Aquí se probaron dos conceptos radicales:
1.  **Dual-Stream Puro:** Las ramas 3x3 y 5x5 nunca se tocan dentro del bloque ("autopistas" paralelas).
2.  **Fat Start / Skinny End:** Mucha más densidad al inicio y menos al final.

### 💀 El Fallo (Post-Mortem)
El modelo colapsó al 47% de precisión. Tras analizar el código y el comportamiento, se detectó el error conceptual:

* **Destrucción por 5x5:** Mantener filtros de 5x5 en capas profundas fue un error crítico. En las últimas capas, los mapas de características son de 7x7 pixeles. Aplicar un filtro de 5x5 ahí es casi una operación global, destruyendo cualquier noción de localización espacial.
* **Aislamiento:** La red tenía dos "cerebros" separados que no se comunicaban.


---

## 🚀 Iteración 3 y 4: Estilo hibrido

### 📝 La Síntesis
Aprendiendo de los fallos anteriores y convergiendo hacia conceptos de ResNet. 

1.  **Arquitectura Híbrida (The Funnel):**
    * **Inicio:** Procesamiento paralelo (3x3 + 5x5) para capturar contexto inicial.
    * **Cuerpo:** Fusión en una sola línea de 3x3. Se eliminan los 5x5 profundos para evitar la destrucción de información detectada en la V2.

2.  **Grid Reduction (Solucionando el problema del MaxPool):**
    * Teóricamente, el MaxPool destruye información valiosa.
    * Se implementó un **Bloque de Reducción**: Una rama hace Pooling y otra hace Convolución con stride (aprende a reducir). Se concatenan. Esto preserva el flujo de información.

3.  **Flujo Residual:**
    * Se implementaron conexiones de salto ($F(x) + x$) para permitir que la información original fluyera sin degradarse.

### 🏆 Conclusión Final
Este proyecto demostró que **la arquitectura es más importante que la fuerza bruta**.

1.  **Eficiencia:** Es posible igualar a un modelo SOTA eliminando el 20% de sus conexiones si se optimiza el flujo de datos.
2.  **Información:** El mayor enemigo de las redes profundas es la pérdida de información (cuellos de botella o MaxPool agresivo).
3.  **Evolución:** Comenzar con intuiciones propias y converger experimentalmente hacia soluciones nuevas, proceso de investigación empírica.

---

## 📦 Instalación y Uso

1. Clona el repositorio y entra en la carpeta del experimento:

```powershell
git clone https://github.com/FabrizioTorrico/data-exploration-notebooks.git
cd "data-exploration-notebooks/03 - GoogLeNet"
```

2. Instalar Pixi (recomendado) y dependencias:

- Este proyecto usa **Pixi** para gestionar entornos y dependencias a través de `pixi.toml`.
- Sigue la documentación oficial de Pixi para la instalación en tu plataforma.

Luego instala las dependencias definidas en `pixi.toml`:

```powershell
pixi install
```

3. Ejecutar entrenamiento / evaluación (ejemplo):

```powershell
pixi run python run.py
```

`run.py` es el runner principal; también hay utilidades en `src/` para ejecuciones más específicas.

---

## 🔧 Dependencias principales

- **Python:** 3.10+ (recomendado 3.12 para uniformidad con el repositorio).
- **PyTorch:** `torch` (+ `torchvision`). Soporta CUDA o CPU según tu hardware.
- **Librerías comunes:** `numpy`, `pillow`, `tqdm`, `matplotlib`.
- **Herramienta de gestión:** `pixi` (opcional pero recomendado para reproducibilidad).

Si usas `pixi`, la mayoría de dependencias se instalarán automáticamente desde `pixi.toml`.

---

## 📂 Estructura del directorio

- `main.py` — Entrada alternativa / scripts auxiliares.
- `run.py` — Runner principal (ejecutar con `pixi run python run.py`).
- `pixi.toml` / `pyproject.toml` — Definición del entorno y metadatos del proyecto.
- `checkpoints/` — Pesos y checkpoints guardados (ej.: checkpoints/googlenet_trained.pth).
- `data/` — Conjuntos de datos para entrenamiento/validación/prueba.
- `results/` — Logs y métricas de entrenamiento (CSV, curvas, métricas).
- `model_images/` — Imágenes y visualizaciones de la arquitectura del modelo.
- `src/` — Código fuente modular:
  - `config.py` — Configuración y parámetros de entrenamiento.
  - `data_loader.py` — Preparación y cargas de datos.
  - `trainer.py` — Bucle de entrenamiento y evaluación.
  - `models/` — Implementaciones de GoogleNet y variantes experimentales.

---

## Uso básico

- Preparar datos: colocar conjuntos en `data/` siguiendo la estructura esperada o usar los loaders incluidos.
- Ajustar `config.py` para seleccionar la variante del modelo, número de épocas, `batch_size`, etc.
- Ejecutar entrenamiento (ejemplo):

```powershell
pixi run python run.py
```

Para evaluación o inferencia rápida, revisa `run.py` y los logs dentro de `results/`.
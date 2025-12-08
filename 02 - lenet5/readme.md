# LeNet-5 PyTorch Experiment: Standard vs. Optimized

Este repositorio contiene una implementación moderna de la arquitectura **LeNet-5** utilizando **PyTorch** y gestionado con **Pixi**.

El objetivo del proyecto fue experimentar con la distribución de parámetros: mantener el mismo "presupuesto" de memoria (~61k parámetros) pero redistribuyendo la capacidad de cómputo para priorizar la extracción de características visuales sobre la densidad de las capas finales.

## 🧪 El Experimento

Se comparan dos arquitecturas entrenadas sobre el dataset MNIST:

1.  **LeNet-5 Base:** La arquitectura clásica (6 filtros $\to$ 16 filtros).
2.  **LeNet-5 Wide-Vis (Modificado):**
    * **Más visión:** Se duplicaron los filtros iniciales (12 filtros $\to$ 24 filtros) para capturar más texturas y formas primitivas.
    * **Cerebro más compacto:** Se redujeron drásticamente las neuronas en las capas totalmente conectadas (Linear) para compensar el peso y evitar el *overfitting*.

## 📊 Resultados y Comparativa

Ambos modelos tienen casi la misma cantidad de parámetros (~61,000), pero la versión modificada demostró ser ligeramente más rápida y precisa.

| Métrica | LeNet-5 Base | LeNet-5 Modificado (Wide) |
| :--- | :---: | :---: |
| **Parámetros Totales** | ~61,706 | ~61,086 |
| **Precisión (Accuracy)** | 98.82% - 99.07% | **99.10% - 99.26%** |
| **Tiempo Promedio / Época** | 4.77s | **4.54s** |
| **Filtros Conv** | [6, 16] | **[12, 24]** |
| **Capas Lineales** | [120, 84] | **[80, 60]** |


## 🛠️ Instalación y Uso

Este proyecto utiliza **Pixi** para la gestión de entornos y dependencias (Python 3.12 + PyTorch con soporte CUDA/CPU).

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/FabrizioTorrico/data-exploration-notebooks.git
    cd "data-exploration-notebooks/02 - lenet5"
    ```

2.  **Instalar dependencias:**
    ```bash
    pixi install
    ```

3.  **Entrenar los modelos:** (tener en cuenta que este necesario tener una GPU compatible)
    ```bash
    pixi run python main.py
    ```

## 📂 Estructura

* `main.py`: Script principal para entrenar y evaluar ambos modelos.
* `model.py`: Definición de las clases `LeNet5` (Base) y `LeNet5_Optimized`.
* `actions.py`: Funciones auxiliares para entrenamiento y evaluación.
* `pixi.toml`: Configuración del entorno y dependencias.

---
*Author: Fabrizio Torrico*
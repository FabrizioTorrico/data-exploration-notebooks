"""
Maneja la transformación de perspectiva de coordenadas de video a vista superior.
"""
import cv2
import numpy as np


class CoordinateTransformer:
    """
    Transforma puntos desde la perspectiva del video a una vista superior (top-down)
    para cálculos precisos de distancia y velocidad.
    """

    def __init__(self, source_polygon: np.ndarray, target_rect: np.ndarray):
        """
        Inicializa la matriz de transformación perspectiva.
        
        Args:
            source_polygon: Puntos en la perspectiva original
            target_rect: Puntos en la perspectiva objetivo
        """
        source = source_polygon.astype(np.float32)
        target = target_rect.astype(np.float32)
        self.transformation_matrix = cv2.getPerspectiveTransform(source, target)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Transforma un conjunto de puntos a la vista superior.
        
        Args:
            points: Array de puntos con forma (N, 2)
            
        Returns:
            Array de puntos transformados
        """
        if points.size == 0:
            return points

        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.transformation_matrix)
        return transformed.reshape(-1, 2)
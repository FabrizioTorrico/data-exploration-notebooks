import cv2
import numpy as np
from collections import defaultdict, deque
import supervision as sv
import config


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforma puntos usando la matriz de perspectiva. esto se hace para calular
        la velocidad en la vista del poligono
        """
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


class SpeedCalculator:
    def __init__(self, fps: int):
        self.fps = fps
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.transformer = ViewTransformer(config.SOURCE_POLYGON, config.TARGET_RECT)

    def update(self, detections: sv.Detections) -> dict[int, float]:
        """
        Actualiza posiciones y retorna un diccionario {tracker_id: velocidad_kmh}
        """

        # Obtener coordenadas de los pies de los vehículos (centro-abajo)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        # Transformar a l poligono de procesamiento
        transformed_points = self.transformer.transform_points(points).astype(int)

        speeds = {}

        # Guardar historial y calcular
        for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
            self.coordinates[tracker_id].append(y)

            # Solo calculamos si tenemos suficientes datos (al menos medio segundo)
            if len(self.coordinates[tracker_id]) < self.fps / 2:
                speeds[tracker_id] = 0.0
            else:
                coordinate_start = self.coordinates[tracker_id][-1]
                coordinate_end = self.coordinates[tracker_id][0]

                distance = abs(coordinate_start - coordinate_end)
                time = len(self.coordinates[tracker_id]) / self.fps

                speed_kmh = (distance / time) * 3.6
                speeds[tracker_id] = speed_kmh

        return speeds

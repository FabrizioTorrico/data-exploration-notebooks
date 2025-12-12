from ultralytics import YOLO
import supervision as sv
import numpy as np
import config


class VehicleDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.CONFIDENCE_THRESHOLD
        )
        self.polygon_zone = sv.PolygonZone(polygon=config.SOURCE_POLYGON)

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """
        Realiza inferencia, filtra por zona y aplica tracking.
        """
        # 1. Inferencia
        result = self.model.predict(
            frame,
            classes=config.VEHICLE_CLASSES,
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        # 2. Filtrar por zona (Solo se procesa lo que está dentro del polígono)
        is_in_zone = self.polygon_zone.trigger(detections)
        detections = detections[is_in_zone]

        # 3. Tracking (Asignar IDs)
        return self.tracker.update_with_detections(detections=detections)

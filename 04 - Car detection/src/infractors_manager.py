import cv2
import os
import numpy as np
import supervision as sv
import config


class InfractorsManager:
    def __init__(self):
        os.makedirs(config.CAPTURES_DIR, exist_ok=True)
        self.captured_ids = set()

    def process_violation(
        self, frame: np.ndarray, speeds: dict[int, float], detections: sv.Detections
    ):
        """
        Verifica el limite de velocidad, si el vehículo no ha sido capturado, guarda un recorte.
        Si el vehículo ya fue capturado, no hace nada (para evitar duplicados).
        """

        for tracker_id, speed in speeds.items():
            if speed > config.SPEED_LIMIT_KMH and tracker_id not in self.captured_ids:
                try:
                    # Buscar el índice del tracker_id en las detecciones actuales
                    idx = np.where(detections.tracker_id == tracker_id)[0][0]
                    x1, y1, x2, y2 = detections.xyxy[idx]

                    # Validar coordenadas (fuera de límites)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # Guardar recorte con información del infractor
                        filename = f"crop_id{tracker_id}_{int(speed)}kmh.jpg"
                        path = os.path.join(config.CAPTURES_DIR, filename)

                        if cv2.imwrite(path, crop):
                            print(f"📸 Infracción capturada: {filename}")
                            self.captured_ids.add(tracker_id)

                except IndexError:
                    pass  # El tracker se perdió justo en este frame

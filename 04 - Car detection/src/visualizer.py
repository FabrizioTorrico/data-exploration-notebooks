import supervision as sv
import cv2
import numpy as np
import config


class Visualizer:
    def __init__(self, resolution_wh):
        """
        Se inicializan los anotadores con parámetros adaptados a la resolución del video.
        ej: cajas, textos y colas de seguimiento.
        """
        self.thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=resolution_wh
        )
        self.text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

        self.box_annotator = sv.BoxAnnotator(thickness=self.thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.text_scale,
            text_thickness=self.thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.thickness, position=sv.Position.BOTTOM_CENTER
        )

    def draw(
        self, frame: np.ndarray, detections: sv.Detections, speeds: dict
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # Generar etiquetas
        labels = []

        for tracker_id in detections.tracker_id:
            speed = speeds.get(tracker_id, 0)
            # etiqueta con ID y velocidad
            labels.append(f"#{tracker_id} - {int(speed)}km/h")

        # Dibujar trazas (colas)
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Dibujar cajas (con color dinámico)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        # Dibujar textos
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        return annotated_frame

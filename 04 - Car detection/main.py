import cv2
import supervision as sv
import config

# Imports de nuestros módulos
from src.vehicle_detector import VehicleDetector
from src.speed_system import SpeedCalculator
from src.infractors_manager import InfractorsManager
from src.visualizer import Visualizer


def main():
    # 1. Setup inicial
    if not os.path.exists(config.SOURCE_VIDEO_PATH):
        print("❌ Error: No se encuentra el video fuente.")
        return

    video_info = sv.VideoInfo.from_video_path(video_path=config.SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(
        source_path=config.SOURCE_VIDEO_PATH
    )

    # 2. Instanciar módulos
    detector = VehicleDetector()
    speed_calculator = SpeedCalculator(fps=video_info.fps)
    infractor_manager = InfractorsManager()
    visualizer = Visualizer(resolution_wh=video_info.resolution_wh)

    print(f"🚀 Iniciando procesamiento de: {config.SOURCE_VIDEO_PATH}")

    with sv.VideoSink(config.TARGET_VIDEO_PATH, video_info) as sink:
        for frame in frame_generator:
            # A. Detectar y Trackear
            detections = detector.detect_and_track(frame)

            # B. Calcular Velocidad
            speeds = speed_calculator.update(detections)

            # C. Verificar Infracciones
            infractor_manager.process_violation(frame, speeds, detections)

            # D. Visualizar y Guardar Video
            annotated_frame = visualizer.draw(frame, detections, speeds)

            sink.write_frame(annotated_frame)
            cv2.imshow("Monitor de Tráfico", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("✅ Procesamiento finalizado.")


if __name__ == "__main__":
    import os

    main()

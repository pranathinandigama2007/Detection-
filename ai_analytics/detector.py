# ai_analytics/detector.py
import math
import cv2
from ultralytics import YOLO
from logging_module.logger import get_logger

logger = get_logger("ai_detector")

class TargetDetector:
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        # Load lightweight nano model (downloads automatically on first run)
        self.model = YOLO("yolov8n.pt")

    def process_frame(self, frame_bgr, frame_metadata: dict, telemetry: dict) -> list:
        """
        Runs real YOLO model on the live frame and calculates geo-coordinates.
        """
        results = self.model(frame_bgr, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                target_geo = self._calculate_target_coordinates(
                    drone_lat=telemetry.get("lat", 0.0),
                    drone_lon=telemetry.get("lon", 0.0),
                    alt=telemetry.get("alt", 50.0),
                    heading=telemetry.get("heading", 0.0)
                )

                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 2),
                    "bbox": xyxy,
                    "latitude": target_geo["lat"],
                    "longitude": target_geo["lon"]
                })

        return detections

    def _calculate_target_coordinates(self, drone_lat: float, drone_lon: float, alt: float, heading: float) -> dict:
        earth_radius = 6378137.0
        offset_meters = alt * math.tan(math.radians(15))
        bearing = math.radians(heading)
        
        d_lat = (offset_meters * math.cos(bearing)) / earth_radius
        d_lon = (offset_meters * math.sin(bearing)) / (earth_radius * math.cos(math.radians(drone_lat)))

        return {
            "lat": drone_lat + math.degrees(d_lat),
            "lon": drone_lon + math.degrees(d_lon)
        }
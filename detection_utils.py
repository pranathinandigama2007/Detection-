"""
DECS - Dynamic Tactical Detection Engine
========================================
- Single-Pass Orientation Invariant Inference (0°, 90°, 180°, 270°)
- Multi-Angle Inverse Coordinate Mapper
- High-Visibility Tactical OSD (Thick Boxes, Enlarged Text Badges)
"""

import cv2
import numpy as np
import math
import os
import socket
import urllib.request
import json
import torch
from ultralytics import YOLO

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def get_live_coordinates():
    try:
        req = urllib.request.Request(
            "https://ipapi.co/json/",
            headers={"User-Agent": "DECS-Tactical-C2/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("latitude", 17.0253)), float(data.get("longitude", 81.7779))
    except Exception:
        return 17.0253, 81.7779

HOST_IP = get_local_ip()
DEFAULT_LAT, DEFAULT_LON = get_live_coordinates()
DEFAULT_ALTITUDE_METERS = 50.0
DEFAULT_HEADING_DEG = 0.0

ANIMAL_SET = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
    "bear", "zebra", "giraffe", "peacock", "snake", "lion", "cheetah"
}

HUMAN_SET = {"person", "human", "man", "woman"}

CATEGORY_COLORS = {
    "HUMAN": (0, 255, 128),       # Neon Green (BGR)
    "ANIMAL": (255, 215, 0),      # Cyan (BGR)
    "OBJECT": (247, 85, 168),     # Purple (BGR)
    "CUSTOM": (0, 140, 255)       # Tactical Orange (BGR)
}

class HighPerformanceDetector:
    def __init__(self, model_path=None, model_filename=None, conf_threshold=0.25, **kwargs):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        actual_model = model_path if model_path else model_filename
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        preferred_weights = [
            actual_model,
            os.path.join(base_dir, "yolov8s-worldv2.pt"),
            os.path.join(base_dir, "yolo12n.pt"),
            os.path.join(base_dir, "drone_ingestion", "yolov8n.pt"),
            os.path.join(base_dir, "yolov8n.pt"),
            "yolov8n.pt"
        ]

        selected_model = None
        for weight_path in preferred_weights:
            if weight_path and os.path.exists(weight_path):
                selected_model = weight_path
                break

        if not selected_model:
            selected_model = "yolov8n.pt"

        print(f"[AI ENGINE] Loading Omni-Angle Model: {os.path.basename(selected_model)} on {self.device.upper()}...")
        self.model = YOLO(selected_model)

        if self.device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 4))

    def detect(self, frame, input_size=320, active_category="ALL", orientation_deg=0):
        h, w = frame.shape[:2]

        # Normalize rotation angle in memory before YOLO forward pass
        rot = orientation_deg % 360
        if rot == 90:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            proc_frame = frame

        # High-speed YOLO inference
        results = self.model.predict(
            source=proc_frame,
            imgsz=input_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            half=False
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(confs)):
            cls_id = classes[i]
            raw_label = self.model.names.get(cls_id, f"object_{cls_id}").lower()
            bx1, by1, bx2, by2 = coords[i]

            # Invert rotated coordinates back to original frame space
            if rot == 90:
                x1 = by1
                y1 = h - bx2
                x2 = by2
                y2 = h - bx1
            elif rot == 180:
                x1 = w - bx2
                y1 = h - by2
                x2 = w - bx1
                y2 = h - by1
            elif rot == 270:
                x1 = w - by2
                y1 = bx1
                x2 = w - by1
                y2 = bx2
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2

            x1 = max(0, min(w, min(x1, x2)))
            y1 = max(0, min(h, min(y1, y2)))
            x2 = max(0, min(w, max(x1, x2)))
            y2 = max(0, min(h, max(y1, y2)))

            is_human = any(term in raw_label for term in HUMAN_SET)
            is_animal = any(term in raw_label for term in ANIMAL_SET)

            if is_human:
                category = "HUMAN"
                display_label = "HUMAN"
            elif is_animal:
                category = "ANIMAL"
                sub_label = raw_label.replace("animal", "").strip().upper()
                display_label = f"ANIMAL: {sub_label}"
            else:
                category = "OBJECT"
                sub_label = raw_label.replace("object", "").strip().upper()
                display_label = f"OBJECT: {sub_label}"

            if active_category != "ALL" and category != active_category:
                continue

            detections.append({
                "class_name": display_label,
                "category": category,
                "confidence": round(float(confs[i]), 2),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "engine": "DECS_FAST_OMNI"
            })

        return detections


def calculate_target_coordinates(pixel_x, pixel_y, frame_width, frame_height, 
                                 drone_lat=None, drone_lon=None, 
                                 altitude_m=DEFAULT_ALTITUDE_METERS, heading_deg=DEFAULT_HEADING_DEG, hfov_deg=84.0):
    if drone_lat is None: drone_lat = DEFAULT_LAT
    if drone_lon is None: drone_lon = DEFAULT_LON

    norm_x = (pixel_x - (frame_width / 2.0)) / frame_width
    norm_y = ((frame_height / 2.0) - pixel_y) / frame_height

    vfov_deg = hfov_deg * (frame_height / frame_width)
    ground_width = 2.0 * altitude_m * math.tan(math.radians(hfov_deg / 2.0))
    ground_height = 2.0 * altitude_m * math.tan(math.radians(vfov_deg / 2.0))

    offset_east = norm_x * ground_width
    offset_north = norm_y * ground_height

    heading_rad = math.radians(heading_deg)
    rot_east = offset_east * math.cos(heading_rad) + offset_north * math.sin(heading_rad)
    rot_north = -offset_east * math.sin(heading_rad) + offset_north * math.cos(heading_rad)

    earth_radius = 6378137.0
    d_lat = (rot_north / earth_radius) * (180.0 / math.pi)
    d_lon = (rot_east / (earth_radius * math.cos(math.radians(drone_lat)))) * (180.0 / math.pi)

    return round(drone_lat + d_lat, 6), round(drone_lon + d_lon, 6)


def get_category_color(category: str):
    return CATEGORY_COLORS.get(category, (247, 85, 168))


def draw_and_package(frame, final_targets, drone_telemetry=None):
    """Draws prominent tactical bounding boxes and high-contrast labels without line clipping."""
    telemetry_payload = []
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_DUPLEX

    drone_lat = drone_telemetry.get("lat", DEFAULT_LAT) if drone_telemetry else DEFAULT_LAT
    drone_lon = drone_telemetry.get("lon", DEFAULT_LON) if drone_telemetry else DEFAULT_LON
    alt_m = drone_telemetry.get("altitude", DEFAULT_ALTITUDE_METERS) if drone_telemetry else DEFAULT_ALTITUDE_METERS
    heading = drone_telemetry.get("heading", DEFAULT_HEADING_DEG) if drone_telemetry else DEFAULT_HEADING_DEG

    for item in final_targets:
        x1, y1, x2, y2 = item['x1'], item['y1'], item['x2'], item['y2']
        cls_name = item['class_name']
        category = item.get('category', 'OBJECT')
        conf = item['confidence']
        color = get_category_color(category)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        target_lat, target_lon = calculate_target_coordinates(
            center_x, center_y, w, h, drone_lat, drone_lon, alt_m, heading
        )

        # Thick 4px Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

        # Tactical Corner Accents
        corner_len = max(24, min(44, (x2 - x1) // 3, (y2 - y1) // 3))
        b_th = 5
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), b_th, cv2.LINE_AA)

        # High-Contrast Large Text Badge
        label_text = f" {cls_name} {int(conf * 100)}% "
        font_scale = 0.85
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        badge_y1 = max(0, y1 - th - 12)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 10)

        cv2.rectangle(frame, (x1 - 2, badge_y1 - 2), (badge_x2 + 2, badge_y2 + 2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, y1 - 6), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        telemetry_payload.append({
            "class_name": cls_name,
            "category": category,
            "confidence": round(conf, 2),
            "latitude": target_lat,
            "longitude": target_lon,
            "engine": item.get('engine', 'DECS_FAST_OMNI')
        })

    return telemetry_payload

ONNXDetector = HighPerformanceDetector

# ---------------------------------------------------------------------------
# Runtime configuration + stream endpoints (consumed by drone_ingestion/rtmp_reader.py)
# ---------------------------------------------------------------------------
ADMIN_CONFIG = {
    "engine_mode": "TACTICAL_V8_ULTRA",
    "min_confidence": 0.28,
    "clahe_enabled": False,
}

WS_URL = "ws://localhost:8000/ws/telemetry"
RTMP_URL = f"rtmp://{HOST_IP}:1935/live/drone1"


def apply_clahe(frame):
    """Applies local contrast enhancement for dark or high-exposure streams."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def setup_display_window(window_name):
    """Configures a resizable OpenCV GUI window that retains aspect ratio."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)


def draw_osd(frame, mode, min_conf, fps=0.0):
    """Draws the upper status HUD bar (mode, confidence threshold, live FPS)."""
    h, w, _ = frame.shape
    banner_height = int(max(40, h * 0.05))
    text = f"MODE: {mode} | CONF: {min_conf:.2f} | FPS: {fps:.1f}"

    cv2.rectangle(frame, (0, 0), (w, banner_height), (15, 23, 42), -1)
    cv2.putText(frame, text, (15, int(banner_height * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, banner_height / 80.0), (56, 189, 248), 2)
"""
DECS - Dynamic Tactical Detection Engine (Military-Grade Accuracy + Ultra-Low Latency)
========================================================================================
- Non-Blocking Threaded Frame Acquisition (Zero-Buffer Lag & Glitch Prevention)
- Asynchronous SAHI-Style Tiled Inference Pass for High Altitude Small Object Recall
- FP16 Half-Precision CUDA Inference Acceleration
- Single-Pass Orientation Invariant Inference (0deg, 90deg, 180deg, 270deg)
- Zero-Dependency ORB + HSV Target Lock Matcher
- High-Visibility Tactical OSD & GPS Coordinate Inverse Mapper
"""

import cv2
import numpy as np
import math
import os
import socket
import time
import urllib.request
import json
import torch
import threading
from queue import Queue
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

# ---------------------------------------------------------------------------
# Runtime configuration + stream endpoints
# ---------------------------------------------------------------------------
ADMIN_CONFIG = {
    "engine_mode": "TACTICAL_V9_ASYNC_ULTRA",
    "min_confidence": 0.22,
    "clahe_enabled": False,
    "tile_every_n_frames": 4,
    "category_conf_multiplier": {
        "HUMAN": 0.75,
        "ANIMAL": 0.90,
        "OBJECT": 1.00,
    },
    "merge_iou_threshold": 0.45,
}

WS_URL = "ws://localhost:8000/ws/telemetry"
RTMP_URL = f"rtmp://{HOST_IP}:1935/live/drone1"


class UltraLowLatencyStreamReader:
    """Continuously drains OpenCV stream buffer to ensure 0-latency playback."""
    def __init__(self, src=0, api_preference=cv2.CAP_FFMPEG):
        self.cap = cv2.VideoCapture(src, api_preference)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Exception:
                    pass
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return False, None
        return True, self.q.get()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


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


def _classify(raw_label: str):
    """Maps a raw model class name to DECS' HUMAN/ANIMAL/OBJECT taxonomy."""
    is_human = any(term in raw_label for term in HUMAN_SET)
    is_animal = any(term in raw_label for term in ANIMAL_SET)
    if is_human:
        return "HUMAN", "HUMAN"
    if is_animal:
        sub_label = raw_label.replace("animal", "").strip().upper()
        return "ANIMAL", f"ANIMAL: {sub_label}"
    sub_label = raw_label.replace("object", "").strip().upper()
    return "OBJECT", f"OBJECT: {sub_label}"


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _merge_detections(detections, iou_threshold):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in detections:
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        duplicate = False
        for k in kept:
            kbox = (k["x1"], k["y1"], k["x2"], k["y2"])
            if _iou(box, kbox) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _generate_tiles(w, h, tile_frac=0.55, overlap=0.20):
    """Yields overlapping tile windows for high-altitude small object detection."""
    tile_w = int(w * tile_frac)
    tile_h = int(h * tile_frac)
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    xs = list(range(0, max(1, w - tile_w) + 1, step_x))
    ys = list(range(0, max(1, h - tile_h) + 1, step_y))
    if not xs or xs[-1] + tile_w < w:
        xs.append(max(0, w - tile_w))
    if not ys or ys[-1] + tile_h < h:
        ys.append(max(0, h - tile_h))

    for ty in ys:
        for tx in xs:
            yield (tx, ty, min(w, tx + tile_w), min(h, ty + tile_h))


class HighPerformanceDetector:
    """
    Dual-model ensemble detector with Async Non-Blocking Tiling:
      - Real-time fast pass runs synchronously every frame.
      - Deep tiled pass runs in a background thread to prevent latency spikes.
    """
    def __init__(self, model_path=None, model_filename=None, conf_threshold=0.25, **kwargs):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = self.device == "cuda"
        self._frame_counter = 0

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        actual_model = model_path if model_path else model_filename

        def resolve(*candidates):
            for c in candidates:
                if c and os.path.exists(c):
                    return c
            return None

        fast_candidates = [
            actual_model,
            os.path.join(base_dir, "yolo26n.pt"),
            os.path.join(base_dir, "yolo12n.pt"),
            os.path.join(base_dir, "drone_ingestion", "yolov8n.pt"),
            os.path.join(base_dir, "yolov8n.pt"),
        ]
        fast_path = resolve(*fast_candidates) or "yolov8n.pt"

        deep_candidates = [
            os.path.join(base_dir, "yolov8s-worldv2.pt"),
            os.path.join(base_dir, "yolo26n.pt"),
            os.path.join(base_dir, "yolo12n.pt"),
        ]
        deep_path = resolve(*deep_candidates) or fast_path

        print(f"[AI ENGINE] Loading FAST model: {os.path.basename(fast_path)} on {self.device.upper()}...")
        self.fast_model = YOLO(fast_path)

        if deep_path == fast_path:
            self.deep_model = self.fast_model
            print("[AI ENGINE] Deep tiled pass reuses fast model.")
        else:
            print(f"[AI ENGINE] Loading DEEP model: {os.path.basename(deep_path)} on {self.device.upper()}...")
            self.deep_model = YOLO(deep_path)

        self.model = self.fast_model

        # Warmup GPU
        if self.device == "cuda":
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.fast_model.predict(dummy, device=self.device, half=self.use_fp16, verbose=False)

        # Thread state for non-blocking tiled pass
        self.async_lock = threading.Lock()
        self.async_in_progress = False
        self.async_tiled_results = []

        if self.device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 4))

    def _run_model(self, model, image, input_size, conf):
        results = model.predict(
            source=image,
            imgsz=input_size,
            conf=conf,
            device=self.device,
            verbose=False,
            half=self.use_fp16,
        )
        if not results or results[0].boxes is None:
            return [], {}
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = model.names
        return list(zip(coords, confs, classes)), names

    def _boxes_to_targets(self, raw, names, offset_x=0, offset_y=0, active_category="ALL", engine_tag="DECS_FAST_OMNI"):
        targets = []
        for coords, conf, cls_id in raw:
            raw_label = names.get(int(cls_id), f"object_{cls_id}").lower()
            category, display_label = _classify(raw_label)

            mult = ADMIN_CONFIG["category_conf_multiplier"].get(category, 1.0)
            effective_conf = self.conf_threshold * mult
            if float(conf) < effective_conf:
                continue

            if active_category != "ALL" and category != active_category:
                continue

            x1, y1, x2, y2 = coords
            targets.append({
                "class_name": display_label,
                "category": category,
                "confidence": round(float(conf), 2),
                "x1": int(x1) + offset_x,
                "y1": int(y1) + offset_y,
                "x2": int(x2) + offset_x,
                "y2": int(y2) + offset_y,
                "engine": engine_tag,
            })
        return targets

    def _async_tiled_pass(self, proc_frame, input_size, pw, ph, unrotate_box_fn, active_category):
        try:
            deep_size = max(input_size, 384)
            async_targets = []
            for (tx1, ty1, tx2, ty2) in _generate_tiles(pw, ph):
                tile_img = proc_frame[ty1:ty2, tx1:tx2]
                if tile_img.size == 0:
                    continue
                t_raw, t_names = self._run_model(self.deep_model, tile_img, deep_size, self.conf_threshold)
                t_global = []
                for coords, cf, cl in t_raw:
                    bx1, by1, bx2, by2 = coords
                    g = np.array([bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1])
                    t_global.append((unrotate_box_fn(g), cf, cl))
                tile_targets = self._boxes_to_targets(
                    t_global, t_names, active_category=active_category, engine_tag="DECS_DEEP_TILED"
                )
                async_targets.extend(tile_targets)

            with self.async_lock:
                self.async_tiled_results = async_targets
        finally:
            with self.async_lock:
                self.async_in_progress = False

    def detect(self, frame, input_size=320, active_category="ALL", orientation_deg=0):
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # Normalize rotation
        rot = orientation_deg % 360
        if rot == 90:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            proc_frame = frame

        ph, pw = proc_frame.shape[:2]

        def unrotate_box(coords):
            bx1, by1, bx2, by2 = coords
            if rot == 90:
                x1, y1, x2, y2 = by1, h - bx2, by2, h - bx1
            elif rot == 180:
                x1, y1, x2, y2 = w - bx2, h - by2, w - bx1, h - by1
            elif rot == 270:
                x1, y1, x2, y2 = w - by2, bx1, w - by1, bx2
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2
            x1 = max(0, min(w, min(x1, x2)))
            y1 = max(0, min(h, min(y1, y2)))
            x2 = max(0, min(w, max(x1, x2)))
            y2 = max(0, min(h, max(y1, y2)))
            return np.array([x1, y1, x2, y2])

        # Synchronous Fast Pass (Real-time detection)
        raw, names = self._run_model(self.fast_model, proc_frame, input_size, self.conf_threshold)
        raw_unrot = [(unrotate_box(c), cf, cl) for c, cf, cl in raw]
        all_targets = self._boxes_to_targets(raw_unrot, names, active_category=active_category, engine_tag="DECS_FAST_OMNI")

        # Asynchronous Tiled Deep Pass (High accuracy without freezing the stream)
        self._frame_counter += 1
        n = max(1, ADMIN_CONFIG["tile_every_n_frames"])

        with self.async_lock:
            if not self.async_in_progress and (self._frame_counter % n == 0):
                self.async_in_progress = True
                threading.Thread(
                    target=self._async_tiled_pass,
                    args=(proc_frame.copy(), input_size, pw, ph, unrotate_box, active_category),
                    daemon=True
                ).start()
            
            all_targets.extend(self.async_tiled_results)

        return _merge_detections(all_targets, ADMIN_CONFIG["merge_iou_threshold"])


class TargetMatcher:
    """Zero-dependency ORB + HSV Target Lock Matcher."""
    MATCH_THRESHOLD = 0.18

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_descriptors = None
        self.ref_hist = None
        self.ref_category = None
        self.active = False
        self.label = None

    def set_reference(self, ref_image, category_hint=None, label="CUSTOM TARGET"):
        if ref_image is None or ref_image.size == 0:
            return False
        gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        _, des = self.orb.detectAndCompute(gray, None)
        hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        self.ref_descriptors = des
        self.ref_hist = hist
        self.ref_category = category_hint
        self.label = label
        self.active = True
        return True

    def clear(self):
        self.active = False
        self.ref_descriptors = None
        self.ref_hist = None
        self.ref_category = None
        self.label = None

    def _score(self, crop):
        if crop is None or crop.size == 0:
            return 0.0

        orb_score = 0.0
        if self.ref_descriptors is not None and len(self.ref_descriptors) > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, des = self.orb.detectAndCompute(gray, None)
            if des is not None and len(des) > 0:
                matches = self.bf.match(self.ref_descriptors, des)
                if matches:
                    good = [m for m in matches if m.distance < 60]
                    orb_score = len(good) / max(1, len(self.ref_descriptors))
                    orb_score = min(1.0, orb_score)

        hist_score = 0.0
        if self.ref_hist is not None:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            hist_score = max(0.0, cv2.compareHist(self.ref_hist, hist, cv2.HISTCMP_CORREL))

        return 0.6 * orb_score + 0.4 * hist_score

    def find_best_match(self, frame, candidates):
        if not self.active:
            return candidates

        best = None
        best_score = 0.0
        h, w = frame.shape[:2]
        for cand in candidates:
            if self.ref_category and cand["category"] != self.ref_category:
                continue
            x1 = max(0, cand["x1"]); y1 = max(0, cand["y1"])
            x2 = min(w, cand["x2"]); y2 = min(h, cand["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            s = self._score(crop)
            if s > best_score:
                best_score = s
                best = cand

        if best is not None and best_score >= self.MATCH_THRESHOLD:
            locked = dict(best)
            locked["class_name"] = f"TARGET LOCK: {self.label} ({int(best_score * 100)}%)"
            locked["category"] = "CUSTOM"
            locked["engine"] = "DECS_TARGET_LOCK"
            return [locked]
        return []


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
    """Draws tactical bounding boxes, corner accents, and contrast text badges."""
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

        corner_len = max(24, min(44, (x2 - x1) // 3, (y2 - y1) // 3))
        b_th = 5
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), b_th, cv2.LINE_AA)

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
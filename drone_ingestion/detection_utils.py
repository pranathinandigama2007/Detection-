"""
DECS - Dynamic Tactical Detection Engine
========================================
- Single-Pass Orientation Invariant Inference (0°, 90°, 180°, 270°)
- Class-Agnostic PyTorch NMS De-duplication
- Sleek Tactical OSD Rendering (Minimal Clutter, Compact Badges)
- High-Precision Multi-Feature Target Lock Engine
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
import torchvision
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
    "CUSTOM": (0, 140, 255),      # Tactical Orange (BGR)
    "TARGET_LOCK": (0, 215, 255)  # Bright Cyan Target Lock
}

class HighPerformanceDetector:
    def __init__(self, model_path=None, model_filename=None, conf_threshold=0.35, **kwargs):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        actual_model = model_path if model_path else model_filename
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        preferred_weights = [
            actual_model,
            os.path.join(base_dir, "yolov8n-visdrone.pt"),
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

        # Precision Target Lock System -- keypoint + homography based (see
        # set_target_lock / _locate_target below). This does NOT depend on
        # YOLO's fixed COCO class list, so it works for ANY object, not just
        # the 80 classes YOLO already knows.
        self.target_locked = False
        self.target_label = "LOCKED TARGET"
        self.ref_gray = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_corners = None  # the 4 corners of the reference image, for homography projection

        # Full-frame ORB needs more features than a small reference crop to
        # get good keypoint coverage across a busy scene.
        self.orb_ref = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
        self.orb_frame = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Minimum number of geometrically-verified (RANSAC inlier) matches
        # required before a lock is accepted as real. This is the actual
        # correctness gate -- it requires genuine structural/textural
        # correspondence with the reference photo, not just similar colors,
        # so it will show nothing rather than lock onto the wrong object.
        self.MIN_GOOD_MATCHES = 10
        self.MIN_INLIERS = 8

        # Brief tracking-continuity memory so a single missed frame doesn't
        # make the box flicker off and on.
        self.last_target_box = None
        self.last_target_conf = 0.0
        self.last_target_time = 0.0
        self.target_memory_ttl_seconds = 0.6

    def set_target_lock(self, reference_img_bytes: bytes, filename: str = "Target"):
        try:
            nparr = np.frombuffer(reference_img_bytes, np.uint8)
            ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if ref_img is None:
                return False, "Failed to decode reference image"

            # NOTE: deliberately NOT running YOLO on the reference image
            # anymore. YOLO only knows 80 COCO classes (no "tire", no most
            # custom objects) -- running it here to "auto-crop" the subject
            # was picking arbitrary/wrong boxes and is what caused target
            # lock to fixate on unrelated people/laptops in testing. The
            # user-supplied photo is used directly instead.
            rh, rw = ref_img.shape[:2]
            if rh < 20 or rw < 20:
                return False, "Reference image is too small."

            self.ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            # Mild contrast normalization so lighting differences between the
            # uploaded photo and the live feed matter less for matching.
            self.ref_gray = cv2.equalizeHist(self.ref_gray)

            self.ref_keypoints, self.ref_descriptors = self.orb_ref.detectAndCompute(self.ref_gray, None)

            if self.ref_descriptors is None or len(self.ref_descriptors) < self.MIN_GOOD_MATCHES:
                return False, (
                    "This image doesn't have enough distinct visual detail to lock onto "
                    "(too plain/blurry/uniform). Try a clearer, closer photo with visible "
                    "texture or markings."
                )

            self.ref_corners = np.float32([
                [0, 0], [rw, 0], [rw, rh], [0, rh]
            ]).reshape(-1, 1, 2)

            clean_filename = os.path.splitext(os.path.basename(filename))[0].upper()
            self.target_label = f"TARGET: {clean_filename}"
            self.target_locked = True
            self.last_target_box = None
            self.last_target_conf = 0.0
            self.last_target_time = 0.0
            print(f"[TARGET LOCK INITIALIZED] {self.target_label} -- {len(self.ref_keypoints)} reference keypoints")
            return True, self.target_label
        except Exception as e:
            print(f"[TARGET LOCK ERROR] {e}")
            return False, str(e)

    def clear_target_lock(self):
        self.target_locked = False
        self.ref_gray = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_corners = None
        self.last_target_box = None
        self.last_target_conf = 0.0
        self.last_target_time = 0.0
        self.target_label = "LOCKED TARGET"
        print("[TARGET LOCK] Cleared. Resuming multi-category tracking.")

    def _locate_target(self, frame_gray):
        """
        Finds the reference target's location in the current frame using
        ORB keypoint matching + RANSAC homography -- classical feature-based
        object localization. Returns (x1, y1, x2, y2, confidence) or None if
        no geometrically consistent match is found this frame. Deliberately
        returns None (rather than a best-effort guess) when the evidence is
        weak: a missing box is far better than a wrong one.
        """
        if self.ref_descriptors is None:
            return None

        frame_kp, frame_des = self.orb_frame.detectAndCompute(frame_gray, None)
        if frame_des is None or len(frame_des) < 2:
            return None

        # k=2 nearest-neighbor matching + Lowe's ratio test -- this is the
        # standard, much more reliable way to filter ambiguous matches than
        # a fixed distance cutoff (rejects matches where the 2nd-best
        # candidate is nearly as good as the best, i.e. not distinctive).
        try:
            knn_matches = self.bf_matcher.knnMatch(self.ref_descriptors, frame_des, k=2)
        except cv2.error:
            return None

        good_matches = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.MIN_GOOD_MATCHES:
            return None

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return None

        inlier_count = int(mask.sum())
        if inlier_count < self.MIN_INLIERS:
            return None

        projected = cv2.perspectiveTransform(self.ref_corners, homography)
        xs = projected[:, 0, 0]
        ys = projected[:, 0, 1]
        x1, y1 = float(np.min(xs)), float(np.min(ys))
        x2, y2 = float(np.max(xs)), float(np.max(ys))

        fh, fw = frame_gray.shape[:2]
        # Reject degenerate/absurd projections (homography can occasionally
        # fold a box inside-out or blow it up huge on weak geometry).
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 10 or box_h < 10 or box_w > fw * 1.5 or box_h > fh * 1.5:
            return None

        x1 = max(0, min(fw, x1)); x2 = max(0, min(fw, x2))
        y1 = max(0, min(fh, y1)); y2 = max(0, min(fh, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        confidence = min(1.0, inlier_count / 20.0)
        return int(x1), int(y1), int(x2), int(y2), confidence

    def _compare_crop(self, crop, candidate_box=None):
        # Retained for backward compatibility with anything else that might
        # reference it; no longer used by the target-lock path (see
        # _locate_target above), which uses homography-based geometric
        # verification instead of color/histogram heuristics.
        return 0.0

    def detect(self, frame, input_size=416, active_category="ALL", orientation_deg=0):
        h, w = frame.shape[:2]

        rot = orientation_deg % 360
        if rot == 90:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            proc_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            proc_frame = frame

        # ------------------------------------------------------------------
        # TARGET LOCK MODE -- bypasses YOLO entirely (it has no concept of
        # arbitrary custom objects like "this specific tire"). Uses ORB
        # keypoint matching + RANSAC homography to find the reference
        # object directly in the frame. This is both more accurate (no more
        # coincidental matches against people/laptops/etc.) and faster (one
        # ORB pass instead of a YOLO forward pass + per-candidate scoring).
        # ------------------------------------------------------------------
        if self.target_locked:
            frame_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            match = self._locate_target(frame_gray)

            now = time.time()
            if match is not None:
                x1, y1, x2, y2, confidence = match
                self.last_target_box = [x1, y1, x2, y2]
                self.last_target_conf = confidence
                self.last_target_time = now
            elif self.last_target_box is not None and (now - self.last_target_time) < self.target_memory_ttl_seconds:
                # Brief tracking-continuity: reuse the last confirmed box for
                # a fraction of a second so a single missed frame doesn't
                # make the box flicker off, but let it disappear quickly if
                # the target is genuinely gone rather than lingering forever.
                x1, y1, x2, y2 = self.last_target_box
                confidence = self.last_target_conf
            else:
                return []

            # Un-rotate the box back into the original (unrotated) frame's
            # coordinate space -- same transform used for regular detections
            # below, applied here to a single box instead of a whole list.
            if rot == 90:
                ux1, uy1, ux2, uy2 = y1, h - x2, y2, h - x1
            elif rot == 180:
                ux1, uy1, ux2, uy2 = w - x2, h - y2, w - x1, h - y1
            elif rot == 270:
                ux1, uy1, ux2, uy2 = w - y2, x1, w - y1, x2
            else:
                ux1, uy1, ux2, uy2 = x1, y1, x2, y2

            ux1 = max(0, min(w, min(ux1, ux2)))
            uy1 = max(0, min(h, min(uy1, uy2)))
            ux2 = max(0, min(w, max(ux1, ux2)))
            uy2 = max(0, min(h, max(uy1, uy2)))

            return [{
                "class_name": self.target_label,
                "category": "CUSTOM",
                "confidence": round(float(confidence), 2),
                "x1": int(ux1),
                "y1": int(uy1),
                "x2": int(ux2),
                "y2": int(uy2),
                "engine": "DECS_TARGET_LOCK"
            }]

        # ------------------------------------------------------------------
        # REGULAR MULTI-CATEGORY DETECTION MODE -- unchanged.
        # ------------------------------------------------------------------
        conf_thresh = self.conf_threshold

        results = self.model.predict(
            source=proc_frame,
            imgsz=input_size,
            conf=conf_thresh,
            iou=0.45,
            device=self.device,
            verbose=False,
            half=True if self.device == "cuda" else False
        )

        detections = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        
        # Strict Class-Agnostic PyTorch NMS
        keep_indices = torchvision.ops.nms(
            boxes.xyxy, 
            boxes.conf, 
            iou_threshold=0.35
        )

        coords = boxes.xyxy[keep_indices].cpu().numpy().astype(int)
        confs = boxes.conf[keep_indices].cpu().numpy()
        classes = boxes.cls[keep_indices].cpu().numpy().astype(int)

        for i in range(len(confs)):
            cls_id = classes[i]
            raw_label = self.model.names.get(cls_id, f"object_{cls_id}").lower()
            bx1, by1, bx2, by2 = coords[i]

            # Revert rotation coordinates
            if rot == 90:
                x1, y1, x2, y2 = by1, h - bx2, by2, h - bx1
            elif rot == 180:
                x1, y1, x2, y2 = w - bx2, h - by2, w - bx1, h - by1
            elif rot == 270:
                x1, y1, x2, y2 = w - by2, bx1, w - by1, bx2
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2

            x1_c = max(0, min(w, min(x1, x2)))
            y1_c = max(0, min(h, min(y1, y2)))
            x2_c = max(0, min(w, max(x1, x2)))
            y2_c = max(0, min(h, max(y1, y2)))

            is_human = any(term in raw_label for term in HUMAN_SET)
            is_animal = any(term in raw_label for term in ANIMAL_SET)

            if is_human:
                category = "HUMAN"
                display_label = "HUMAN"
            elif is_animal:
                category = "ANIMAL"
                sub_label = raw_label.replace("animal", "").strip().upper()
                display_label = f"ANIMAL: {sub_label}" if sub_label else "ANIMAL"
            else:
                category = "OBJECT"
                sub_label = raw_label.replace("object", "").strip().upper()
                display_label = f"OBJECT: {sub_label}" if sub_label else "OBJECT"

            if active_category != "ALL" and category != active_category:
                continue

            detections.append({
                "class_name": display_label,
                "category": category,
                "confidence": round(float(confs[i]), 2),
                "x1": int(x1_c),
                "y1": int(y1_c),
                "x2": int(x2_c),
                "y2": int(y2_c),
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
    telemetry_payload = []
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        corner_len = max(8, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
        b_th = 2
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), b_th, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), b_th, cv2.LINE_AA)

        label_text = f"{cls_name} {int(conf * 100)}%"
        
        # --- ENLARGED TEXT & BADGE SETTINGS ---
        font_scale = 1  # Increased from 0.45 to 0.8 for significantly larger text
        font_thickness = 2 # Increased from 1 to 2 for extra legibility
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        padding = 6
        bg_x1 = x1
        bg_y1 = max(0, y1 - th - (padding * 2)) if y1 - th - (padding * 2) > 0 else y1
        bg_x2 = min(w, x1 + tw + (padding * 2))
        bg_y2 = min(h, bg_y1 + th + (padding * 2))

        # Render background pill badge
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw high-contrast text inside badge
        cv2.putText(
            frame, 
            label_text, 
            (bg_x1 + padding, bg_y1 + th + padding - 2), 
            font, 
            font_scale, 
            (0, 0, 0), 
            font_thickness, 
            cv2.LINE_AA
        )

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
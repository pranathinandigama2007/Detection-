"""
DECS - Universal Multi-Input Tactical Hub (Military-Grade Accuracy Edition)
============================================================================
- Universal Ingestion: RTMP, RTSP, HTTP/HTTPS, UDP, and Native Skydroid GCS Feeds
- Crash-proof, auto-reconnecting capture loop (proven stable design from
  the "detection f2" production build)
- Dual-model ensemble detection with periodic tiled deep passes for
  maximum small/distant-object recall at drone altitude (see
  drone_ingestion/detection_utils.py)
- Dynamic rotation routing (0deg, 90deg, 180deg, 270deg) -- works the same
  whether the feed is a gimbal-stabilized drone camera or a handheld phone
- Automatic IP geolocation fallback
"""

import sys
import os
import cv2
import json
import asyncio
import threading
import time
import numpy as np
import urllib.request
from typing import Set
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from drone_ingestion.detection_utils import (
    HighPerformanceDetector,
    TargetMatcher,
    draw_and_package,
    get_local_ip,
)


class DetectionRecorder:
    """
    Records the DETECTED feed (the same annotated frame streamed to the
    dashboard, boxes/labels included) to a local .mp4 file -- not the
    dashboard UI itself, just the processed video. Supports start/pause/
    resume/stop. The save folder is chosen by the user per-recording via
    the dashboard's "Save Location" field, since this is a local desktop
    app scenario rather than a browser-sandboxed one -- the Python backend
    writes directly to whatever filesystem path is given.
    """

    def __init__(self, default_dir):
        self.default_dir = default_dir
        self.writer = None
        self.state = "IDLE"  # IDLE, RECORDING, PAUSED
        self.filepath = None
        self.fps = 15.0
        self.frame_size = None
        self.lock = threading.Lock()

    def start(self, frame, save_dir=None):
        with self.lock:
            if self.state in ("RECORDING", "PAUSED"):
                return {"status": "ERROR", "message": "A recording is already in progress. Stop it first."}
            if frame is None:
                return {"status": "ERROR", "message": "No live frame available yet -- connect a stream first."}

            target_dir = (save_dir or "").strip() or self.default_dir
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                return {"status": "ERROR", "message": f"Could not create/access folder '{target_dir}': {e}"}

            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"DECS_detected_feed_{ts}.mp4"
            filepath = os.path.join(target_dir, filename)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, self.fps, self.frame_size)
            if not writer.isOpened():
                return {"status": "ERROR", "message": f"OpenCV could not open a writer for '{filepath}'."}

            self.writer = writer
            self.filepath = filepath
            self.state = "RECORDING"
            print(f"[RECORDER] Started -> {filepath}")
            return {"status": "RECORDING", "filepath": filepath}

    def write_frame(self, frame):
        # Called from the inference worker for every processed frame.
        with self.lock:
            if self.state != "RECORDING" or self.writer is None:
                return
            h, w = frame.shape[:2]
            if (w, h) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)

    def pause(self):
        with self.lock:
            if self.state == "RECORDING":
                self.state = "PAUSED"
                return {"status": "PAUSED", "filepath": self.filepath}
            return {"status": "ERROR", "message": f"Cannot pause from state {self.state}."}

    def resume(self):
        with self.lock:
            if self.state == "PAUSED":
                self.state = "RECORDING"
                return {"status": "RECORDING", "filepath": self.filepath}
            return {"status": "ERROR", "message": f"Cannot resume from state {self.state}."}

    def stop(self):
        with self.lock:
            if self.state == "IDLE" or self.writer is None:
                return {"status": "ERROR", "message": "No active recording to stop."}
            self.writer.release()
            self.writer = None
            path = self.filepath
            self.filepath = None
            self.frame_size = None
            self.state = "IDLE"
            print(f"[RECORDER] Stopped -> {path}")
            return {"status": "STOPPED", "filepath": path}

    def get_status(self):
        with self.lock:
            return {"state": self.state, "filepath": self.filepath}

app = FastAPI(title="DECS Universal Tactical C2 Hub", version="19.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LiveLocationState:
    def __init__(self):
        self.latitude = 17.0253
        self.longitude = 81.7779
        self.accuracy = 25.0

location_state = LiveLocationState()

# Resolve Host Location on startup
try:
    req = urllib.request.Request(
        "https://ipapi.co/json/",
        headers={"User-Agent": "DECS-Tactical-C2/1.0"}
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        data = json.loads(resp.read().decode())
        location_state.latitude = float(data.get("latitude", 17.0253))
        location_state.longitude = float(data.get("longitude", 81.7779))
except Exception:
    pass

class VideoHub:
    def __init__(self):
        self.video_sockets: Set[WebSocket] = set()
        self.telemetry_sockets: Set[WebSocket] = set()

    def add_video(self, ws: WebSocket): self.video_sockets.add(ws)
    def remove_video(self, ws: WebSocket): self.video_sockets.discard(ws)
    def add_telemetry(self, ws: WebSocket): self.telemetry_sockets.add(ws)
    def remove_telemetry(self, ws: WebSocket): self.telemetry_sockets.discard(ws)

hub = VideoHub()

class UniversalIngestionPipeline:
    """
    Crash-proof, self-healing capture pipeline (based on the proven
    "detection f2" ResilientPipeline design) extended to accept ANY input
    protocol OpenCV/FFmpeg can open: RTSP, RTMP, HTTP(S) MJPEG/MPEG-TS,
    UDP raw streams, and Skydroid GCS RTSP feeds. The capture loop itself
    doesn't need to know or care which protocol it's reading -- FFmpeg
    handles that -- so this class is unchanged in structure from the
    stable single-protocol version, only the URL fed into it differs.
    """

    def __init__(self):
        self.stream_url = None
        self.is_running = False

        self.cap_thread = None
        self.infer_thread = None

        self.latest_raw_frame = None
        self.latest_processed_jpeg = None
        self.latest_targets = []
        self.lock = threading.Lock()

        self.detector = None
        self.active_category = "ALL"
        self.active_orientation = 0

        self.recorder = DetectionRecorder(default_dir=os.path.join(BASE_DIR, "recordings"))
        self.target_matcher = TargetMatcher()

    def set_orientation(self, deg: int):
        self.active_orientation = deg % 360
        print(f"[TACTICAL HUB] Active Vision Angle set to: {self.active_orientation} deg")

    def set_filter(self, category: str):
        self.active_category = category.upper()

    def start(self, stream_url: str):
        self.stop()
        self.stream_url = stream_url.strip()
        self.is_running = True

        if self.detector is None:
            self.detector = HighPerformanceDetector(conf_threshold=0.22)

        print(f"[TACTICAL HUB] Pipeline active on: {self.stream_url}")

        self.cap_thread = threading.Thread(target=self._capture_pump, daemon=True)
        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)

        self.cap_thread.start()
        self.infer_thread.start()

    def stop(self):
        self.is_running = False
        if self.cap_thread and self.cap_thread.is_alive():
            self.cap_thread.join(timeout=0.6)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=0.6)

        # Never leave a video file handle open if the stream disconnects
        # mid-recording.
        if self.recorder.state != "IDLE":
            self.recorder.stop()

        self.latest_raw_frame = None
        self.latest_processed_jpeg = None
        self.latest_targets = []
        self.stream_url = None

    def _capture_pump(self):
        """
        Universal capture pump supporting:
        - RTSP / TCP
        - Direct RTMP
        - HTTP / HTTPS (MPEG-TS, MJPEG)
        - UDP Raw Video Streams
        - Skydroid GCS RTSP feeds
        Self-healing: survives reconnects, drops, and temporary stream gaps
        without crashing the process.
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay|"
            "max_delay;0|probesize;32|analyzeduration;0|sync;ext"
        )
        cap = None
        fail_count = 0

        while self.is_running:
            if cap is None or not cap.isOpened():
                # Direct hardware/network open via OpenCV CAP_FFMPEG -- this
                # single call transparently handles rtsp://, rtmp://, http://,
                # https://, and udp:// URLs; FFmpeg picks the right demuxer
                # based on the URL scheme.
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.3)
                fail_count = 0
                continue

            grabbed = cap.grab()
            if not grabbed:
                fail_count += 1
                if fail_count > 30:  # Stream temporarily interrupted
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
                continue

            fail_count = 0
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                with self.lock:
                    self.latest_raw_frame = frame

        if cap is not None:
            cap.release()

    def _inference_worker(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 65]

        while self.is_running:
            frame_sample = None
            deg = self.active_orientation
            cat = self.active_category

            with self.lock:
                if self.latest_raw_frame is not None:
                    frame_sample = self.latest_raw_frame.copy()

            if frame_sample is None:
                time.sleep(0.01)
                continue

            # Orientation-invariant inference pass (fast full-frame every
            # call, periodic tiled deep pass for small/distant targets --
            # see HighPerformanceDetector.detect in detection_utils.py)
            targets = self.detector.detect(
                frame_sample,
                input_size=288,
                active_category=cat,
                orientation_deg=deg
            )

            # If a target lock is active, collapse the full detection list
            # down to (at most) the single best match -- this is what
            # implements "stop all remaining detection, search only for
            # this one thing".
            if self.target_matcher.active:
                targets = self.target_matcher.find_best_match(frame_sample, targets)

            draw_and_package(
                frame_sample,
                targets,
                drone_telemetry={"lat": location_state.latitude, "lon": location_state.longitude}
            )

            # Record the DETECTED feed (post-annotation), not the raw feed
            # and not the dashboard UI -- exactly what gets streamed below.
            self.recorder.write_frame(frame_sample)

            _, buf = cv2.imencode('.jpg', frame_sample, encode_params)
            jpeg_bytes = buf.tobytes()

            with self.lock:
                self.latest_targets = targets
                self.latest_processed_jpeg = jpeg_bytes

            time.sleep(0.025)

pipeline = UniversalIngestionPipeline()

class StreamConfigRequest(BaseModel):
    source_type: str = "rtsp"  # "rtmp", "rtsp", "skydroid", "http", "https", "custom"
    url: str = ""
    ip: str = "127.0.0.1"
    port: int = 1935
    path: str = "live/drone1"

class OrientationRequest(BaseModel):
    orientation: int

class FilterConfigRequest(BaseModel):
    category: str

class ClientLocationReport(BaseModel):
    latitude: float
    longitude: float
    accuracy: float

@app.post("/api/stream/orientation")
async def update_stream_orientation(req: OrientationRequest):
    pipeline.set_orientation(req.orientation)
    return {"status": "SUCCESS", "orientation": pipeline.active_orientation}

@app.post("/api/stream/filter")
async def update_category_filter(req: FilterConfigRequest):
    pipeline.set_filter(req.category)
    return {"status": "SUCCESS", "active_category": pipeline.active_category}

@app.post("/api/system/location")
async def update_location(report: ClientLocationReport):
    location_state.latitude = report.latitude
    location_state.longitude = report.longitude
    location_state.accuracy = report.accuracy
    return {"status": "SUCCESS", "lat": report.latitude, "lon": report.longitude}

@app.get("/api/system/info")
async def get_system_info():
    return {
        "ip": get_local_ip(),
        "latitude": location_state.latitude,
        "longitude": location_state.longitude,
        "active_stream": pipeline.stream_url,
        "is_streaming": pipeline.is_running,
        "active_category": pipeline.active_category
    }

@app.post("/api/stream/connect")
async def connect_stream(cfg: StreamConfigRequest):
    # Route based on selected protocol / equipment
    if cfg.source_type == "custom" and cfg.url:
        target_uri = cfg.url.strip()
    elif cfg.source_type == "skydroid":
        # Direct Skydroid H12/H16 default LAN RTSP or custom URL
        target_uri = cfg.url.strip() if cfg.url else "rtsp://192.168.144.108:554/live/ch0"
    elif cfg.source_type in ["http", "https"]:
        target_uri = cfg.url.strip()
    elif cfg.source_type == "rtsp":
        if cfg.url:
            target_uri = cfg.url.strip()
        else:
            clean_path = cfg.path.lstrip("/")
            target_uri = f"rtsp://{cfg.ip}:{cfg.port}/{clean_path}"
    else:
        # Default internal MediaMTX RTSP bridge for incoming RTMP mobile feeds
        clean_path = cfg.path.lstrip("/")
        target_uri = f"rtsp://127.0.0.1:8554/{clean_path}"

    print(f"[STREAM INGEST] Resolving input URI: {target_uri}")
    pipeline.start(target_uri)
    return {"status": "CONNECTED", "url": target_uri}

@app.post("/api/stream/disconnect")
async def disconnect_stream():
    pipeline.stop()
    return {"status": "DISCONNECTED"}

# ---------------------------------------------------------------------------
# Recording controls -- records the DETECTED (annotated) feed only, not the
# whole dashboard. Start / Pause / Resume / Stop.
# ---------------------------------------------------------------------------
class RecordStartRequest(BaseModel):
    save_dir: str = ""  # user-chosen folder; blank = default ./recordings

@app.post("/api/record/start")
async def start_recording(req: RecordStartRequest):
    with pipeline.lock:
        frame = pipeline.latest_raw_frame
    if not pipeline.is_running or frame is None:
        return {"status": "ERROR", "message": "No active detected feed to record. Connect a stream first."}
    return pipeline.recorder.start(frame, save_dir=req.save_dir)

@app.post("/api/record/pause")
async def pause_recording():
    return pipeline.recorder.pause()

@app.post("/api/record/resume")
async def resume_recording():
    return pipeline.recorder.resume()

@app.post("/api/record/stop")
async def stop_recording():
    return pipeline.recorder.stop()

@app.get("/api/record/status")
async def recording_status():
    return pipeline.recorder.get_status()

# ---------------------------------------------------------------------------
# Target lock -- upload a reference image of a specific object/animal/person,
# the pipeline narrows detection down to matches of that specific target only.
# See TargetMatcher in drone_ingestion/detection_utils.py for how matching
# works and its real accuracy limits (classical CV, not deep face-recognition).
# ---------------------------------------------------------------------------
@app.post("/api/target/set")
async def set_target(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if ref_img is None:
        return {"status": "ERROR", "message": "Could not decode uploaded image."}

    # Ensure a detector exists (even before a stream is connected) so we can
    # run detection on the reference image itself to auto-crop the main
    # subject and infer its category (HUMAN/ANIMAL/OBJECT).
    if pipeline.detector is None:
        pipeline.detector = HighPerformanceDetector(conf_threshold=0.22)

    ref_targets = pipeline.detector.detect(ref_img, input_size=320, active_category="ALL", orientation_deg=0)
    if ref_targets:
        main = max(ref_targets, key=lambda t: (t["x2"] - t["x1"]) * (t["y2"] - t["y1"]))
        crop = ref_img[main["y1"]:main["y2"], main["x1"]:main["x2"]]
        category = main["category"]
        label = main["class_name"]
    else:
        # Nothing detected in the reference image (e.g. a tight face crop
        # with no full body) -- fall back to using the whole image.
        crop = ref_img
        category = None
        label = "CUSTOM TARGET"

    ok = pipeline.target_matcher.set_reference(crop, category_hint=category, label=label)
    if not ok:
        return {"status": "ERROR", "message": "Reference image was empty or invalid."}

    return {"status": "TARGET_LOCKED", "category": category, "label": label}

@app.post("/api/target/clear")
async def clear_target():
    pipeline.target_matcher.clear()
    return {"status": "TARGET_CLEARED"}

@app.get("/api/target/status")
async def target_status():
    return {
        "active": pipeline.target_matcher.active,
        "label": pipeline.target_matcher.label,
        "category": pipeline.target_matcher.ref_category,
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(ws: WebSocket):
    await ws.accept()
    hub.add_telemetry(ws)
    try:
        while True:
            targets = []
            with pipeline.lock:
                targets = list(pipeline.latest_targets)
            if targets:
                await ws.send_text(json.dumps({"type": "detection", "targets": targets}))
            await asyncio.sleep(0.08)
    except (WebSocketDisconnect, Exception):
        hub.remove_telemetry(ws)

@app.websocket("/ws/video")
async def websocket_video_endpoint(ws: WebSocket):
    await ws.accept()
    hub.add_video(ws)
    try:
        while True:
            frame_bytes = None
            with pipeline.lock:
                frame_bytes = pipeline.latest_processed_jpeg
            if frame_bytes is not None:
                await ws.send_bytes(frame_bytes)
            await asyncio.sleep(0.025)
    except (WebSocketDisconnect, Exception):
        hub.remove_video(ws)

frontend_path = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

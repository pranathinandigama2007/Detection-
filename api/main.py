"""
DECS - Multi-Input C2 Tactical Hub
==================================
- Target Lock Endpoint API with Instant Clean Validation
- Real-time High FPS Video & Telemetry Pipeline
- Multi-protocol ingestion & zero-latency recording
"""

import sys
import os
import cv2
import json
import asyncio
import threading
import time
import urllib.request
from typing import Set, Optional
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from drone_ingestion.detection_utils import (
    HighPerformanceDetector,
    draw_and_package,
    get_local_ip,
)

app = FastAPI(title="DECS Universal Tactical C2 Hub", version="18.2")

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
    def __init__(self):
        self.stream_url = None
        self.is_running = False
        
        self.cap_thread = None
        self.infer_thread = None
        
        self.latest_raw_frame = None
        self.latest_processed_jpeg = None
        self.latest_targets = []
        self.lock = threading.Lock()
        
        self.detector = HighPerformanceDetector(conf_threshold=0.35)
        self.active_category = "ALL"
        self.active_orientation = 0

        self.is_recording = False
        self.is_recording_paused = False
        self.record_writer = None
        self.record_filepath = None

    def set_orientation(self, deg: int):
        self.active_orientation = deg % 360

    def set_filter(self, category: str):
        self.active_category = category.upper()

    def start(self, stream_url: str):
        self.stop()
        self.stream_url = stream_url.strip()
        self.is_running = True

        self.cap_thread = threading.Thread(target=self._capture_pump, daemon=True)
        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)

        self.cap_thread.start()
        self.infer_thread.start()

    def stop(self):
        self.stop_recording()
        self.is_running = False
        if self.cap_thread and self.cap_thread.is_alive():
            self.cap_thread.join(timeout=0.6)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=0.6)
            
        self.latest_raw_frame = None
        self.latest_processed_jpeg = None
        self.latest_targets = []
        self.stream_url = None

    def start_recording(self, save_dir: Optional[str] = None):
        if not self.is_running:
            return False, "No active stream to record."
        
        if not save_dir or not os.path.exists(save_dir):
            save_dir = os.path.join(BASE_DIR, "recordings")
        
        os.makedirs(save_dir, exist_ok=True)
        filename = f"DECS_RECORDING_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.record_filepath = os.path.join(save_dir, filename)
        
        self.is_recording = True
        self.is_recording_paused = False
        return True, self.record_filepath

    def pause_recording(self):
        if self.is_recording:
            self.is_recording_paused = True
            return True, self.record_filepath
        return False, "Not recording"

    def resume_recording(self):
        if self.is_recording:
            self.is_recording_paused = False
            return True, self.record_filepath
        return False, "Not recording"

    def stop_recording(self):
        self.is_recording = False
        self.is_recording_paused = False
        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None
        path = self.record_filepath
        self.record_filepath = None
        return path

    def _capture_pump(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0|sync;ext"
        )
        cap = None
        fail_count = 0

        while self.is_running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.3)
                fail_count = 0
                continue

            grabbed = cap.grab()
            if not grabbed:
                fail_count += 1
                if fail_count > 30:
                    cap.release()
                    cap = None
                    time.sleep(0.4)
                else:
                    time.sleep(0.005)
                continue

            fail_count = 0
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                with self.lock:
                    self.latest_raw_frame = frame

        if cap is not None:
            cap.release()

    def _inference_worker(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 78]

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

            targets = self.detector.detect(
                frame_sample,
                input_size=416,
                active_category=cat,
                orientation_deg=deg
            )

            draw_and_package(
                frame_sample,
                targets,
                drone_telemetry={"lat": location_state.latitude, "lon": location_state.longitude}
            )

            if self.is_recording and not self.is_recording_paused:
                h, w = frame_sample.shape[:2]
                if self.record_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.record_writer = cv2.VideoWriter(self.record_filepath, fourcc, 20.0, (w, h))
                self.record_writer.write(frame_sample)

            _, buf = cv2.imencode('.jpg', frame_sample, encode_params)
            jpeg_bytes = buf.tobytes()

            with self.lock:
                self.latest_targets = targets
                self.latest_processed_jpeg = jpeg_bytes

            time.sleep(0.012)

pipeline = UniversalIngestionPipeline()

class StreamConfigRequest(BaseModel):
    source_type: str = "rtsp"
    url: str = ""
    ip: str = "127.0.0.1"
    port: int = 1935
    path: str = "live/drone1"

class OrientationRequest(BaseModel):
    orientation: int

class FilterConfigRequest(BaseModel):
    category: str

class RecordStartRequest(BaseModel):
    save_dir: Optional[str] = None

class ClientLocationReport(BaseModel):
    latitude: float
    longitude: float
    accuracy: float

@app.post("/api/target/set")
async def set_target_lock(file: UploadFile = File(...)):
    contents = await file.read()
    success, result_label = pipeline.detector.set_target_lock(contents, filename=file.filename)
    if success:
        return {"status": "TARGET_LOCKED", "label": result_label}
    return {"status": "ERROR", "message": result_label}

@app.post("/api/target/clear")
async def clear_target_lock():
    pipeline.detector.clear_target_lock()
    return {"status": "TARGET_CLEARED"}

@app.post("/api/record/start")
async def start_recording(req: RecordStartRequest):
    ok, path_or_msg = pipeline.start_recording(req.save_dir)
    if ok:
        return {"status": "RECORDING", "filepath": path_or_msg}
    return {"status": "ERROR", "message": path_or_msg}

@app.post("/api/record/pause")
async def pause_recording():
    ok, path_or_msg = pipeline.pause_recording()
    if ok:
        return {"status": "PAUSED", "filepath": path_or_msg}
    return {"status": "ERROR", "message": path_or_msg}

@app.post("/api/record/resume")
async def resume_recording():
    ok, path_or_msg = pipeline.resume_recording()
    if ok:
        return {"status": "RECORDING", "filepath": path_or_msg}
    return {"status": "ERROR", "message": path_or_msg}

@app.post("/api/record/stop")
async def stop_recording():
    saved_path = pipeline.stop_recording()
    return {"status": "STOPPED", "filepath": saved_path or "None"}

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
    if cfg.source_type == "custom" and cfg.url:
        target_uri = cfg.url.strip()
    elif cfg.source_type == "skydroid":
        target_uri = cfg.url.strip() if cfg.url else "rtsp://192.168.144.108:554/live/ch0"
    elif cfg.source_type in ["http", "https"]:
        target_uri = cfg.url.strip()
    elif cfg.source_type == "rtsp":
        target_uri = cfg.url.strip() if cfg.url else f"rtsp://{cfg.ip}:{cfg.port}/{cfg.path.lstrip('/')}"
    else:
        target_uri = f"rtsp://127.0.0.1:8554/{cfg.path.lstrip('/')}"

    pipeline.start(target_uri)
    return {"status": "CONNECTED", "url": target_uri}

@app.post("/api/stream/disconnect")
async def disconnect_stream():
    pipeline.stop()
    return {"status": "DISCONNECTED"}

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
            await asyncio.sleep(0.05)
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
            await asyncio.sleep(0.02)
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
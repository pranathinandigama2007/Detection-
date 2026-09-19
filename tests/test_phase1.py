# tests/test_phase1.py
import pytest
import asyncio
import os
from fastapi.testclient import TestClient

from api.main import app
from database_module.lite_db import init_sqlite_db, save_detection_log, fetch_detection_history
from drone_ingestion.detection_utils import ONNXDetector

client = TestClient(app)

@pytest.mark.asyncio
async def test_sqlite_database_operations():
    """Verifies embedded SQLite creation, inserting, and querying."""
    # Ensure clean state
    if os.path.exists("decs_local.db"):
        os.remove("decs_local.db")
        
    await init_sqlite_db()
    
    # Insert dummy detection
    await save_detection_log(
        stream_id="test_drone",
        class_name="car",
        confidence=0.88,
        lat=17.0253,
        lon=81.7779,
        engine="ONNX_CPU"
    )
    
    # Query history
    history = await fetch_detection_history(min_confidence=0.50, limit=10)
    assert len(history) == 1
    assert history[0]["class_name"] == "car"
    assert history[0]["confidence"] == 0.88


def test_fastapi_endpoints():
    """Verifies basic API responses without external database dependency."""
    response = client.get("/")
    assert response.status_code == 200

    rtmp_resp = client.get("/api/stream/generate-rtmp?stream_id=drone_test")
    assert rtmp_resp.status_code == 200
    assert "rtmp_url" in rtmp_resp.json()


def test_onnx_detector_initialization():
    """Verifies ONNX detector model structure loading."""
    if os.path.exists("yolov8n.onnx"):
        detector = ONNXDetector(model_path="yolov8n.onnx")
        assert detector.net is not None
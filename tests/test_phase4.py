# tests/test_phase4.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_websocket_telemetry_stream():
    with client.websocket_connect("/ws/telemetry") as websocket:
        assert websocket is not None
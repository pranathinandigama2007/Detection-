# api/websocket_manager.py
from fastapi import WebSocket
from typing import List
import json
from logging_module.logger import get_logger

logger = get_logger("websocket_manager")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_client_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("websocket_client_disconnected", total_connections=len(self.active_connections))

    async def broadcast_telemetry(self, drone_id: str, payload: dict):
        """Broadcasts live MAVLink telemetry to all connected clients."""
        message = json.dumps({"type": "telemetry", "drone_id": drone_id, "data": payload})
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error("error_broadcasting_telemetry", error=str(e))

    async def broadcast_detection(self, stream_id: str, detections: list):
        """Broadcasts live AI detections from scrcpy/RTSP to all connected clients."""
        message = json.dumps({"type": "detection", "stream_id": stream_id, "targets": detections})
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error("error_broadcasting_detection", error=str(e))

ws_manager = ConnectionManager()
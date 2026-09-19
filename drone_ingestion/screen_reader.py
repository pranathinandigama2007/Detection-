"""
DECS - Live Screen Region Ingestion & Detection Engine
======================================================
Captures video from any 3rd party software/browser viewer displaying
the drone stream and runs real-time target detection.
"""

import cv2
import json
import asyncio
import os
import sys
import time
import argparse
import numpy as np
from mss import mss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import websockets
    from websockets.exceptions import InvalidURI, WebSocketException
except ImportError:
    print("[ERROR] websockets missing. Install via: pip install websockets")
    sys.exit(1)

from drone_ingestion.detection_utils import (
    ONNXDetector,
    draw_and_package,
    draw_osd,
    apply_clahe,
    setup_display_window,
    ADMIN_CONFIG,
    WS_URL
)


def parse_args():
    parser = argparse.ArgumentParser(description="DECS Screen Capture Detection Engine")
    parser.add_argument("--top", type=int, default=100, help="Screen region Y offset (top)")
    parser.add_argument("--left", type=int, default=100, help="Screen region X offset (left)")
    parser.add_argument("--width", type=int, default=1280, help="Screen region width")
    parser.add_argument("--height", type=int, default=720, help="Screen region height")
    parser.add_argument("--stream_id", type=str, default="screen_capture_feed")
    parser.add_argument("--input_size", type=int, default=320)
    return parser.parse_args()


async def main():
    args = parse_args()
    stream_id = args.stream_id
    window_name = f"DECS Screen Target Tracking - [{stream_id}]"

    # Define capture bounding box on monitor 1
    bounding_box = {
        "top": args.top,
        "left": args.left,
        "width": args.width,
        "height": args.height
    }

    detector = ONNXDetector(
        model_path="yolov8n.onnx", 
        conf_threshold=ADMIN_CONFIG.get("min_confidence", 0.35)
    )

    print(f"[DECS Screen Capture] Capturing Region: {bounding_box}")
    setup_display_window(window_name)

    ws = None
    try:
        ws = await websockets.connect(WS_URL)
        print(f"[{stream_id}] Connected to Telemetry Hub: {WS_URL}")
    except (ConnectionRefusedError, OSError, InvalidURI, WebSocketException):
        print(f"[WARNING] Telemetry Hub offline. Running in Local Mode...")

    last_tx = 0
    sct = mss()

    try:
        while True:
            # Grab real-time screen pixels
            sct_img = sct.grab(bounding_box)
            # Convert BGRA screen image to BGR for OpenCV processing
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

            if ADMIN_CONFIG.get("clahe_enabled", False):
                frame = apply_clahe(frame)

            # Inference Run
            detections = detector.detect(frame, input_size=args.input_size)
            telemetry = draw_and_package(frame, detections)
            draw_osd(frame, ADMIN_CONFIG.get("engine_mode", "FAST_ONNX"), ADMIN_CONFIG.get("min_confidence", 0.35))

            # Send Telemetry over WebSocket
            now = time.time()
            if ws and (telemetry or (now - last_tx) > 1.0):
                payload = {
                    "type": "detection",
                    "stream_id": stream_id,
                    "targets": telemetry
                }
                try:
                    await ws.send(json.dumps(payload))
                    last_tx = now
                except WebSocketException:
                    ws = None

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"[{stream_id}] Stopped by operator.")
                break

            await asyncio.sleep(0.001)

    finally:
        if ws:
            await ws.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
    
import pytest
from ai_analytics.detector import TargetDetector

def test_target_detector_pipeline():
    detector = TargetDetector(confidence_threshold=0.5)
    
    mock_frame_meta = {"frame_id": 101, "stream_id": "stream_alpha"}
    mock_telemetry = {
        "lat": 16.8200,
        "lon": 82.2300,
        "alt": 100.0,
        "heading": 90.0
    }
    
    detections = detector.process_frame(mock_frame_meta, mock_telemetry)
    
    assert len(detections) == 1
    target = detections[0]
    
    assert target["class_name"] == "surveillance_target"
    assert target["confidence"] == 0.88
    
    # Due East heading (90 deg): latitude remains ~16.82, longitude increases Eastward
    assert pytest.approx(target["latitude"], abs=1e-5) == 16.8200
    assert target["longitude"] > 82.2300

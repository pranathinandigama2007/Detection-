# database_module/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime
from geoalchemy2 import Geometry
from database_module.session import Base

class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(String, index=True)
    class_name = Column(String, index=True)
    confidence = Column(Float)
    
    # Raw Coordinates
    latitude = Column(Float)
    longitude = Column(Float)
    
    # PostGIS Spatial Point (SRID 4326 = WGS 84 GPS standard)
    location = Column(Geometry(geometry_type='POINT', srid=4326))
    
    # Enable timezone=True so PostgreSQL handles UTC timestamps properly
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
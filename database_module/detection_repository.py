# database_module/detection_repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from database_module.models import DetectionLog
from database_module.session import AsyncSessionLocal
from logging_module.logger import get_logger

logger = get_logger("detection_repository")

async def log_detections_to_db(stream_id: str, detections: list):
    """
    Asynchronously persists target detection events into PostGIS.
    """
    if not detections:
        return

    async with AsyncSessionLocal() as session:
        try:
            for det in detections:
                lat = det.get("latitude")
                lon = det.get("longitude")
                
                # Create Spatial Point geometry for PostGIS
                point = Point(lon, lat)
                spatial_geom = from_shape(point, srid=4326)

                log_entry = DetectionLog(
                    stream_id=stream_id,
                    class_name=det.get("class_name"),
                    confidence=det.get("confidence"),
                    latitude=lat,
                    longitude=lon,
                    location=spatial_geom
                )
                session.add(log_entry)

            await session.commit()
            logger.info("detections_logged_to_postgis", count=len(detections))
        except Exception as e:
            await session.rollback()
            logger.error("failed_to_log_detections", error=str(e))
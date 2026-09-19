# message_broker/redis_producer.py
import redis.asyncio as aioredis
import json
from config_module.settings import settings
from logging_module.logger import get_logger

logger = get_logger("redis_producer")

class StreamProducer:
    def __init__(self, redis_url: str = None):
        url = redis_url or getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
        self.redis = aioredis.from_url(url, decode_responses=True)

    async def publish_telemetry(self, drone_id: str, telemetry_data: dict):
        """Pushes real-time MAVLink telemetry into a Redis Stream."""
        stream_key = f"stream:telemetry:{drone_id}"
        await self.redis.xadd(
            stream_key,
            {"data": json.dumps(telemetry_data)},
            maxlen=1000,
            approximate=True
        )

    async def publish_frame_metadata(self, stream_id: str, frame_data: dict):
        """Pushes decoded video frame payload into a Redis Stream."""
        stream_key = f"stream:frames:{stream_id}"
        await self.redis.xadd(
            stream_key,
            frame_data,
            maxlen=500,
            approximate=True
        )

    async def close(self):
        await self.redis.aclose()
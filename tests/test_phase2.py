# tests/test_phase2.py
import pytest
import pytest_asyncio
import json
from testcontainers.redis import RedisContainer
from message_broker.redis_producer import StreamProducer

@pytest_asyncio.fixture
async def redis_producer():
    # Spins up a temporary Redis container automatically
    with RedisContainer("redis:7-alpine") as redis_container:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        redis_url = f"redis://{host}:{port}"
        
        producer = StreamProducer(redis_url=redis_url)
        yield producer
        await producer.close()

@pytest.mark.asyncio
async def test_redis_telemetry_stream(redis_producer):
    drone_id = "test_drone_01"
    telemetry_payload = {
        "lat": 16.82,
        "lon": 82.23,
        "alt": 45.5,
        "heading": 180.0
    }
    
    # 1. Publish telemetry payload to Redis Stream
    await redis_producer.publish_telemetry(drone_id=drone_id, telemetry_data=telemetry_payload)
    
    # 2. Read back from Redis Stream
    stream_key = f"stream:telemetry:{drone_id}"
    messages = await redis_producer.redis.xread({stream_key: '0-0'}, count=1)
    
    assert len(messages) > 0
    stream_name, entries = messages[0]
    entry_id, data = entries[0]
    
    parsed = json.loads(data["data"])
    assert parsed["lat"] == 16.82
    assert parsed["heading"] == 180.0

    # Clean up
    await redis_producer.redis.delete(stream_key)
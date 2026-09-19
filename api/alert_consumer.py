import asyncio
import json
import aio_pika
import structlog

logger = structlog.get_logger()

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
RABBITMQ_QUEUE = "alerts_queue"


async def process_alert(message: aio_pika.IncomingMessage):
    async with message.process():
        alert_data = json.loads(message.body.decode())
        
        # Renamed keyword parameter from event to alert_event to avoid structlog key collision
        logger.warning(
            "🚨 CRITICAL ALERT DISPATCHED 🚨",
            alert_event=alert_data.get("event"),
            target=alert_data.get("class"),
            confidence=f"{int(alert_data.get('confidence', 0) * 100)}%",
            lat=alert_data.get("latitude"),
            lon=alert_data.get("longitude")
        )
        
        target_name = alert_data.get("class", "UNKNOWN").upper()
        lat = alert_data.get("latitude")
        lon = alert_data.get("longitude")
        print(f"\n[ALERT WORKER] -> ACTION REQUIRED: High confidence {target_name} identified at LAT: {lat}, LON: {lon}\n")


async def main():
    logger.info("starting_rabbitmq_alert_consumer_worker")
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(RABBITMQ_QUEUE, durable=True)
        
        logger.info("listening_for_threat_alerts_on_rabbitmq", queue=RABBITMQ_QUEUE)
        
        await queue.consume(process_alert)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("alert_consumer_worker_stopped")
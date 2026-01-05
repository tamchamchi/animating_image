import asyncio
import websockets
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class WebSocketService:
    def __init__(self, third_party_websocket_url: str):
        self.third_party_websocket_url = third_party_websocket_url
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connection_lock = asyncio.Lock()
        self._last_attempt = 0.0
        self.RETRY_INTERVAL = 5  # seconds

        logger.info(
            f"WebSocketService initialized for URL: {third_party_websocket_url}")

    async def _ensure_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        async with self._connection_lock:
            if self._websocket and not self._websocket.closed:
                return self._websocket

            now = time.time()
            if now - self._last_attempt < self.RETRY_INTERVAL:
                return None

            self._last_attempt = now
            logger.info(
                f"Trying to connect WS: {self.third_party_websocket_url}")

            try:
                self._websocket = await websockets.connect(self.third_party_websocket_url)
                logger.info("WebSocket connected")
                return self._websocket
            except Exception as e:
                logger.warning(f"WebSocket not available yet: {e}")
                self._websocket = None
                return None

    async def send_data_to_third_party(self, data: dict) -> dict:
        json_data = json.dumps(data)

        websocket = await self._ensure_connection()
        if not websocket:
            logger.debug("WebSocket not connected, skip sending")
            return {
                "status": "skipped",
                "message": "WebSocket server not available yet"
            }

        try:
            await websocket.send(json_data)
            logger.info("Data sent to third party")
            return {"status": "success"}
        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                "WebSocket closed during send, will reconnect later")
            self._websocket = None
            return {"status": "retry_later"}
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}", exc_info=True)
            self._websocket = None
            return {"status": "error", "message": str(e)}

    async def close_connection(self):
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            logger.info("WebSocket connection closed")
        self._websocket = None

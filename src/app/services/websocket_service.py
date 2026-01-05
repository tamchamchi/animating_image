import asyncio
import websockets
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebSocketService:
    """
    Manages a persistent WebSocket connection to a third-party server
    for forwarding data.

    This service ensures that a single WebSocket connection is maintained
    and automatically attempts to reconnect if the connection is lost
    during data transmission.
    """

    def __init__(self, third_party_websocket_url: str):
        """
        Initializes the WebSocketService with the target third-party WebSocket URL.

        Args:
            third_party_websocket_url (str): The full URL of the third-party
                                             WebSocket server (e.g., "ws://host:port/path").
        """
        self.third_party_websocket_url = third_party_websocket_url
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connection_lock = asyncio.Lock()
        logger.info(
            f"WebSocketService initialized for URL: {third_party_websocket_url}")

    async def _ensure_connection(self) -> websockets.WebSocketClientProtocol:
        """
        Ensures that an active WebSocket connection to the third-party server exists.
        If no connection is active or if the existing one is closed, it attempts
        to establish a new connection.

        This method uses an asyncio.Lock to prevent multiple concurrent attempts
        to establish a connection.

        Raises:
            Exception: If an error occurs during connection establishment.

        Returns:
            websockets.WebSocketClientProtocol: The active WebSocket connection.
        """
        async with self._connection_lock:
            if self._websocket and not self._websocket.closed:
                logger.debug("Existing WebSocket connection is active.")
                return self._websocket

            logger.info(
                f"Attempting to establish WebSocket connection to: {self.third_party_websocket_url}")
            try:
                self._websocket = await websockets.connect(self.third_party_websocket_url)
                logger.info("WebSocket connection established successfully.")
                return self._websocket
            except Exception as e:
                logger.error(
                    f"Failed to establish WebSocket connection: {e}", exc_info=True)
                self._websocket = None
                raise  # Re-raise the exception to inform the caller about the connection failure

    async def send_data_to_third_party(self, data: dict) -> dict:
        """
        Sends data to the third-party server via the maintained WebSocket connection.
        If the connection is closed during an attempt, it tries to reconnect and resend the data.

        Args:
            data (dict): The dictionary data to be sent, which will be
                         serialized to a JSON string.

        Returns:
            dict: A dictionary indicating the status of the data forwarding
                  (e.g., {"status": "success", "message": "..."}).
        """
        json_data = json.dumps(data)
        try:
            websocket = await self._ensure_connection()
            await websocket.send(json_data)
            logger.info(
                f"Successfully forwarded data to third party: {json_data}")
            return {"status": "success", "message": "Data forwarded successfully."}
        except websockets.exceptions.ConnectionClosedOK:
            logger.warning(
                "WebSocket connection was closed. Attempting to reconnect and resend data.")
            self._websocket = None  # Reset connection to force _ensure_connection to create a new one
            try:
                websocket = await self._ensure_connection()
                await websocket.send(json_data)
                logger.info(
                    f"Successfully forwarded data to third party (re-attempt): {json_data}")
                return {"status": "success", "message": "Data forwarded successfully after reconnect."}
            except Exception as e:
                logger.error(
                    f"Failed to send data after reconnect: {e}", exc_info=True)
                return {"status": "error", "message": f"Failed to forward data after reconnect: {str(e)}"}
        except Exception as e:
            logger.error(
                f"Failed to send data to third party via WebSocket: {e}", exc_info=True)
            self._websocket = None  # Reset connection on any other error
            return {"status": "error", "message": f"Failed to forward data: {str(e)}"}

    async def close_connection(self):
        """
        Closes the active WebSocket connection to the third-party server if it is open.
        Logs if no active connection exists.
        """
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            logger.info("WebSocket connection with third party closed.")
            self._websocket = None
        else:
            logger.info("No active WebSocket connection to close.")

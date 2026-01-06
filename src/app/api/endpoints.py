from fastapi import APIRouter, File, Form, Query, UploadFile
import json
import logging
import websockets

from src.app.services.animation_service import AnimationService
from src.app.services.background_service import BackgroundService
from src.app.services.character_service import CharacterService
from src.app.services.game_service import GameService
from src.app.schema.location_data import LocationData

logger = logging.getLogger(__name__)


def create_api_router(connected_websocket_clients: set) -> APIRouter:
    router = APIRouter()

    char_service = CharacterService()
    anim_service = AnimationService()
    game_service = GameService()
    background_service = BackgroundService()

    # --- 1. Create Character ---
    @router.post("/character/create-by-face")
    async def create_char_face(
        face_image: UploadFile = File(...), body_image: UploadFile = File(...)
    ):
        return await char_service.create_from_face(face_image, body_image)

    @router.post("/character/create-by-prompt")
    async def create_char_prompt(prompt: str = Form(...)):
        return await char_service.create_from_prompt(prompt)

    # --- 2. Create Animation ---
    @router.post("/animation/init")
    async def anim_init(file: UploadFile = File(...)):
        return await anim_service.init_session(file)

    @router.post("/animation/{anim_id}/step1")
    async def anim_step1(anim_id: str):
        return await anim_service.step1_decompose(anim_id)

    @router.post("/animation/{anim_id}/step2")
    async def anim_step2(anim_id: str):
        return await anim_service.step2_pose(anim_id)

    @router.post("/animation/{anim_id}/step3")
    async def anim_step3(anim_id: str, action: str = "walk"):
        return await anim_service.step3_animate(anim_id, action)

    # --- 3. Create Background ---
    @router.post(
        "/background/{anim_id}/analyze/model",
    )
    async def analyze_background_model(
        anim_id: str,
        file: UploadFile = File(...,
                                description="Background image file (jpg, png)"),
        confidence_threshold: float = Query(
            0.40,
            ge=0.0,
            le=1.0,
            description="Confidence threshold for object detection (0.0 to 1.0)",
        ),
    ):
        return await background_service.get_polygon_by_model(
            anim_id=anim_id, file=file, confidence_threshold=confidence_threshold
        )

    @router.post(
        "/background/{anim_id}/analyze/svg",
    )
    async def analyze_background_svg(
        anim_id: str,
        file: UploadFile = File(...,
                                description="Background image file (jpg, png)"),
        top_k: int = Query(
            30, ge=1, le=100, description="Number of top polygons to extract"
        ),
    ):
        return await background_service.get_polygon_by_svg(
            anim_id=anim_id, file=file, top_k=top_k
        )

    # --- 4. Create Game Zone ---
    @router.post("/game/{game_id}/get_resource")
    async def get_resource(game_id: str):
        return await game_service.get_resources(game_id)

    # --- 5. Update Location ---
    @router.post("/updateLocation")
    async def update_character_location(location_data: LocationData):
        logger.info(
            f"Received updateLocation API call: {location_data.dict()}")

        if connected_websocket_clients:
            message = json.dumps(location_data.dict())
            disconnected_clients = []
            for client in connected_websocket_clients:
                try:
                    await client.send(message)
                    logger.debug(
                        f"Sent data to WebSocket client {client.remote_address}")
                except websockets.exceptions.ConnectionClosedOK:
                    logger.warning(
                        f"Client {client.remote_address} was already closed, will remove.")
                    disconnected_clients.append(client)
                except Exception as e:
                    logger.error(
                        f"Failed to send data to WebSocket client {client.remote_address}: {e}")
                    disconnected_clients.append(client)

            for client in disconnected_clients:
                if client in connected_websocket_clients:
                    connected_websocket_clients.remove(client)

            return {"status": "success", "message": "Location data sent to WebSocket clients"}
        else:
            logger.warning(
                "No WebSocket clients connected to send location data.")
            return {"status": "warning", "message": "No WebSocket clients connected"}

    return router

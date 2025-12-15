from fastapi import APIRouter, File, Form, UploadFile

from src.app.services.animation_service import AnimationService
from src.app.services.character_service import CharacterService
from src.app.services.game_service import GameService

router = APIRouter()
char_service = CharacterService()
anim_service = AnimationService()
game_service = GameService()


# --- 1. Create Character ---
@router.post("/character/create-by-face")
async def create_char_face(
    face_image: UploadFile = File(...),
    body_image: UploadFile = File(...)
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

# --- 4. Create Game Zone ---
@router.post("/game/{game_id}/get_resource")
async def get_resource(game_id: str):
    return await game_service.get_resources(game_id)

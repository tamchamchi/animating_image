import asyncio
from src.app.core.config import settings

# Import các class AI của bạn từ thư mục src
from src.face_segmenter import SegFormerB5FaceSegmenter
from src.text_to_image import NanoBananaGenerator
from src.concept_decomposer import ConcreteObjectDecomposer
from src.pose_estimator import MMPoseEstimator
from src.animator import MetaAnimator

class AIContainer:
    def __init__(self):
        print("--- LOADING AI MODELS (This may take a while) ---")
        
        # 1. Init Character Creators
        self.face_segmenter = SegFormerB5FaceSegmenter(device=settings.DEVICE) # hoặc cpu
        self.generator = NanoBananaGenerator(api_key=settings.GEMINI_API_KEY)
        
        # 2. Init Animation Tools
        self.decomposer = ConcreteObjectDecomposer(device=settings.DEVICE) 
        
        self.pose_estimator = MMPoseEstimator(
            cfg_path=settings.MMPOSE_CONFIG,
            ckpt_path=settings.MMPOSE_CHECKPOINT,
            device=settings.DEVICE
        )
        
        self.animator = MetaAnimator()

        # 3. Concurrency Control
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
        print("--- AI MODELS LOADED SUCCESSFULLY ---")

# Singleton instance
ai_container = AIContainer()
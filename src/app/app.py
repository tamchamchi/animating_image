import os
import sys
import streamlit as st
from pathlib import Path
import torch
from dotenv import load_dotenv

# --- SYSTEM PATH SETUP ---
# Fix import issues to allow importing from 'src' and 'steps'
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
os.chdir(project_root)

# --- IMPORTS ---
from src.animator.meta_animator import MetaAnimator  # noqa: E402
from src.concept_decomposer.object_decomposer.object_decomposer import ConcreteObjectDecomposer  # noqa: E402
from src.image_style_transfer.nano_banana_style_transfer import NanoBananaStyleTransfer  # noqa: E402
from src.pipeline.animation_pipeline import AnimationGenerationPipeline  # noqa: E402
from src.pose_estimator.mmpose_estimator import MMPoseEstimator  # noqa: E402
from src.text_to_image import NanoBananaGenerator  # noqa: E402
from src.face_segmenter import SegFormerB5FaceSegmenter  # noqa: E402

# Import Step Routers
import steps.step1_character as step1  # noqa: E402
import steps.step2_animation as step2  # noqa: E402
import steps.step3_background as step3  # noqa: E402
import steps.step4_render as step4  # noqa: E402

load_dotenv()
st.set_page_config(page_title="AI Animation Studio", layout="wide")

# --- MODEL LOADING (Singleton) ---


@st.cache_resource
def load_models():
    """Initialize models once and cache them."""
    api_key = os.getenv("GOOGLE_API_KEY")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_cfg = os.getenv("POSE_MODEL_CFG_PATH")
    pose_ckpt = os.getenv("POSE_MODEL_CKPT_PATH")

    if not api_key:
        st.error("Missing GOOGLE_API_KEY in .env")
        return None, None

    print(f"🚀 Loading models on {device}...")

    # Initialize components
    gen_model = NanoBananaGenerator(api_key)
    style_model = NanoBananaStyleTransfer(api_key)
    pose_model = MMPoseEstimator(
        cfg_path=pose_cfg, ckpt_path=pose_ckpt, device=device)

    decomposer_instance = ConcreteObjectDecomposer()

    face_segmenter = SegFormerB5FaceSegmenter(device=device)

    # Initialize Pipeline
    pipeline = AnimationGenerationPipeline(
        style_transfer=style_model,
        object_decomposer=decomposer_instance,
        pose_estimator=pose_model,
        animator=MetaAnimator()
    )

    return pipeline, gen_model, decomposer_instance, face_segmenter

# --- STATE MANAGEMENT ---


def init_session_state():
    """Initialize global variables."""
    defaults = {
        'step': 1,
        'char_name': "character_01",
        'char_image': None,
        'output_dir': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- MAIN APP ---


def main():
    init_session_state()
    pipeline, image_generator, object_decomposer, face_segmenter = load_models()

    if not pipeline:
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("🎥 AI Studio")
    options = ["1. Character", "2. Animation", "3. Background", "4. Export"]

    # Sync sidebar with session state
    current_index = st.session_state.step - 1
    selected_option = st.sidebar.radio(
        "Workflow:", options, index=current_index)

    # Update step based on selection
    st.session_state.step = int(selected_option.split(".")[0])

    # --- ROUTER LOGIC ---
    if st.session_state.step == 1:
        step1.show(image_generator, face_segmenter)
    elif st.session_state.step == 2:
        step2.show(pipeline)
    elif st.session_state.step == 3:
        step3.show(object_decomposer)
    elif st.session_state.step == 4:
        step4.show()

    # Footer
    st.markdown("---")
    st.caption(f"AI Animation Pipeline | Step {st.session_state.step}/4")


if __name__ == "__main__":
    main()

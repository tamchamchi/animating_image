import os
import streamlit as st
from pathlib import Path
from PIL import Image
import torch
from dotenv import load_dotenv
import glob
import sys

# --- FIX IMPORT ERROR ---
# Get the absolute path of the current file
current_file_path = Path(__file__).resolve()

# Determine the project root (2 levels up from src/app/)
project_root = current_file_path.parents[2]

# Add project root to sys.path if not already present
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Change working directory to project_root so relative paths (like .env) work correctly
os.chdir(project_root)
# -----------------------------

# Import custom modules
# Assuming directory structure remains as original script
from src.animator.meta_animator import MetaAnimator  # noqa: E402
from src.concept_decomposer.object_decomposer.object_decomposer import ( # noqa: E402
    ConcreteObjectDecomposer,
)
from src.image_style_transfer.nano_banana_style_transfer import NanoBananaStyleTransfer # noqa: E402
from src.pipeline.animation_pipeline import AnimationGenerationPipeline # noqa: E402
from src.pipeline.input_data import AnimationPipelineInput # noqa: E402
from src.pose_estimator.mmpose_estimator import MMPoseEstimator # noqa: E402
from src.utils.prompt import PROMPT_IMAGE_STYLE_TRANSFER, PROMPT_SUBJECT_GENERATION # noqa: E402
from src.text_to_image import NanoBananaGenerator # noqa: E402

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="AI Character Animator", layout="wide")

# --- PART 1: INITIALIZE MODELS (Using Cache to prevent reloading) ---
@st.cache_resource
def load_models():
    """Initialize the entire pipeline. This function runs only once."""
    API_KEY = os.getenv("GOOGLE_API_KEY")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for critical environment variables
    POSE_MODEL_CFG_PATH = os.getenv("POSE_MODEL_CFG_PATH")
    POSE_MODEL_CKPT_PATH = os.getenv("POSE_MODEL_CKPT_PATH")
    
    if not API_KEY:
        st.error("GOOGLE_API_KEY not found in .env file")
        return None, None
        
    print(f"Loading models on {DEVICE}...")

    # Initialize individual components
    image_generator = NanoBananaGenerator(API_KEY)
    style_transfer = NanoBananaStyleTransfer(API_KEY)
    object_decomposer = ConcreteObjectDecomposer()
    
    pose_estimator = MMPoseEstimator(
        cfg_path=POSE_MODEL_CFG_PATH, 
        ckpt_path=POSE_MODEL_CKPT_PATH, 
        device=DEVICE
    )
    
    animator = MetaAnimator()

    # Initialize the main Pipeline
    pipeline = AnimationGenerationPipeline(
        style_transfer=style_transfer,
        object_decomposer=object_decomposer,
        pose_estimator=pose_estimator,
        animator=animator
    )
    
    return pipeline, image_generator

# Load models
pipeline, image_generator = load_models()

# --- PART 2: USER INTERFACE (UI) ---

st.title("🧙‍♂️ AI Character Animator")
st.markdown("Tạo hoạt hình nhân vật từ văn bản hoặc hình ảnh.")

# Configuration Sidebar
with st.sidebar:
    st.header("Cấu hình")
    char_name = st.text_input("Tên nhân vật (Viết liền, không dấu)", value="char_demo")
    
    # Select Input Mode
    mode = st.radio("Chọn chế độ đầu vào:", ("Text Description", "Upload Image"))
    
    # Upload Style Reference Image (Required for pipeline)
    st.subheader("Ảnh phong cách (Style Ref)")
    style_ref_file = st.file_uploader("Upload ảnh Style Reference", type=["png", "jpg", "jpeg"])
    
    # Select Actions
    all_actions = ["standing", "jumping", "running", "jesse_dancing", "waving"]
    selected_actions = st.multiselect("Chọn hành động:", all_actions, default=["standing", "waving"])

# --- PART 3: LOGIC PROCESSING ---

if not pipeline:
    st.stop() # Stop if models failed to load

# Variable to hold the Content Image
content_image = None
generated_prompt = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Đầu vào")
    
    # Handle Input based on Mode
    if mode == "Text Description":
        user_prompt = st.text_area("Mô tả nhân vật của bạn:", 
                                   value="a child with blue glass wearing a red hoodie and green pant")
        if st.button("Tạo ảnh từ văn bản"):
            if user_prompt:
                with st.spinner("Đang tạo nhân vật từ văn bản..."):
                    full_prompt = PROMPT_SUBJECT_GENERATION.format(subject=user_prompt)
                    try:
                        content_image = image_generator.generate(prompt=full_prompt)
                        st.session_state['generated_image'] = content_image # Save to session state
                        st.success("Đã tạo ảnh xong!")
                    except Exception as e:
                        st.error(f"Lỗi tạo ảnh: {e}")
            else:
                st.warning("Vui lòng nhập mô tả.")
        
        # Retrieve image from session if already generated
        if 'generated_image' in st.session_state:
            content_image = st.session_state['generated_image']
            st.image(content_image, caption="Ảnh nhân vật được tạo", use_container_width=True)

    elif mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload ảnh nhân vật", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            content_image = Image.open(uploaded_file).convert("RGB")
            st.image(content_image, caption="Ảnh nhân vật đã upload", use_container_width=True)

# Run Pipeline Button
with col2:
    st.subheader("2. Kết quả Hoạt hình")
    
    if st.button("🎬 Bắt đầu tạo Animation", type="primary"):
        # Validate inputs
        if not char_name:
            st.error("Vui lòng nhập tên nhân vật.")
        elif content_image is None:
            st.error("Vui lòng cung cấp ảnh nhân vật (Tạo từ text hoặc Upload).")
        elif style_ref_file is None:
            st.error("Vui lòng upload ảnh Style Reference.")
        elif not selected_actions:
            st.error("Vui lòng chọn ít nhất 1 hành động.")
        else:
            # Prepare Data
            try:
                style_ref = Image.open(style_ref_file).convert("RGB")
                
                # Configure output path
                base_path = Path(os.getcwd()) / "src" / "configs" / "characters"
                char_folder = base_path
                
                # Create folder if it doesn't exist
                output_dir = base_path / char_name
                os.makedirs(output_dir, exist_ok=True)

                data = AnimationPipelineInput(
                    style_ref=style_ref,
                    content_image=content_image,
                    system_prompt=PROMPT_IMAGE_STYLE_TRANSFER,
                    char_folder=char_folder,
                    char_name=char_name,
                    actions=selected_actions
                )

                with st.spinner("Đang xử lý Animation Pipeline (Có thể mất vài phút)..."):
                    # Run the pipeline
                    animation_generation_pipeline = pipeline
                    animation_generation_pipeline.run(data)
                
                st.success(f"Hoàn thành! Kết quả lưu tại: {output_dir}")
                
                # --- DISPLAY GIFS IN GRID ---
                # Find all generated gif files
                gif_files = glob.glob(str(output_dir / "*.gif")) + glob.glob(str(output_dir / "**" / "*.gif"))
                
                if gif_files:
                    st.markdown("### Các Animation đã tạo:")
                    
                    # Create 3 columns for the grid
                    cols = st.columns(3)
                    
                    for index, gif_file in enumerate(gif_files):
                        file_name = os.path.basename(gif_file)
                        
                        # Calculate which column this image belongs to (0, 1, or 2)
                        col_index = index % 3
                        
                        # Display image in the calculated column
                        with cols[col_index]:
                            st.image(gif_file, caption=file_name, use_container_width=True)
                else:
                    st.warning("Pipeline chạy xong nhưng không tìm thấy file GIF nào. Vui lòng kiểm tra log.")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                import traceback
                st.code(traceback.format_exc())
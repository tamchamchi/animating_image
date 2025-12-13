import streamlit as st
import os
import glob
import numpy as np
from PIL import Image
from src.utils.prompt import PROMPT_SUBJECT_GENERATION
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BODY_FOLDER_PATH = os.getenv("BODY_FOLDER_PATH")


def show(image_generator, face_segmenter):
    st.header("Step 1: Create Character")
    
    col1, col2 = st.columns([1, 1])
    
    # -------------------- LEFT COLUMN --------------------
    with col1:
        st.session_state.char_name = st.text_input(
            "Character Name:",
            value=st.session_state.char_name
        )
        
        mode = st.radio(
            "Input Mode:",
            ("Text Description", "Upload Image", "Merge Face & Body")
        )
        
        # -------------------------------------------------
        # MODE 1 — TEXT PROMPT GENERATION
        # -------------------------------------------------
        if mode == "Text Description":
            prompt = st.text_area(
                "Description:",
                "a child with blue glass wearing a red hoodie"
            )
            if st.button("Generate"):
                with st.spinner("Generating..."):
                    try:
                        full_prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)
                        st.session_state.char_image = image_generator.generate(prompt=full_prompt)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # -------------------------------------------------
        # MODE 2 — DIRECT IMAGE UPLOAD
        # -------------------------------------------------
        elif mode == "Upload Image":
            file = st.file_uploader("Upload Character", type=["png", "jpg"])
            if file:
                st.session_state.char_image = Image.open(file).convert("RGB")

        # -------------------------------------------------
        # MODE 3 — MERGE FACE & BODY
        # -------------------------------------------------
        elif mode == "Merge Face & Body":
            st.markdown("### 1. Face & Body Setup")

            # Face: Upload or Camera
            face_source = st.radio(
                "Face Input Method:",
                ("Upload Image", "Use Camera")
            )

            face_file = None

            if face_source == "Upload Image":
                face_file = st.file_uploader("Upload Face Image", type=["png", "jpg", "jpeg"])

            else:
                cam_img = st.camera_input("Take a photo")
                if cam_img:
                    face_file = cam_img

            # Body templates
            if os.path.exists(BODY_FOLDER_PATH):
                body_files = sorted(
                    glob.glob(os.path.join(BODY_FOLDER_PATH, "*.png"))
                    + glob.glob(os.path.join(BODY_FOLDER_PATH, "*.jpg"))
                )
                body_names = [os.path.basename(p) for p in body_files]
                selected_body_name = st.selectbox("Select Body Template:", body_names)

                selected_body_path = (
                    os.path.join(BODY_FOLDER_PATH, selected_body_name)
                    if selected_body_name else None
                )
            else:
                st.error(f"Directory not found: {BODY_FOLDER_PATH}")
                selected_body_path = None

            # Merge pipeline
            if face_file and selected_body_path:
                try:
                    face_pil = Image.open(face_file).convert("RGB")
                    body_pil = Image.open(selected_body_path).convert("RGB")

                    face_np = np.array(face_pil)
                    body_np = np.array(body_pil)

                    # Cache segmentation
                    file_id = getattr(face_file, 'file_id', face_file.name)

                    if st.session_state.get("last_face_id") != file_id:
                        with st.spinner("Segmenting Face..."):
                            _, _, _, face_crop = face_segmenter.segment(face_np)
                            st.session_state.current_face_crop = face_crop
                            st.session_state.last_face_id = file_id

                    face_crop = st.session_state.current_face_crop

                    # Anchor adjustment
                    st.markdown("### 2. Adjust Position")
                    colx, coly = st.columns(2)

                    with colx:
                        anchor_x = st.number_input("Anchor X", value=365, step=5)
                    with coly:
                        anchor_y = st.number_input("Anchor Y", value=-10, step=5)

                    # Apply to body
                    merged_result = face_segmenter.apply_to_body(
                        body_np,
                        face_crop,
                        anchor_point=(int(anchor_x), int(anchor_y))
                    )

                    preview = Image.fromarray(merged_result)
                    st.session_state.char_image = preview

                except Exception as e:
                    st.error(f"Error merging images: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # -------------------- RIGHT COLUMN --------------------
    with col2:
        st.subheader("Preview Result")

        if st.session_state.char_image:
            st.image(st.session_state.char_image, caption="Final Character", use_container_width=True)

            st.markdown("---")
            if st.button("Next Step ➡️", type="primary"):
                if not st.session_state.char_name:
                    st.warning("Please enter a character name first.")
                else:
                    st.session_state.step = 2
                    st.rerun()
        else:
            st.info("Character preview will appear here.")

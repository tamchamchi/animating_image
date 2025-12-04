import streamlit as st
import os
import glob
import numpy as np
from PIL import Image
from src.utils.prompt import PROMPT_SUBJECT_GENERATION
from dotenv import load_dotenv

# Load environment variables (e.g., path to body assets)
load_dotenv()

BODY_FOLDER_PATH = os.getenv("BODY_FOLDER_PATH")

def show(image_generator, face_segmenter):
    st.header("Step 1: Create Character")
    
    col1, col2 = st.columns([1, 1])
    
    # --- LEFT COLUMN: INPUT CONFIGURATION ---
    with col1:
        st.session_state.char_name = st.text_input("Character Name:", value=st.session_state.char_name)
        
        # Select Input Mode: Text, Direct Upload, or Face-Body Merge
        mode = st.radio("Input Mode:", ("Text Description", "Upload Image", "Merge Face & Body"))
        
        # --- MODE 1: TEXT DESCRIPTION ---
        if mode == "Text Description":
            prompt = st.text_area("Description:", "a child with blue glass wearing a red hoodie")
            if st.button("Generate"):
                with st.spinner("Generating..."):
                    try:
                        # Format prompt and generate image using the AI model
                        full_prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)
                        st.session_state.char_image = image_generator.generate(prompt=full_prompt)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # --- MODE 2: UPLOAD IMAGE ---                
        elif mode == "Upload Image":
            file = st.file_uploader("Upload Character", type=["png", "jpg"])
            if file:
                # Load user-uploaded image directly
                st.session_state.char_image = Image.open(file).convert("RGB")

        # --- MODE 3: MERGE FACE & BODY ---
        elif mode == "Merge Face & Body":
            st.markdown("### 1. Face & Body Setup")
            
            # A. Upload Face Image
            face_file = st.file_uploader("Upload Face Image", type=["png", "jpg", "jpeg"])
            
            # B. Select Body Template from Folder
            if os.path.exists(BODY_FOLDER_PATH):
                # List all PNG/JPG files in the body directory
                body_files = sorted(glob.glob(os.path.join(BODY_FOLDER_PATH, "*.png")) + 
                                    glob.glob(os.path.join(BODY_FOLDER_PATH, "*.jpg")))
                
                # Extract filenames for the dropdown list
                body_names = [os.path.basename(p) for p in body_files]
                
                selected_body_name = st.selectbox("Select Body Template:", body_names)
                
                # Reconstruct full path for the selected file
                if selected_body_name:
                    selected_body_path = os.path.join(BODY_FOLDER_PATH, selected_body_name)
            else:
                st.error(f"Directory not found: {BODY_FOLDER_PATH}")
                selected_body_path = None

            # C. Processing Logic (Merge)
            if face_file and selected_body_path:
                try:
                    # 1. Load Images as PIL -> Numpy
                    face_pil = Image.open(face_file).convert("RGB")
                    body_pil = Image.open(selected_body_path).convert("RGB")
                    
                    face_img_np = np.array(face_pil)
                    body_img_np = np.array(body_pil)
                    
                    # 2. Segment Face (With Caching)
                    # We use session_state to cache the face crop. 
                    # Only re-run segmentation if the uploaded file changes.
                    file_id = face_file.file_id if hasattr(face_file, 'file_id') else face_file.name
                    
                    if st.session_state.get('last_face_id') != file_id:
                        with st.spinner("Segmenting Face..."):
                            # Run the segmentation model to get the cropped face
                            _, _, _, face_crop = face_segmenter.segment(face_img_np)
                            st.session_state.current_face_crop = face_crop
                            st.session_state.last_face_id = file_id
                    
                    # Retrieve cropped face from state
                    face_crop = st.session_state.current_face_crop

                    # 3. Adjust Anchor Point
                    st.markdown("### 2. Adjust Position")
                    st.info("Adjust X/Y to attach the face to the body.")
                    
                    c_x, c_y = st.columns(2)
                    with c_x:
                        anchor_x = st.number_input("Anchor X", value=365, step=5)
                    with c_y:
                        anchor_y = st.number_input("Anchor Y", value=-10, step=5)

                    # 4. Apply Face to Body
                    # Combine the body image and face crop at the specified anchor point
                    merged_result = face_segmenter.apply_to_body(
                        body_img_np, 
                        face_crop, 
                        anchor_point=(int(anchor_x), int(anchor_y))
                    )
                    
                    # Convert result back to PIL for display and saving
                    preview_image = Image.fromarray(merged_result)
                    
                    # Update global character image in session state
                    st.session_state.char_image = preview_image
                    
                except Exception as e:
                    st.error(f"Error merging images: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # --- RIGHT COLUMN: PREVIEW & NAVIGATION ---
    with col2:
        st.subheader("Preview Result")
        if st.session_state.char_image:
            # Show the final generated/uploaded/merged image
            st.image(st.session_state.char_image, caption="Final Character", use_container_width=True)
            
            st.markdown("---")
            # Button to proceed to Animation Generation (Step 2)
            if st.button("Next Step ➡️", type="primary"):
                if not st.session_state.char_name:
                    st.warning("Please enter a character name first.")
                else:
                    st.session_state.step = 2
                    st.rerun()
        else:
            st.info("Character preview will appear here.")
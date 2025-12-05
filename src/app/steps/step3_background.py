import streamlit as st
import os
import cv2
import json
import numpy as np
from pathlib import Path
from src.concept_decomposer.object_decomposer import draw_results


def show(object_decomposer):
    st.header("Step 3: Background Analysis")

    # 1. Check if character name exists (from Step 1)
    if not st.session_state.get('char_name'):
        st.warning("Please define a Character Name in Step 1 first.")
        if st.button("⬅️ Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
        return

    char_name = st.session_state.char_name

    # Define storage directory: src/configs/characters/{char_name}/background
    base_path = Path(os.getcwd()) / "src" / "configs" / \
        "characters" / char_name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Background")
        bg_file = st.file_uploader("Upload Background Image", type=[
                                   "jpg", "jpeg", "png"])

        # Configure Targets (User can edit or keep default)
        st.subheader("Detection Targets")
        # Default targets from your sample code
        default_targets = json.dumps({
            "ac_controller": "thermostat . electronic device with screen . digital wall controller",
            "notice_board": "framed text . framed certificate"
        }, indent=2)

        targets_str = st.text_area(
            "Targets (JSON Format):", value=default_targets, height=150)
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.40)

        run_btn = st.button("🔍 Detect Objects", type="primary")

    with col2:
        st.subheader("2. Visualization")
        # Placeholder to display result image
        result_placeholder = st.empty()

    # --- PROCESSING LOGIC ---
    if bg_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(bg_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)  # BGR format

        # Display original image if not run yet
        if not run_btn:
            # Convert BGR to RGB for Streamlit display
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                     caption="Original Background", use_container_width=True)

        if run_btn:
            try:
                # 1. Create output directory
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                # Save original image to folder
                original_img_path = base_path / "background.jpg"
                cv2.imwrite(str(original_img_path), input_image)

                # 2. Parse Targets from text area
                targets = json.loads(targets_str)

                final_output = []
                vis_data = []

                with st.spinner("Detecting objects..."):
                    # 3. Detect loop
                    for output_name, model_prompt in targets.items():
                        # Call detect function (with prompt list containing 1 element)
                        detections = object_decomposer.detect_objects(
                            input_image,
                            [model_prompt],
                            threshold=confidence_threshold
                        )

                        for det in detections:
                            item = {
                                "name": output_name,
                                "bbox": det['bbox'],
                                "score": det['score'],
                                "polygon": det["polygon"]
                            }
                            final_output.append(item)

                            # Data for drawing
                            vis_data.append(item)

                # 4. Save JSON result
                json_path = base_path / "detected_objects.json"
                with open(json_path, 'w') as f:
                    json.dump(final_output, f, indent=2)

                # 5. Draw and Save Visualized image
                if vis_data:
                    vis_image = draw_results(input_image.copy(), vis_data)
                    vis_img_path = base_path / "result_visualized.jpg"
                    cv2.imwrite(str(vis_img_path), vis_image)

                    # Display on UI (Convert BGR -> RGB)
                    result_placeholder.image(
                        cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB),
                        caption="Detected Objects",
                        use_container_width=True
                    )
                    st.success(f"Saved results to: {base_path}")

                    # Display JSON on screen
                    # st.markdown("### Detected JSON Data")
                    # st.json(final_output)
                else:
                    st.warning(
                        "No objects detected with current prompts/threshold.")
                    result_placeholder.image(
                        cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                        caption="No detections",
                        use_container_width=True
                    )

            except json.JSONDecodeError:
                st.error("Error: Invalid JSON format in Targets field.")
            except Exception as e:
                st.error(f"Processing Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("Next Step: Export ➡️"):
            st.session_state.step = 4
            st.rerun()
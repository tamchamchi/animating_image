import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.concept_decomposer.object_decomposer import draw_results
from src.utils.svg_file import (
    extract_topk_polygons,
    get_svg_size,
    polygons_to_json_object,
    restore_polygon_to_image_coords,
)


def show(object_decomposer, svg_converter):
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

        # --- ANALYSIS MODE ---
        st.subheader("Choose Analysis Mode")
        analysis_mode = st.radio(
            "Select method:",
            ["Deep Learning Model", "SVG Polygon Analysis"],
            index=0,
        )

        # --- TARGETS for Deep Learning ---
        default_targets = json.dumps({
            "ac_controller": "thermostat . electronic device with screen . digital wall controller",
            "notice_board": "framed text . framed certificate"
        }, indent=2)

        if analysis_mode == "Deep Learning Model":
            st.subheader("Detection Targets")
            targets_str = st.text_area(
                "Targets (JSON Format):", value=default_targets, height=150)
            confidence_threshold = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.40)
        else:
            st.subheader("Adjust Selected Area")
            topk = st.slider(
                "Top K", 0, 100, 30)

        run_btn = st.button("▶️ Run Analysis", type="primary")

    with col2:
        st.subheader("2. Visualization")
        result_placeholder = st.empty()

    # --- PROCESSING LOGIC ---
    if bg_file is not None:
        file_bytes = np.asarray(bytearray(bg_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

        if not run_btn:
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                     caption="Original Background",
                     use_container_width=True)

        if run_btn:
            try:
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                # Save uploaded background
                original_img_path = base_path / "background.jpg"
                cv2.imwrite(str(original_img_path), input_image)

                final_output = []
                vis_data = []

                # -------------------------------
                # OPTION 1: Deep Learning Model
                # -------------------------------
                if analysis_mode == "Deep Learning Model":
                    st.info("🔍 Running Deep Learning detection...")

                    targets = json.loads(targets_str)

                    with st.spinner("Detecting objects..."):
                        for output_name, model_prompt in targets.items():
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
                                vis_data.append(item)

                # -------------------------------
                # OPTION 2: SVG Polygon Analysis
                # -------------------------------
                else:
                    st.info("🟦 Running SVG background extraction...")

                    with st.spinner("Converting background to SVG..."):
                        if isinstance(input_image, np.ndarray):
                            pil_bg = Image.fromarray(cv2.cvtColor(
                                input_image, cv2.COLOR_BGR2RGB))
                        else:
                            pil_bg = input_image

                        svg_bg = svg_converter.convert(
                            [pil_bg], limit=20000)
                        svg_path = base_path / "background.svg"
                        svg_path.write_text(svg_bg)

                        # Extract polygons (top 30)
                        polys = extract_topk_polygons(svg_path, top_k=topk)

                        # Map polygons back to image coordinates
                        svg_w, svg_h = get_svg_size(svg_path)
                        bg_h, bg_w = input_image.shape[:2]
                        restored = [
                            restore_polygon_to_image_coords(
                                poly, svg_w, svg_h, bg_w, bg_h)
                            for poly in polys
                        ]

                        json_objects = polygons_to_json_object(restored[1:])
                        final_output = json_objects  # replace output with SVG objects

                        # Visualize polygons
                        vis_image = input_image.copy()
                        for obj in restored[1:]:
                            pts = np.array(obj, np.int32)
                            cv2.polylines(vis_image, [pts], isClosed=True, color=(
                                0, 255, 0), thickness=2)
                        vis_data = restored[1:]

                # ------------------------------------------------
                # SAVE FINAL OUTPUT JSON
                # ------------------------------------------------
                json_path = base_path / "detected_objects.json"
                with open(json_path, 'w') as f:
                    json.dump(final_output, f, indent=2)

                # ------------------------------------------------
                # VISUALIZATION
                # ------------------------------------------------
                if analysis_mode == "Deep Learning Model":
                    vis_image = draw_results(input_image.copy(), vis_data)
                else:
                    # Already drawn above
                    pass

                vis_img_path = base_path / "result_visualized.jpg"
                cv2.imwrite(str(vis_img_path), vis_image)

                result_placeholder.image(
                    cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB),
                    caption="Analysis Result",
                    use_container_width=True
                )

                st.success(f"Saved results to: {base_path}")

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

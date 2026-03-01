import streamlit as st
import os
import glob
import time
from pathlib import Path
from PIL import Image
from src.pipeline.input_data import AnimationPipelineInput
from src.utils.prompt import PROMPT_IMAGE_STYLE_TRANSFER

def update_gallery(output_dir_path, container):
    """Helper function to redraw the GIF image grid into the container."""
    # Find all gif files
    gifs = glob.glob(str(output_dir_path / "*.gif")) + glob.glob(str(output_dir_path / "**" / "*.gif"))
    
    # Sort files so the newest ones appear first
    gifs.sort(key=os.path.getmtime, reverse=True) 

    if gifs:
        with container.container():
            st.subheader(f"Gallery ({len(gifs)} animations)")
            # --- GRID 3 COLUMNS ---
            cols = st.columns(3)
            for i, gif in enumerate(gifs):
                file_name = os.path.basename(gif)
                # Format filename for better display
                action_name = file_name.replace(".gif", "").replace("_", " ").title()
                
                with cols[i % 3]:
                    st.image(gif, caption=action_name, use_container_width=True)

def show(pipeline):
    st.header("Step 2: Generate Animation")
    
    # Validate Step 1 completion
    if not st.session_state.char_image:
        st.warning("Please create a character in Step 1.")
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()
        return

    col_conf, col_view = st.columns([1, 2])

    # --- CONFIGURATION AREA (Left Column) ---
    with col_conf:
        st.image(st.session_state.char_image, width=150, caption="Character Input")
        st.markdown("---")

        # 1. Style Transfer Option
        st.subheader("Configuration")
        use_style_transfer = st.checkbox("🎨 Apply Style Transfer", value=False)
        
        style_ref_image = None
        style_file = None
        
        if use_style_transfer:
            style_file = st.file_uploader("Upload Style Ref (Required)", type=["png", "jpg", "jpeg"])
            if style_file:
                style_ref_image = Image.open(style_file).convert("RGB")
                st.image(style_ref_image, caption="Style Reference", width=150)
        else:
            st.info("Using original character colors.")
            style_ref_image = st.session_state.char_image

        # 2. Select Actions
        available_actions = ["standing", "jumping", "running", "jesse_dancing", "waving", "speaking"]
        selected_actions = st.multiselect(
            "Actions:", 
            available_actions, 
            default=["waving"]
        )

        # Overwrite Option
        force_regenerate = st.checkbox("Force Regenerate (Overwrite old files)", value=False, help="If selected, the system will regenerate the GIF even if the file already exists.")
        
        st.markdown("---")
        
        # 3. Run Button
        run_btn = st.button("🎬 Run Pipeline", type="primary")

    # --- DISPLAY AREA (Right Column) ---
    with col_view:
        gallery_placeholder = st.empty()
        
        # Define output path beforehand to check for files
        base_path = Path(os.getcwd()) / "src" / "configs" / "characters"
        output_dir = base_path / st.session_state.char_name

        has_results = False
        
        # If output from previous run exists, display immediately
        if output_dir.exists():
            update_gallery(output_dir, gallery_placeholder)
            if list(output_dir.glob("*.gif")) or list(output_dir.glob("**/*.gif")):
                has_results = True

        if run_btn:
            # --- VALIDATION ---
            if not selected_actions:
                st.error("Please select at least one action.")
                return
            elif use_style_transfer and not style_file:
                st.error("Please upload a Style Reference image.")
                return
            
            # --- SETUP PATHS ---
            try:
                os.makedirs(output_dir, exist_ok=True)
                st.session_state.output_dir = str(output_dir)
                
                # --- PROGRESS BAR ---
                progress_text = "Starting generation..."
                my_bar = st.progress(0, text=progress_text)
                total_actions = len(selected_actions)

                # --- LOOP PROCESSING EACH ACTION ---
                for idx, action in enumerate(selected_actions):
                    
                    # Update progress bar
                    current_progress = int((idx / total_actions) * 100)
                    my_bar.progress(current_progress, text=f"Checking: {action} ({idx+1}/{total_actions})...")
                    
                    # --- SKIP LOGIC ---
                    # Assume pipeline saves file as: {char_name}/{action}.gif
                    # You need to adjust the filename if the pipeline saves differently (e.g., {action}_output.gif)
                    target_file = output_dir / f"{action}.gif"
                    
                    if target_file.exists() and not force_regenerate:
                        # File already exists and Force is not selected -> Skip
                        time.sleep(0.5) # Pause briefly so the user can read the notification
                        st.toast(f"Skipped '{action}' - File already exists.", icon="⏭️")
                        continue 

                    # --- RUN PIPELINE ---
                    my_bar.progress(current_progress, text=f"Generating: {action}...")
                    
                    data = AnimationPipelineInput(
                        style_ref=style_ref_image, 
                        content_image=st.session_state.char_image,
                        system_prompt=PROMPT_IMAGE_STYLE_TRANSFER,
                        char_folder=base_path,
                        char_name=st.session_state.char_name,
                        actions=[action] 
                    )
                    
                    pipeline.run(data, use_style_transfer=use_style_transfer)
                    
                    # Update Gallery immediately after generation
                    update_gallery(output_dir, gallery_placeholder)
                
                # Finalize
                my_bar.progress(100, text="All tasks finished!")
                time.sleep(1)
                my_bar.empty()
                st.success("Process Completed!")

                has_results = True 
                
                # Show Next Step button

            except Exception as e:
                st.error(f"Error during processing: {e}")
                import traceback
                st.code(traceback.format_exc())
                
        if has_results:
            st.markdown("---")
            if st.button("Next Step ➡️", key="btn_go_step3", type="primary"):
                st.session_state.step = 3
                st.rerun()
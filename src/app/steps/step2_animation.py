import streamlit as st
import os
import glob
import time
from pathlib import Path
from PIL import Image
from src.pipeline.input_data import AnimationPipelineInput
from src.utils.prompt import PROMPT_IMAGE_STYLE_TRANSFER

def update_gallery(output_dir_path, container):
    """Hàm helper để vẽ lại lưới ảnh GIF vào container."""
    # Tìm tất cả file gif
    gifs = glob.glob(str(output_dir_path / "*.gif")) + glob.glob(str(output_dir_path / "**" / "*.gif"))
    
    # Sắp xếp để file mới nhất hiện lên đầu
    gifs.sort(key=os.path.getmtime, reverse=True) 

    if gifs:
        with container.container():
            st.subheader(f"Gallery ({len(gifs)} animations)")
            # --- GRID 3 COLUMNS ---
            cols = st.columns(3)
            for i, gif in enumerate(gifs):
                file_name = os.path.basename(gif)
                # Format tên cho đẹp
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

    # --- KHU VỰC CẤU HÌNH (Cột trái) ---
    with col_conf:
        st.image(st.session_state.char_image, width=150, caption="Character Input")
        st.markdown("---")

        # 1. Tùy chọn Style Transfer
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

        # 2. Chọn hành động
        available_actions = ["standing", "jumping", "running", "jesse_dancing", "waving", "speaking"]
        selected_actions = st.multiselect(
            "Actions:", 
            available_actions, 
            default=["waving"]
        )

        # Tùy chọn ghi đè
        force_regenerate = st.checkbox("Force Regenerate (Ghi đè file cũ)", value=False, help="Nếu chọn, hệ thống sẽ tạo lại GIF kể cả khi file đã tồn tại.")
        
        st.markdown("---")
        
        # 3. Nút chạy
        run_btn = st.button("🎬 Run Pipeline", type="primary")

    # --- KHU VỰC HIỂN THỊ (Cột phải) ---
    with col_view:
        gallery_placeholder = st.empty()
        
        # Xác định đường dẫn output trước để check file
        base_path = Path(os.getcwd()) / "src" / "configs" / "characters"
        output_dir = base_path / st.session_state.char_name
        
        # Nếu đã có output từ lần trước, hiển thị ngay
        if output_dir.exists():
            update_gallery(output_dir, gallery_placeholder)

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

                # --- VÒNG LẶP XỬ LÝ TỪNG ACTION ---
                for idx, action in enumerate(selected_actions):
                    
                    # Cập nhật thanh tiến trình
                    current_progress = int((idx / total_actions) * 100)
                    my_bar.progress(current_progress, text=f"Checking: {action} ({idx+1}/{total_actions})...")
                    
                    # --- LOGIC SKIP (BỎ QUA) ---
                    # Giả định pipeline lưu file dạng: {tên_nhân_vật}/{action}.gif
                    # Bạn cần điều chỉnh tên file nếu pipeline lưu khác (ví dụ: {action}_output.gif)
                    target_file = output_dir / f"{action}.gif"
                    
                    if target_file.exists() and not force_regenerate:
                        # File đã tồn tại và không chọn Force -> Bỏ qua
                        time.sleep(0.5) # Dừng một chút để user kịp đọc thông báo
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
                    
                    # Update Gallery ngay sau khi tạo xong
                    update_gallery(output_dir, gallery_placeholder)
                
                # Hoàn tất
                my_bar.progress(100, text="All tasks finished!")
                time.sleep(1)
                my_bar.empty()
                st.success("Process Completed!")
                
                # Hiện nút Next Step
                if st.button("Next Step ➡️", key="next_step_btn"):
                    st.session_state.step = 3
                    st.rerun()

            except Exception as e:
                st.error(f"Error during processing: {e}")
                import traceback
                st.code(traceback.format_exc())
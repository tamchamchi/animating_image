import streamlit as st
from PIL import Image
from src.utils.prompt import PROMPT_SUBJECT_GENERATION

def show(image_generator):
    st.header("Step 1: Create Character")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.session_state.char_name = st.text_input("Character Name:", value=st.session_state.char_name)
        mode = st.radio("Input Mode:", ("Text Description", "Upload Image"))
        
        if mode == "Text Description":
            prompt = st.text_area("Description:", "a child with blue glass wearing a red hoodie")
            if st.button("Generate"):
                with st.spinner("Generating..."):
                    try:
                        full_prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)
                        st.session_state.char_image = image_generator.generate(prompt=full_prompt)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
        elif mode == "Upload Image":
            file = st.file_uploader("Upload", type=["png", "jpg"])
            if file:
                st.session_state.char_image = Image.open(file).convert("RGB")

    with col2:
        if st.session_state.char_image:
            st.image(st.session_state.char_image, caption="Preview", width=300)
            if st.button("Next Step ➡️", type="primary"):
                st.session_state.step = 2
                st.rerun()
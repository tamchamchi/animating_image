import streamlit as st
import os
from pathlib import Path

# Import Game class from your module
try:
    from src.render import Game
except ImportError:
    # Fallback to avoid errors if path is not configured correctly, but app.py handles this
    pass

def show():
    st.header("Step 4: Interactive Playground")
    st.caption("Launch the interactive environment using the assets generated in previous steps.")

    # 1. Check input information
    if not st.session_state.get('char_name'):
        st.warning("⚠️ No character selected. Please start from Step 1.")
        if st.button("⬅️ Go to Step 1"):
            st.session_state.step = 1
            st.rerun()
        return

    char_name = st.session_state.char_name
    
    # 2. Construct data path
    # This path points to: src/configs/characters/{char_name}
    base_path = Path(os.getcwd()) / "src" / "configs" / "characters" / char_name
    
    # Check if folder exists
    if not base_path.exists():
        st.error(f"❌ Data directory not found: {base_path}")
        return

    # 3. Info display interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Asset Configuration")
        st.info(f"**Character:** {char_name}")
        st.code(str(base_path), language="bash")
        
        # Quick check for important files
        has_gifs = list(base_path.glob("*.gif")) or list(base_path.glob("**/*.gif"))
        has_bg = (base_path / "detected_objects.json").exists()
        
        st.write("Checking assets:")
        st.checkbox("Animations (GIFs)", value=bool(has_gifs), disabled=True)
        st.checkbox("Background Analysis", value=bool(has_bg), disabled=True)

    with col2:
        st.subheader("🎮 Launch Game")
        st.markdown(
            """
            Click the button below to open the **Interactive Window**.
            
            *Note: This will open a separate window on the host machine. 
            The web interface will pause until you close the game window.*
            """
        )
        
        # Launch button
        if st.button("🚀 Run Game Engine", type="primary"):
            if not bool(has_gifs):
                st.error("Please generate animations in Step 2 first.")
            else:
                try:
                    with st.spinner(f"Launching game for '{char_name}'... Check your taskbar!"):
                        # --- YOUR LOGIC ---
                        # Convert path to string because legacy libraries often use strings
                        game_data_path = str(base_path)
                        
                        # Initialize Game
                        gamer = Game(data_path=game_data_path)
                        
                        # Run Game (This function will block the thread until the game closes)
                        gamer.run()
                        
                    st.success("Game session ended.")
                except Exception as e:
                    st.error(f"Failed to launch game: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # --- Footer Navigation ---
    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🔄 Restart"):
            # Reset session to start a new character
            st.session_state.step = 1
            st.session_state.char_name = ""
            st.session_state.char_image = None
            st.rerun()
    with c2:
        if st.button("⬅️ Back to Step 3"):
            st.session_state.step = 3
            st.rerun()
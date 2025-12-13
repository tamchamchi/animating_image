import subprocess
from pyngrok import ngrok

# --- Start Streamlit app ---
port = 8501
streamlit_cmd = f"streamlit run ./src/app/app.py --server.port {port}"
subprocess.Popen(streamlit_cmd.split())

# --- Create public URL with ngrok ---
public_url = ngrok.connect(port).public_url
print("🌍 Public URL:", public_url)

# Keep process alive
input("Press ENTER to stop...\n")

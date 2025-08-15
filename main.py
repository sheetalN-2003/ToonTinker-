import streamlit as st
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
from PIL import Image
import numpy as np
import time
from io import BytesIO
from transformers import pipeline
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set up the app
st.set_page_config(layout="wide", page_title="✨ AI Comic Generator Pro")
st.title("✨ AI Comic Generator Pro")
st.subheader("Optimized for Streamlit Deployment")

# Initialize session state
if 'comic_data' not in st.session_state:
    st.session_state.comic_data = {
        'panels': [],
        'characters': {},
        'style': "Comic Book"
    }

# --- OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        # Use smaller base model
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lite version of ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        panel_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            controlnet=controlnet,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Smaller story analyzer
        story_analyzer = pipeline("text-generation", model="gpt2")
        
        return {
            "base": base_pipe,
            "panel": panel_pipe,
            "story": story_analyzer
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# --- SIMPLIFIED VOICE INPUT ---
class VoiceProcessor:
    def __init__(self):
        self.audio_buffer = queue.Queue()
    
    def recv(self, frame):
        self.audio_buffer.put(frame)
        return frame

if st.sidebar.checkbox("Enable Voice Input"):
    webrtc_ctx = webrtc_streamer(
        key="voice-input",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"audio": True},
        audio_frame_callback=VoiceProcessor().recv,
    )

# --- MAIN UI ---
tab1, tab2 = st.tabs(["📖 Create Comic", "⚙️ Settings"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        story_input = st.text_area("Enter your story:", height=200,
                                 value="A superhero battles robots in futuristic city")
        
        if st.button("Generate Comic", type="primary"):
            models = load_models()
            if models:
                with st.spinner("Creating your comic..."):
                    try:
                        # Generate consistent character
                        char_prompt = "comic book character, full body, dynamic pose"
                        character = models["base"](
                            char_prompt,
                            num_inference_steps=30,
                            width=512,
                            height=768
                        ).images[0]
                        
                        # Generate panels
                        panels = []
                        for i in range(2):  # Limit to 2 panels for demo
                            panel = models["panel"](
                                f"Comic panel {i+1}: {story_input}",
                                num_inference_steps=20,
                                width=512,
                                height=512
                            ).images[0]
                            panels.append(panel)
                        
                        # Display results
                        st.session_state.comic_data['panels'] = panels
                        st.success("Comic generated!")
                        
                    except torch.cuda.OutOfMemoryError:
                        st.error("GPU memory full! Try smaller images or fewer panels")
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
    
    with col2:
        if st.session_state.comic_data.get('panels'):
            st.subheader("Your Comic")
            for i, panel in enumerate(st.session_state.comic_data['panels']):
                st.image(panel, caption=f"Panel {i+1}", use_column_width=True)

with tab2:
    st.header("Settings")
    art_style = st.selectbox("Art Style", ["Comic Book", "Manga", "Noir"])
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")

# --- LITE VERSION NOTES ---
st.sidebar.markdown("""
### Lite Version Features:
- Optimized for Streamlit Cloud
- 2 panel maximum
- Smaller AI models
- Basic voice input
- CPU fallback support

For full features, run locally with GPU.
""")

if __name__ == "__main__":
    # Hide some Streamlit elements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

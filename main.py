import streamlit as st
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import time
import hashlib
import json
import os
from pathlib import Path
from transformers import pipeline
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- AUTHENTICATION SYSTEM ---
USER_DB = "users.json"
ADMIN_USER = "admin@comic.com"  # Predefined admin email

def init_auth():
    if not Path(USER_DB).exists():
        with open(USER_DB, "w") as f:
            json.dump({
                ADMIN_USER: {
                    "password": hashlib.sha256("admin123".encode()).hexdigest(),
                    "role": "admin",
                    "comics": []
                }
            }, f)

def authenticate(email, password):
    try:
        with open(USER_DB) as f:
            users = json.load(f)
        if email in users and users[email]["password"] == hashlib.sha256(password.encode()).hexdigest():
            return users[email]["role"]
    except:
        pass
    return None

def register_user(email, password, role="user"):
    try:
        with open(USER_DB) as f:
            users = json.load(f)
        if email in users:
            return False
        users[email] = {
            "password": hashlib.sha256(password.encode()).hexdigest(),
            "role": role,
            "comics": []
        }
        with open(USER_DB, "w") as f:
            json.dump(users, f)
        return True
    except:
        return False

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="✨ ComicGen Pro")
st.title("✨ ComicGen Pro - Next-Gen AI Comic Creator")

# Initialize systems
init_auth()
if 'comic_data' not in st.session_state:
    st.session_state.comic_data = {
        'panels': [],
        'characters': {},
        'style': "Comic Book"
    }

# --- AUTHENTICATION UI ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

if not st.session_state.authenticated:
    auth_tab, reg_tab = st.tabs(["Login", "Register"])

    with auth_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            role = authenticate(email, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.email = email
                st.rerun()
            else:
                st.error("Invalid credentials")

    with reg_tab:
        reg_email = st.text_input("Email", key="reg_email")
        reg_pwd = st.text_input("Password", type="password", key="reg_pwd")
        reg_pwd_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd2")
        if st.button("Create Account"):
            if reg_pwd != reg_pwd_confirm:
                st.error("Passwords don't match")
            elif register_user(reg_email, reg_pwd):
                st.success("Account created! Please login")
            else:
                st.error("Registration failed (user may exist)")
    
    st.stop()

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        common_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "safety_checker": None
        }

        # Base model
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            **common_kwargs
        ).to(device)
        
        # ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            **common_kwargs
        ).to(device)
        
        panel_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            controlnet=controlnet,
            **common_kwargs
        ).to(device)
        
        return {
            "base": base_pipe,
            "panel": panel_pipe
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# --- VOICE INPUT ---
class VoiceProcessor:
    def __init__(self):
        self.audio_buffer = queue.Queue()
    
    def recv(self, frame):
        self.audio_buffer.put(frame)
        return frame

# --- ADMIN DASHBOARD ---
if st.session_state.role == "admin":
    with st.expander("🔧 Admin Panel", expanded=False):
        st.subheader("User Management")
        with open(USER_DB) as f:
            users = json.load(f)
        
        user_emails = list(users.keys())
        selected_user = st.selectbox("Select User", user_emails)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete User"):
                if selected_user != ADMIN_USER:
                    del users[selected_user]
                    with open(USER_DB, "w") as f:
                        json.dump(users, f)
                    st.rerun()
        with col2:
            new_role = st.selectbox("Change Role", ["admin", "user"])
            if st.button("Update Role"):
                users[selected_user]["role"] = new_role
                with open(USER_DB, "w") as f:
                    json.dump(users, f)

# --- COMIC MANAGEMENT ---
def save_comic(comic_data):
    with open(USER_DB) as f:
        users = json.load(f)
    users[st.session_state.email]["comics"].append({
        "title": comic_data.get("title", "Untitled"),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "panels": len(comic_data["panels"]),
        "data": comic_data  # Save full comic data
    })
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def load_comic(comic_data):
    st.session_state.comic_data = comic_data

# --- MAIN UI ---
st.sidebar.header(f"Welcome, {st.session_state.email}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

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

tab1, tab2, tab3 = st.tabs(["🎨 Create Comic", "📚 My Comics", "⚙️ Settings"])

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
                        # Generate panels
                        panels = []
                        for i in range(2):
                            panel = models["panel"](
                                f"Comic panel {i+1}: {story_input}",
                                num_inference_steps=20,
                                width=512,
                                height=512
                            ).images[0]
                            panels.append(panel)
                        
                        st.session_state.comic_data['panels'] = panels
                        st.success("Comic generated!")
                        
                    except torch.cuda.OutOfMemoryError:
                        st.error("GPU memory full! Try smaller images")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if st.session_state.comic_data.get('panels'):
            st.subheader("Your Comic")
            for i, panel in enumerate(st.session_state.comic_data['panels']):
                st.image(panel, caption=f"Panel {i+1}", use_column_width=True)
            
            comic_title = st.text_input("Save as:", "My Awesome Comic")
            if st.button("Save Comic"):
                save_comic(st.session_state.comic_data)
                st.success("Comic saved!")

with tab2:
    with open(USER_DB) as f:
        user_data = json.load(f)[st.session_state.email]
    
    if not user_data["comics"]:
        st.info("No saved comics yet")
    else:
        for idx, comic in enumerate(user_data["comics"]):
            with st.expander(f"{comic['title']} ({comic['date']})"):
                st.write(f"Panels: {comic['panels']}")
                if st.button("Load Comic", key=f"load_{idx}"):
                    load_comic(comic["data"])
                    st.rerun()
                if st.button("Delete Comic", key=f"del_{idx}"):
                    user_data["comics"].pop(idx)
                    with open(USER_DB, "w") as f:
                        json.dump(user_data, f)
                    st.rerun()

with tab3:
    st.header("Settings")
    st.session_state.comic_data['style'] = st.selectbox(
        "Art Style", 
        ["Comic Book", "Manga", "Noir", "Watercolor", "Cyberpunk"],
        index=["Comic Book", "Manga", "Noir", "Watercolor", "Cyberpunk"].index(st.session_state.comic_data['style'])
    )
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")

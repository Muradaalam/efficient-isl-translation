import base64
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path


def display_video(video_bytes, mime="video/mp4"):
    encoded = base64.b64encode(video_bytes).decode("utf-8")

    components.html(
        f"""
        <video width="100%" height="100%" controls autoplay muted loop playsinline>
            <source src="data:{mime};base64,{encoded}" type="{mime}">
        </video>
        """,
        height=500
    )


st.title("Video Playback Test")

mp4 = Path("demo/outputs/sample_skeleton.mp4")
webm = Path("demo/outputs/sample_skeleton.webm")

col1, col2 = st.columns(2)

with col1:
    st.subheader("MP4 - H.264")
    display_video(mp4.read_bytes())

with col2:
    st.subheader("WebM - VP9")
    display_video(webm.read_bytes(), "video/webm")

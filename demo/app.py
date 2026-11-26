import time
import base64
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from pose_format import Pose

from demo.video_to_pose import convert_video_to_pose
from demo.visualization import visualize_pose


TARGET_FRAMES = 256
TARGET_LANDMARKS = 543


st.set_page_config(
    page_title="ISL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load CSS

def load_css():
    path = Path(__file__).parent / "style.css"
    st.html(f"<style>{path.read_text()}</style>")


# Video player

def display_video(video_bytes, mime="video/mp4", height=300):
    encoded = base64.b64encode(video_bytes).decode("utf-8")

    components.html(
        f"""
        <html>
        <body style="margin:0; padding:0;">
            <div style="display:flex;
                        justify-content:center;
                        align-items:center;
                        height:{height}px;">

                <video controls autoplay muted loop playsinline
                    style="max-width:100%;
                           max-height:100%;
                           width:auto;
                           height:auto;
                           object-fit:contain;
                           border-radius:8px;">

                    <source src="data:{mime};base64,{encoded}"
                            type="{mime}">

                </video>
            </div>
        </body>
        </html>
        """,
        height=height + 10
    )


# Section heading

def section_title(title):
    st.html(f'<div class="section-title">{title}</div>')


# Pipeline visualization

def show_pipeline(uploaded=False, processed=False):

    stages = [
        "Video Input",
        "Pose Extraction",
        "Adaptive Sampling",
        "Translation Model",
        "English Sentence"
    ]

    html = '<div class="pipeline">'

    for i, stage in enumerate(stages):

        completed = (i == 0 and uploaded) or (i == 1 and processed)

        background = "#DCFCE7" if completed else "#E2E8F0"
        color = "#15803D" if completed else "#64748B"
        number = "✓" if completed else str(i + 1)

        html += f"""
        <div class="pipeline-step">
            <div class="pipeline-number"
                style="background:{background}; color:{color};">
                {number}
            </div>
            {stage}
        </div>
        """

        if i < len(stages) - 1:
            html += '<div class="pipeline-arrow">→</div>'

    html += "</div>"

    st.html(html)


# Video processing

def process_video(video_bytes):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        video_path = temp_dir / "input.mp4"
        pose_path = temp_dir / "input.pose"
        skeleton_path = temp_dir / "skeleton.mp4"
        overlay_path = temp_dir / "overlay.mp4"

        video_path.write_bytes(video_bytes)

        start = time.perf_counter()

        with st.spinner("Extracting pose landmarks..."):
            convert_video_to_pose(video_path, pose_path)

        extraction_time = time.perf_counter() - start

        with st.spinner("Generating skeleton visualization..."):
            visualize_pose(pose_path, skeleton_path)

        with st.spinner("Generating landmark overlay..."):
            visualize_pose(pose_path, overlay_path, video_path)

        pose_bytes = pose_path.read_bytes()
        pose = Pose.read(pose_bytes)

        skeleton_bytes = skeleton_path.read_bytes()
        overlay_bytes = overlay_path.read_bytes()

        if not skeleton_bytes or not overlay_bytes:
            raise RuntimeError("Generated video is empty")

        return {
            "input": video_bytes,
            "pose": pose_bytes,
            "skeleton": skeleton_bytes,
            "overlay": overlay_bytes,
            "frames": pose.body.data.shape[0],
            "landmarks": pose.body.data.shape[2],
            "fps": float(pose.body.fps),
            "extraction_time": extraction_time,
            "total_time": time.perf_counter() - start
        }


load_css()


# ==================================================
# HEADER
# ==================================================

st.html("""
<div class="hero">

    <div class="hero-label">
        RESEARCH DEMONSTRATION
    </div>

    <div class="hero-title">
        Indian Sign Language Translator
    </div>

    <div class="hero-subtitle">
        Efficient Continuous ISL-to-English Translation
        using Adaptive Frame Sampling
    </div>

    <span class="tech-tag">MediaPipe Holistic</span>
    <span class="tech-tag">Adaptive Sampling</span>
    <span class="tech-tag">T5 + LoRA</span>

</div>
""")


# ==================================================
# VIDEO UPLOAD
# ==================================================

section_title("01 — Upload Sign Language Video")

upload_col, action_col = st.columns([3, 1])

with upload_col:

    uploaded_video = st.file_uploader(
        "Upload an MP4 video",
        type=["mp4"],
        label_visibility="collapsed"
    )

with action_col:

    st.caption("Adaptive Sampling")

    process_clicked = st.button(
        "Process Video",
        type="primary",
        use_container_width=True,
        disabled=uploaded_video is None
    )


if uploaded_video is None:

    st.session_state.pop("result", None)

    show_pipeline()

    st.info("Upload an ISL video to begin the demonstration.")

    st.stop()


video_bytes = uploaded_video.getvalue()


# ==================================================
# PROCESS VIDEO
# ==================================================

if process_clicked:

    st.session_state.pop("result", None)

    try:
        st.session_state.result = process_video(video_bytes)
        st.rerun()

    except Exception as e:
        st.exception(e)


result = st.session_state.get("result")

processed = (
    result is not None
    and result["input"] == video_bytes
)

show_pipeline(uploaded=True, processed=processed)


# ==================================================
# VIDEO AND POSE VISUALIZATION
# ==================================================

section_title("02 — Video and Pose Analysis")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("**Original Video**")

    display_video(video_bytes)

    st.caption(uploaded_video.name)


with col2:

    st.markdown("**Extracted Skeleton**")

    if processed:

        display_video(result["skeleton"])

        st.caption("Body, face and hand landmarks")

    else:

        st.info("Awaiting pose extraction")


with col3:

    st.markdown("**Landmark Overlay**")

    if processed:

        display_video(result["overlay"])

        st.caption("Landmarks mapped to original video")

    else:

        st.info("Awaiting pose extraction")


if processed:

    # ==================================================
    # PROCESSING STATISTICS
    # ==================================================

    section_title("03 — Processing Statistics")

    fps = result["fps"]
    frames = result["frames"]

    duration = frames / fps if fps > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Input Frames", frames)

    c2.metric("Frame Rate", f"{fps:.0f} FPS")

    c3.metric("Video Duration", f"{duration:.2f} s")

    c4.metric("Extracted Landmarks", result["landmarks"])


    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Target Landmarks", TARGET_LANDMARKS)

    c6.metric("Target Frames", TARGET_FRAMES)

    c7.metric(
        "Pose Extraction",
        f'{result["extraction_time"]:.2f} s'
    )

    c8.metric(
        "Total Processing",
        f'{result["total_time"]:.2f} s'
    )


    # ==================================================
    # ADAPTIVE FRAME SAMPLING
    # ==================================================

    section_title("04 — Adaptive Frame Sampling")

    st.html("""
    <div class="adaptive-box">

        <h3 style="margin-top:0;">
            Motion-Guided Frame Selection
        </h3>

        <p style="color:#475569; margin-bottom:0;">
            Adaptive sampling combines temporal coverage
            with motion-based frame selection to preserve
            informative signing movements while reducing
            redundant frames.
        </p>

    </div>
    """)

    c1, c2 = st.columns(2)

    c1.metric("Original Sequence", frames)

    c2.metric("Target Sequence Length", TARGET_FRAMES)

    st.caption("Adaptive sampler integration pending.")


    # ==================================================
    # ENGLISH TRANSLATION
    # ==================================================

    section_title("05 — English Translation")

    st.html("""
    <div class="translation-box">

        <div class="translation-label">
            GENERATED ENGLISH SENTENCE
        </div>

        <div class="translation-text">
            Awaiting trained model
        </div>

        <div style="color:#64748B; font-size:13px;">
            Adaptive pose sequence → T5 → English
        </div>

    </div>
    """)


    # ==================================================
    # DOWNLOAD RESULTS
    # ==================================================

    section_title("06 — Download Results")

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "Download Pose File",
            result["pose"],
            "extracted.pose",
            "application/octet-stream",
            use_container_width=True
        )

    with d2:

        st.download_button(
            "Download Skeleton",
            result["skeleton"],
            "skeleton.mp4",
            "video/mp4",
            use_container_width=True
        )

    with d3:

        st.download_button(
            "Download Overlay",
            result["overlay"],
            "overlay.mp4",
            "video/mp4",
            use_container_width=True
        )


# ==================================================
# FOOTER
# ==================================================

st.html("""
<div class="project-footer">
    Efficient Continuous Indian Sign Language Translation
    | Research Project
</div>
""")

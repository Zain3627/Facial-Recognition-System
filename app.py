"""Streamlit entrypoint for the FaceNet-based recognition demo."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import streamlit.components.v1 as st_components

# Add project directory to path for imports
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    BACKBONE_MODEL_DIR,
    DEPLOY_METADATA_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_STORE_PATH,
    NUM_REGISTRATION_SHOTS,
    SIMILARITY_THRESHOLD,
)
from services.embedding_service import FaceNetEmbeddingService
from services.recognizer import EmbeddingRecognizer, RecognitionResult
from storage.embedding_store import EmbeddingStore
from utils.image_utils import load_image_as_array
from utils.face_detection import FaceDetector, FaceRegion


st.set_page_config(page_title="FaceNet Recognition", page_icon="👤", layout="wide")


@st.cache_resource(show_spinner="Loading FaceNet model…")
def load_services() -> tuple[
    dict,
    FaceNetEmbeddingService,
    EmbeddingStore,
    EmbeddingRecognizer,
    FaceDetector,
]:
    metadata = json.loads(Path(DEPLOY_METADATA_PATH).read_text(encoding="utf-8"))

    # Get embedding dim from transfer head metadata (input.shape[1]) or default to 128
    embedding_dim = metadata.get("input", {}).get("shape", [None, 128])[1]

    embedding_service = FaceNetEmbeddingService(
        backbone_dir=BACKBONE_MODEL_DIR,
        embedding_dim=embedding_dim,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    embedding_store = EmbeddingStore(EMBEDDING_STORE_PATH)

    recognizer = EmbeddingRecognizer(
        embedding_service=embedding_service,
        embedding_store=embedding_store,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    
    # Face detector for multi-face detection
    face_detector = FaceDetector(
        min_confidence=0.9,
        min_face_size=20,
        margin=0.2,
    )

    return metadata, embedding_service, embedding_store, recognizer, face_detector


metadata, embedding_service, embedding_store, recognizer, face_detector = load_services()



def _init_registration_session() -> None:
    if "registration_frames" not in st.session_state:
        st.session_state["registration_frames"] = []
    if "registration_hashes" not in st.session_state:
        st.session_state["registration_hashes"] = set()


def _reset_registration_session() -> None:
    st.session_state["registration_frames"] = []
    st.session_state["registration_hashes"] = set()
    _clear_registration_capture()


def _clear_registration_capture() -> None:
    st.session_state.pop("registration_camera", None)
    st.session_state.pop("registration_last_digest", None)


def _store_capture(frame_bytes: bytes) -> None:
    frames: List[bytes] = st.session_state["registration_frames"]
    hashes = st.session_state["registration_hashes"]
    digest = hashlib.sha1(frame_bytes).hexdigest()
    if digest in hashes:
        return
    if len(frames) >= NUM_REGISTRATION_SHOTS:
        return
    frames.append(frame_bytes)
    hashes.add(digest)


def _render_sidebar(model_metadata: dict) -> None:
    st.sidebar.header("FaceNet Recognition")
    labels = embedding_store.list_labels()
    st.sidebar.info(f"{len(labels)} identities enrolled.")


def _draw_face_results(
    image_bytes: bytes,
    faces: List[FaceRegion],
    results: List[RecognitionResult],
) -> bytes:
    """Draw bounding boxes and labels on an image.
    
    Returns the annotated image as bytes.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Try to use a larger font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for face, result in zip(faces, results):
        x1, y1, x2, y2 = face.box
        
        # Choose color based on match
        if result.label:
            color = (0, 255, 0)  # Green for match
            label_text = f"{result.label} ({result.similarity:.2f})"
        else:
            color = (255, 0, 0)  # Red for no match
            label_text = f"Unknown ({result.similarity:.2f})"
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)
    
    # Convert back to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _process_faces(image_array: np.ndarray) -> Tuple[List[FaceRegion], List[RecognitionResult]]:
    """Detect faces and run recognition on each.
    
    Returns tuple of (faces, results).
    """
    faces = face_detector.detect_faces(image_array)
    results = []
    
    for face in faces:
        # Crop face with margin
        face_crop = face_detector.crop_face(image_array, face, target_size=(160, 160))
        # Run recognition on cropped face
        result = recognizer.identify(face_crop)
        results.append(result)
    
    return faces, results


def _identify_section() -> None:
    st.subheader("Identify faces")
    sources = ["Upload", "Camera"]
    source = st.radio("Select input source", sources, horizontal=True)

    image_bytes: Optional[bytes] = None
    if source == "Upload":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        camera_capture = st.camera_input("Capture a photo", key="identification_camera")
        if camera_capture is not None:
            image_bytes = camera_capture.getvalue()

    if image_bytes is not None:
        st.image(image_bytes, caption="Query image")

    trigger = st.button("Run recognition", disabled=image_bytes is None)
    if trigger and image_bytes is not None:
        with st.spinner("Detecting faces and running recognition..."):
            # Load as RGB float32 [0, 1]
            image_array = load_image_as_array(image_bytes)
            
            # Detect and process all faces
            faces, results = _process_faces(image_array)
            
            if not faces:
                st.warning("❌ No faces detected in the image.")
            else:
                # Draw annotated image
                annotated_bytes = _draw_face_results(image_bytes, faces, results)
                st.image(annotated_bytes, caption=f"Detected {len(faces)} face(s)")
                
                # Show results
                st.divider()
                for i, (face, result) in enumerate(zip(faces, results)):
                    if result.label:
                        st.success(f"Face {i+1}: ✅ **{result.label}** (similarity {result.similarity:.2f})")
                    else:
                        st.warning(f"Face {i+1}: ❌ No match (best similarity {result.similarity:.2f})")




def _registration_section() -> None:
    st.subheader("Register a new user")

    st.write(
        "Capture five shots using your webcam. The app will detect your face in each shot. "
        "After all captures are collected and a name is provided, the embeddings will be stored."
    )

    _init_registration_session()

    frames: List[bytes] = st.session_state["registration_frames"]
    shots_taken = len(frames)

    st.write(f"Captured shots: {shots_taken} / {NUM_REGISTRATION_SHOTS}")
    st.progress(min(shots_taken / NUM_REGISTRATION_SHOTS, 1.0))

    allow_more_shots = shots_taken < NUM_REGISTRATION_SHOTS

    if allow_more_shots:
        st.info("Use the webcam below to capture the next shot. Make sure your face is visible.")
    else:
        st.success("All required shots captured. You can save or reset below.")

    if allow_more_shots:
        capture = st.camera_input("Capture registration photo", key="registration_camera")
        if capture is not None:
            frame_bytes = capture.getvalue()
            digest = hashlib.sha1(frame_bytes).hexdigest()

            if digest != st.session_state.get("registration_last_digest"):
                # Check if a face is detected
                image_array = load_image_as_array(frame_bytes)
                faces = face_detector.detect_faces(image_array)
                
                if faces:
                    _store_capture(frame_bytes)
                    st.session_state["registration_last_digest"] = digest
                    _clear_registration_capture()
                    st.rerun()
                else:
                    st.error("❌ No face detected in the image. Please try again.")
    else:
        capture = None

    if frames:
        st.write(f"Collected {len(frames)} / {NUM_REGISTRATION_SHOTS} frames")
        st.image(
            frames,
            caption=[f"Shot {i + 1}" for i in range(len(frames))],
            width=160,
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("Name to register")
    with col2:
        if st.button("🗑️ Reset", use_container_width=True):
            _reset_registration_session()
            st.rerun()

    enough_frames = len(frames) >= NUM_REGISTRATION_SHOTS
    can_persist = bool(name.strip()) and enough_frames

    if st.button("Save enrollment", disabled=not can_persist, use_container_width=True):
        clean_name = name.strip()
    
        try:
            with st.spinner("Processing faces and generating embeddings..."):
                # Load images, detect faces, crop, and generate embeddings
                embedding_list = []
                for frame in frames:
                    img = load_image_as_array(frame)
                    faces = face_detector.detect_faces(img)
                    
                    if faces:
                        # Use the first (largest) detected face
                        face_crop = face_detector.crop_face(img, faces[0], target_size=(160, 160))
                        batch = embedding_service.embed_batch([face_crop])
                        embedding_list.append(batch.embeddings[0])
                    else:
                        # Fallback to full image if no face detected
                        batch = embedding_service.embed_batch([img])
                        embedding_list.append(batch.embeddings[0])
    
                # Store embeddings
                embedding_store.upsert_embeddings(clean_name, embedding_list)
    
            _reset_registration_session()
            st.success(f"✅ Stored embeddings for {clean_name}.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save embeddings: {e}")


def _live_video_section() -> None:
    st.subheader("Live Video Recognition")
    
    # Initialize session state
    if "live_results" not in st.session_state:
        st.session_state["live_results"] = []
    if "live_annotated_frame" not in st.session_state:
        st.session_state["live_annotated_frame"] = None
    if "live_running" not in st.session_state:
        st.session_state["live_running"] = False
    if "last_frame_hash" not in st.session_state:
        st.session_state["last_frame_hash"] = None
    
    # Layout columns
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.write("**🎮 Controls**")
        
        if st.session_state["live_running"]:
            if st.button("⏹️ Stop", type="primary", use_container_width=True):
                st.session_state["live_running"] = False
                st.rerun()
            st.success("🔴 **LIVE** - Auto-capturing 1 frame/sec")
        else:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                st.session_state["live_running"] = True
                st.rerun()
            st.info("Click Start to begin auto-capture")
        
        st.divider()
        st.write("**🎯 Recognition Results**")
        
        if st.session_state["live_results"]:
            for entry in st.session_state["live_results"][:10]:
                if entry["label"]:
                    st.success(f"🕐 {entry['time']} → ✅ **{entry['label']}** ({entry['similarity']:.2f})")
                else:
                    st.warning(f"🕐 {entry['time']} → ❌ Unknown ({entry['similarity']:.2f})")
        else:
            st.caption("No results yet.")
        
        st.divider()
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state["live_results"] = []
            st.session_state["live_annotated_frame"] = None
            st.rerun()
    
    with col1:
        is_running = st.session_state["live_running"]
        
        # Create unique key for camera to force refresh
        camera_key = f"live_cam_{int(time.time())}" if is_running else "live_cam_static"
        
        if is_running:
            st.info("🔴 **LIVE MODE** - Frames are being captured automatically. Keep your face visible.")
            
            # Use camera_input which works reliably
            # We'll auto-refresh the page to get new frames
            captured = st.camera_input(
                "Live capture (auto-refreshing)",
                key=camera_key,
                label_visibility="collapsed"
            )
            
            if captured is not None:
                frame_bytes = captured.getvalue()
                frame_hash = hashlib.sha1(frame_bytes).hexdigest()[:16]
                
                # Only process if it's a new frame
                if frame_hash != st.session_state["last_frame_hash"]:
                    st.session_state["last_frame_hash"] = frame_hash
                    
                    try:
                        # Process the frame
                        image_array = load_image_as_array(frame_bytes)
                        faces, results = _process_faces(image_array)
                        
                        if faces:
                            annotated_bytes = _draw_face_results(frame_bytes, faces, results)
                            st.session_state["live_annotated_frame"] = annotated_bytes
                            
                            # Update results
                            timestamp = time.strftime("%H:%M:%S")
                            for face, result in zip(faces, results):
                                entry = {
                                    "time": timestamp,
                                    "label": result.label,
                                    "similarity": result.similarity
                                }
                                st.session_state["live_results"].insert(0, entry)
                            st.session_state["live_results"] = st.session_state["live_results"][:25]
                            
                            # Show annotated result
                            st.image(annotated_bytes, caption=f"Detected {len(faces)} face(s)")
                        else:
                            st.warning("No faces detected")
                            st.image(frame_bytes, caption="No faces detected")
                            
                    except Exception as e:
                        st.error(f"Error processing: {e}")
            
            # Auto-refresh after delay
            time.sleep(1.0)
            st.rerun()
        else:
            # Show static video preview when not running
            video_preview = """
            <div style="text-align: center; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
                <video id="preview" autoplay playsinline muted
                    style="width: 100%; max-width: 640px; border-radius: 12px; border: 3px solid #2196F3; background: #000;">
                </video>
                <div style="margin-top: 10px; padding: 10px; border-radius: 8px; background: #e3f2fd; color: #1565c0; font-weight: 500;">
                    ⏸️ Camera ready. Click <b>Start</b> to begin auto-capture.
                </div>
            </div>
            <script>
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
            .then(stream => { document.getElementById('preview').srcObject = stream; })
            .catch(err => console.error(err));
            </script>
            """
            st_components.html(video_preview, height=420)
            
            # Show last annotated frame if available
            if st.session_state["live_annotated_frame"]:
                st.image(st.session_state["live_annotated_frame"], caption="Last processed frame")
    
    # Full history expander
    if st.session_state["live_results"]:
        st.divider()
        with st.expander(f"📜 Full History ({len(st.session_state['live_results'])} entries)"):
            for entry in st.session_state["live_results"]:
                if entry["label"]:
                    st.write(f"🕐 {entry['time']} → ✅ **{entry['label']}** ({entry['similarity']:.2f})")
                else:
                    st.write(f"🕐 {entry['time']} → ❌ Unknown ({entry['similarity']:.2f})")


_render_sidebar(metadata)
tabs = st.tabs(["Identify", "Live Video", "Register"])
with tabs[0]:
    _identify_section()
with tabs[1]:
    _live_video_section()
with tabs[2]:
    _registration_section()

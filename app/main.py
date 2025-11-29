"""Streamlit entrypoint for the FaceNet-based recognition demo."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, cast

import numpy as np
import time
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import streamlit_webrtc
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )

    STREAMLIT_WEBRTC_AVAILABLE = True
except ImportError:
    RTCConfiguration = None
    VideoProcessorBase = cast(Any, object)
    WebRtcMode = None
    webrtc_streamer = None
    STREAMLIT_WEBRTC_AVAILABLE = False

from app.config import (
    BACKBONE_MODEL_DIR,
    DEPLOY_METADATA_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_STORE_PATH,
    NUM_REGISTRATION_SHOTS,
    SIMILARITY_THRESHOLD,
)
from app.services.embedding_service import FaceNetEmbeddingService
from app.services.recognizer import EmbeddingRecognizer, RecognitionResult
from app.storage.embedding_store import EmbeddingStore
from app.utils.image_utils import load_image_as_array


st.set_page_config(page_title="FaceNet Recognition", page_icon="👤", layout="wide")

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}

LIVE_SAMPLE_INTERVAL_SECONDS = 0.5


if STREAMLIT_WEBRTC_AVAILABLE:

    class LiveRecognitionProcessor(VideoProcessorBase):

        def __init__(
            self,
            recognizer: EmbeddingRecognizer,
            sample_interval: float = LIVE_SAMPLE_INTERVAL_SECONDS,
        ) -> None:
            self._recognizer = recognizer
            self._sample_interval = max(0.1, sample_interval)
            self._last_timestamp = 0.0
            self.last_result: Optional[RecognitionResult] = None
            self.result_sequence = 0

        def recv(self, frame):  # type: ignore[override]
            now = time.time()
            if now - self._last_timestamp >= self._sample_interval:
                rgb = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                self.last_result = self._recognizer.identify(rgb)
                self.result_sequence += 1
                self._last_timestamp = now
            return frame


@st.cache_resource(show_spinner="Loading FaceNet model…")
def load_services() -> tuple[dict, FaceNetEmbeddingService, EmbeddingStore, EmbeddingRecognizer]:
    metadata = json.loads(Path(DEPLOY_METADATA_PATH).read_text(encoding="utf-8"))
    embedding_service = FaceNetEmbeddingService(
        backbone_dir=BACKBONE_MODEL_DIR,
        embedding_dim=int(metadata.get("embedding_dim", 128)),
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    embedding_store = EmbeddingStore(EMBEDDING_STORE_PATH)
    recognizer = EmbeddingRecognizer(
        embedding_service=embedding_service,
        embedding_store=embedding_store,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    return metadata, embedding_service, embedding_store, recognizer


metadata, embedding_service, embedding_store, recognizer = load_services()


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
    st.sidebar.header("Enrolled identities")
    labels = embedding_store.list_labels()
    if labels:
        for label in sorted(labels):
            st.sidebar.write(f"• {label}")
    else:
        st.sidebar.info("No identities enrolled yet.")


def _identify_section() -> None:
    st.subheader("Identify a face")
    sources = ["Upload", "Camera"]
    if STREAMLIT_WEBRTC_AVAILABLE:
        sources.append("Live Video")
    source = st.radio("Select input source", sources, horizontal=True)

    if source == "Live Video":
        if STREAMLIT_WEBRTC_AVAILABLE and webrtc_streamer is not None and WebRtcMode is not None:
            st.info("Press START to begin live identification. Results update continuously while recording.")
            context = webrtc_streamer(
                key="live_identification",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                video_processor_factory=lambda: LiveRecognitionProcessor(
                    recognizer, sample_interval=LIVE_SAMPLE_INTERVAL_SECONDS
                ),
            )
            result_placeholder = st.empty()

            # Continuous polling loop while the stream is active
            if context.state.playing:
                processor = context.video_processor
                if processor is not None and processor.last_result is not None:
                    result = processor.last_result
                    if result.label:
                        result_placeholder.success(
                            f"✅ Match: **{result.label}** (similarity {result.similarity:.2f})"
                        )
                    else:
                        result_placeholder.warning(
                            f"❌ No match (best similarity {result.similarity:.2f})"
                        )
                else:
                    result_placeholder.info("🔄 Processing frames…")
                # Trigger a rerun after a short delay to poll for new results
                time.sleep(LIVE_SAMPLE_INTERVAL_SECONDS)
                st.rerun()
            else:
                result_placeholder.caption("Press START above to begin live recognition.")
        else:
            st.warning(
                "Install the optional dependency 'streamlit-webrtc' (and 'av') to enable live video recognition."
            )
        return

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
        st.image(image_bytes, caption="Query image", use_container_width=True)

    trigger = st.button("Run recognition", disabled=image_bytes is None)
    if trigger and image_bytes is not None:
        image_array = load_image_as_array(image_bytes)
        result = recognizer.identify(image_array)
        if result.label:
            st.success(f"Match found: {result.label} (similarity {result.similarity:.2f})")
        else:
            st.warning(
                "No stored identity exceeded the similarity threshold. "
                f"Best similarity: {result.similarity:.2f}"
            )


def _registration_section() -> None:
    st.subheader("Register a new user")
    st.write(
        "Capture five shots using your webcam. After all captures are collected and a name is provided, "
        "the embeddings will be stored for future identification."
    )
    _init_registration_session()

    frames: List[bytes] = st.session_state["registration_frames"]
    shots_taken = len(frames)

    st.write(f"Captured shots: {shots_taken} / {NUM_REGISTRATION_SHOTS}")
    st.progress(min(shots_taken / NUM_REGISTRATION_SHOTS, 1.0))

    allow_more_shots = shots_taken < NUM_REGISTRATION_SHOTS
    if allow_more_shots:
        st.info("Use the webcam below to capture the next shot.")
    else:
        st.success("All required shots captured. You can save or reset below.")

    if allow_more_shots:
        capture = st.camera_input("Capture registration photo", key="registration_camera")
        if capture is not None:
            frame_bytes = capture.getvalue()
            digest = hashlib.sha1(frame_bytes).hexdigest()
            if digest != st.session_state.get("registration_last_digest"):
                _store_capture(frame_bytes)
                st.session_state["registration_last_digest"] = digest
                _clear_registration_capture()
                st.rerun()
    else:
        capture = None



    if frames:
        st.write(f"Collected {len(frames)} / {NUM_REGISTRATION_SHOTS} frames")
        st.image(frames, caption=[f"Shot {i + 1}" for i in range(len(frames))], width=160)

    name = st.text_input("Name to register")
    enough_frames = len(frames) >= NUM_REGISTRATION_SHOTS
    can_persist = bool(name.strip()) and enough_frames
    if st.button("Save enrollment", disabled=not can_persist):
        clean_name = name.strip()
        images = [load_image_as_array(frame) for frame in frames]
        embedding_batch = embedding_service.embed_batch(images)
        embedding_store.upsert_embeddings(clean_name, embedding_batch.embeddings)
        _reset_registration_session()
        st.success(f"Stored embeddings for {clean_name}.")


_render_sidebar(metadata)
tabs = st.tabs(["Identify", "Register"])
with tabs[0]:
    _identify_section()
with tabs[1]:
    _registration_section()

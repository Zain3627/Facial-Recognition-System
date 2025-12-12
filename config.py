"""Shared configuration for the Streamlit facial recognition app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOY_MODEL_DIR = PROJECT_ROOT / "deployment_model"
DEPLOY_METADATA_PATH = DEPLOY_MODEL_DIR / "transfer_head" / "metadata.json"
BACKBONE_MODEL_DIR = PROJECT_ROOT / "models" / "facenet"
TRANSFER_HEAD_PATH = DEPLOY_MODEL_DIR / "transfer_head" / "transfer_head.keras"
APP_DATA_DIR = PROJECT_ROOT / "app_data"
EMBEDDING_STORE_PATH = APP_DATA_DIR / "embeddings_store.json"
SIMILARITY_THRESHOLD = 0.60
NUM_REGISTRATION_SHOTS = 5
EMBEDDING_BATCH_SIZE = 32

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

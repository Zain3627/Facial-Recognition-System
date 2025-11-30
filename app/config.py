"""Shared configuration for the Streamlit facial recognition app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_MODEL_DIR = PROJECT_ROOT / "deployment_model" / "facenet_transfer_model"
DEPLOY_METADATA_PATH = PROJECT_ROOT / "deployment_model" / "metadata.json"
BACKBONE_MODEL_DIR = PROJECT_ROOT / "models" / "facenet"
APP_DATA_DIR = PROJECT_ROOT / "app_data"
EMBEDDING_STORE_PATH = APP_DATA_DIR / "embeddings_store.json"
SIMILARITY_THRESHOLD = 0.75
NUM_REGISTRATION_SHOTS = 5
EMBEDDING_BATCH_SIZE = 32

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

"""Script to populate embeddings_store.json with all training data."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# Load training data
print("Loading training embeddings and labels...")
train_embeddings = np.load(PROJECT_ROOT / "data" / "facenet_train_embeddings.npy")
train_labels = np.load(PROJECT_ROOT / "data" / "y_train.npy", allow_pickle=True)

print(f"Loaded {len(train_embeddings)} embeddings")
print(f"Loaded {len(train_labels)} labels")

# Load name mappings from LFW dataset
lfw_names = pd.read_csv(PROJECT_ROOT / "LFW" / "lfw_allnames.csv")
print(f"Loaded {len(lfw_names)} person names from LFW dataset")

# Create label to name mapping (assuming labels are indices into the name list)
unique_labels = np.unique(train_labels)
print(f"Found {len(unique_labels)} unique identities")

# Group embeddings by label
embeddings_by_label = {}
for emb, label in zip(train_embeddings, train_labels):
    if label not in embeddings_by_label:
        embeddings_by_label[label] = []
    embeddings_by_label[label].append(emb)

# Prepare data for JSON storage
users_data = []
now = datetime.utcnow().isoformat()

for label_idx in sorted(embeddings_by_label.keys()):
    # Get person name - use numeric index if name not available
    if label_idx < len(lfw_names):
        person_name = lfw_names.iloc[label_idx]['name']
    else:
        person_name = f"Person_{label_idx}"
    
    embeddings_list = embeddings_by_label[label_idx]
    
    # Normalize embeddings (L2 normalization)
    normalized_embeddings = []
    for emb in embeddings_list:
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            normalized = emb / norm
        else:
            normalized = emb
        normalized_embeddings.append(normalized.tolist())
    
    user_entry = {
        "label": person_name,
        "embeddings": normalized_embeddings,
        "created_at": now,
        "updated_at": now
    }
    users_data.append(user_entry)
    
    if len(users_data) % 50 == 0:
        print(f"Processed {len(users_data)} identities...")

# Save to JSON
output_path = PROJECT_ROOT / "app_data" / "embeddings_store.json"
output_data = {"users": users_data}

print(f"\nSaving {len(users_data)} identities to {output_path}...")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Successfully saved {len(users_data)} identities with {len(train_embeddings)} total embeddings")
print(f"Average embeddings per person: {len(train_embeddings) / len(users_data):.1f}")

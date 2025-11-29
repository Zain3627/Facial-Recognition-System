# Facial Recognition System

## Overview
This project implements an end-to-end facial recognition system using deep learning architectures (FaceNet). It features a complete pipeline from data preprocessing and model training to a production-ready Streamlit web application. The system allows for real-time face identification via webcam, user registration, and persistent storage of face embeddings.

## Objectives
- **High Accuracy:** Achieve robust face recognition performance using the FaceNet architecture.
- **Real-time Processing:** Enable live video identification with low latency.
- **User-Friendly Interface:** Provide a simple, interactive web application for end-users.
- **Scalable Architecture:** Modular design separating services, storage, and UI.


## Features
- **Face Identification:**
  - **Upload:** Identify faces from uploaded images (JPG, PNG).
  - **Camera:** Capture a snapshot for immediate identification.
  - **Live Video:** Real-time identification from a webcam feed with periodic updates.
- **User Registration:**
  - Guided 5-shot webcam capture process.
  - Automatic deduplication of captured frames.
  - Generation and storage of normalized face embeddings.
- **Embedding Storage:**
  - Persistent JSON-based storage for enrolled identities.
  - Nearest-neighbor search using cosine similarity (via dot product of normalized vectors).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd facial_recognition_system
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r app/requirements.txt
   ```

   *Note: For live video support, `streamlit-webrtc` and `av` are required.*

## Usage

1. **Run the Streamlit App:**
   Execute the following command from the project root:
   ```bash
   streamlit run app/main.py
   ```

2. **Navigate the Interface:**
   - **Identify Tab:** Choose between "Upload", "Camera", or "Live Video" to recognize faces.
   - **Register Tab:** Follow the on-screen instructions to capture 5 photos and register a new identity.

## Model Details
The system uses **FaceNet** as the backbone for generating 128-dimensional face embeddings.
- **Input:** 160x160 RGB images.
- **Preprocessing:** Resizing and normalization ([-1, 1]).
- **Matching:** L2-normalized embeddings are compared using dot product similarity. The default acceptance threshold is set to **0.6**.

## Team
- Basel Tarek Abdelmonsif
- Mohammed Essam Shehata
- Mostafa Sayed Salah
- Naser Ali Naser
- Zain Tamer Zain El-Abdin Awad

## License
This project is part of the Microsoft Machine Learning Project Round 3.
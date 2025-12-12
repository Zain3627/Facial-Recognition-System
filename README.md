# Facial Recognition System

A real-time facial recognition system built with **FaceNet** embeddings and **transfer learning**, deployed as an interactive **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project implements a complete facial recognition pipeline:

1. **FaceNet Backbone**: Pre-trained FaceNet model for generating 128-dimensional face embeddings
2. **Transfer Learning**: Fine-tuned classification head on the LFW (Labeled Faces in the Wild) dataset
3. **Real-time Recognition**: Live webcam face detection and identification
4. **User Registration**: Enroll new faces via webcam capture

## 🏗️ Project Structure

```
facial_recognition_system/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker deployment configuration
├── data/                       # Training data and embeddings
│   ├── X_train.npy            # Training images
│   ├── y_train.npy            # Training labels
│   ├── facenet_train_embeddings.npy
│   └── ...
├── models/
│   └── facenet/               # Pre-trained FaceNet backbone
├── deployment_model/
│   ├── metadata.json          # Model metadata
│   ├── facenet_transfer_model/  # Fine-tuned model
│   └── transfer_head/         # Classification head
├── app_data/
│   └── embeddings_store.json  # Enrolled face embeddings
├── services/
│   ├── embedding_service.py   # FaceNet embedding extraction
│   └── recognizer.py          # Face recognition logic
├── storage/
│   └── embedding_store.py     # Embedding persistence
├── utils/
│   ├── image_utils.py         # Image preprocessing
│   └── face_detection.py      # Face detection utilities
├── notebooks/
│   ├── 01_data_exploration_and_augmentation.ipynb
│   ├── 02_facenet_baseline_evaluation.ipynb
│   └── 03_facenet_transfer_learning.ipynb
├── LFW/                        # LFW dataset
│   └── lfw-deepfunneled/      # Aligned face images
└── documentation/
    └── documentation.tex      # LaTeX documentation
```

## 🚀 Features

- **Multi-face Detection**: Detect and recognize multiple faces in a single image
- **Real-time Video Recognition**: Live webcam stream with continuous face matching
- **User Registration**: Capture 5 photos to enroll a new identity
- **Similarity Threshold**: Configurable matching threshold (default: 0.60)
- **303+ Pre-enrolled Identities**: Trained on LFW dataset with 640 face embeddings

## 📋 Requirements

- Python 3.10+
- TensorFlow 2.x
- Streamlit
- MediaPipe (for face detection)
- NumPy, Pillow, OpenCV

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Zain3627/Facial-Recognition-System.git
cd Facial-Recognition-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

Ensure the following directories contain the required model files:
- `models/facenet/` - FaceNet backbone (SavedModel format)
- `deployment_model/transfer_head/` - Fine-tuned classification head

## 🎮 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### App Features

#### 🔍 Identify Tab
- **Upload**: Upload an image to identify faces
- **Camera**: Capture a photo using your webcam
- Click "Run recognition" to detect and identify all faces in the image

#### 📝 Register Tab
- Enter a name for the new identity
- Capture 5 photos from different angles
- Click "Save enrollment" to store the face embeddings

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t facial-recognition-system .
```

### Run the Container

```bash
docker run -p 8501:8501 facial-recognition-system
```

## 📊 Model Architecture

### FaceNet Backbone
- **Input**: 160×160×3 RGB face images
- **Output**: 128-dimensional L2-normalized embeddings
- **Architecture**: Inception-ResNet-V1

### Recognition Pipeline

```
Input Image → Face Detection → Crop & Align → FaceNet Embedding → Similarity Search → Match/No Match
```

### Similarity Matching
- **Method**: Cosine similarity (dot product of normalized embeddings)
- **Threshold**: 0.60 (configurable in `config.py`)
- **Comparison**: Query embedding vs. average embedding per enrolled identity

## 📓 Notebooks

1. **01_data_exploration_and_augmentation.ipynb**
   - LFW dataset exploration
   - Data augmentation techniques
   - Train/validation/test split

2. **02_facenet_baseline_evaluation.ipynb**
   - FaceNet embedding extraction
   - Baseline classification performance
   - Embedding visualization (t-SNE)

3. **03_facenet_transfer_learning.ipynb**
   - Transfer learning with frozen backbone
   - Classification head training
   - MLflow experiment tracking
   - Model export and deployment

## ⚙️ Configuration

Edit `config.py` to customize:

```python
SIMILARITY_THRESHOLD = 0.60    # Match confidence threshold
NUM_REGISTRATION_SHOTS = 5     # Photos required for registration
EMBEDDING_BATCH_SIZE = 32      # Batch size for embedding extraction
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Enrolled Identities | 303 |
| Total Embeddings | 640 |
| Embedding Dimension | 128 |
| Face Detection | MediaPipe |
| Recognition Speed | ~0.5s per frame |

## 🔧 Troubleshooting

### Common Issues

1. **"No match" for all faces**
   - Lower `SIMILARITY_THRESHOLD` in `config.py`
   - Ensure good lighting and face visibility
   - Check that embeddings are properly loaded

2. **Slow performance**
   - Reduce image resolution
   - Use GPU-enabled TensorFlow

3. **Camera not working**
   - Allow camera permissions in browser
   - Check webcam connection

## 📚 References

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Labeled Faces in the Wild (LFW) Dataset](http://vis-www.cs.umass.edu/lfw/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Zain** - [GitHub](https://github.com/Zain3627)

---

⭐ Star this repository if you found it helpful!

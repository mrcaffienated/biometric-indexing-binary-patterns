"""
feature_utils.py
Centralized feature extraction for multi-modal biometric system.
- Face: DeepFace FaceNet (128-D)
- Iris: MobileNetV2 feature extraction (128-D)
- Fingerprint: MobileNetV2 feature extraction (128-D)
"""

import numpy as np
import os

# ── Face Embedding ──────────────────────────────────────────────────
def extract_face_embedding(image_path: str) -> np.ndarray:
    """Extract 128-D face embedding using DeepFace FaceNet."""
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )
    return np.array(result[0]["embedding"])


# ── Generic CNN Embedding (for Iris & Fingerprint) ──────────────────
_mobilenet_model = None

# Compatible imports for both TF ≤2.15 (tensorflow.keras) and TF ≥2.16 (standalone keras)
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except (ImportError, AttributeError):
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input
    from keras.utils import load_img, img_to_array

def _get_mobilenet():
    """Lazy-load MobileNetV2 to save memory."""
    global _mobilenet_model
    if _mobilenet_model is None:
        _mobilenet_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        )
    return _mobilenet_model


def _extract_generic_embedding(image_path: str) -> np.ndarray:
    """Extract embedding from any image using MobileNetV2 → project to 128-D."""

    model = _get_mobilenet()

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()  # 1280-D

    # Project down to 128-D using a fixed random projection
    np.random.seed(99)
    proj = np.random.randn(len(features), 128)
    embedding_128 = features @ proj
    # L2 normalize
    norm = np.linalg.norm(embedding_128)
    if norm > 0:
        embedding_128 = embedding_128 / norm
    return embedding_128


def extract_iris_embedding(image_path: str) -> np.ndarray:
    """Extract 128-D iris embedding."""
    return _extract_generic_embedding(image_path)


def extract_fingerprint_embedding(image_path: str) -> np.ndarray:
    """Extract 128-D fingerprint embedding."""
    return _extract_generic_embedding(image_path)


# ── Binary Template Creation ────────────────────────────────────────
def create_binary_template(embedding: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Apply cancelable transform and binarize."""
    protected = np.dot(embedding, projection_matrix)
    threshold = np.median(protected)
    return (protected > threshold).astype(int)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Hamming similarity between two binary templates."""
    return float(np.sum(a == b) / len(a))


# ── Projection Matrix Management ───────────────────────────────────
def get_or_create_projection_matrix(path: str, dim: int = 128) -> np.ndarray:
    """Load existing projection matrix or create a new one."""
    if os.path.exists(path):
        return np.load(path)
    R = np.random.randn(dim, dim)
    np.save(path, R)
    return R

"""
feature_utils.py
Centralized feature extraction for multi-modal biometric system.
- Face: DeepFace FaceNet (128-D)
- Iris: MobileNetV2 feature extraction (128-D)
- Fingerprint: MobileNetV2 feature extraction (128-D)

NOTE: TensorFlow/Keras imports are deferred (lazy) so that FBP functions
remain importable even when TF is unavailable or fails to load.
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
_keras_modules = {}  # cache for lazy-loaded keras functions


def _load_keras_modules():
    """Lazy-load Keras modules, compatible with TF ≤2.15 and TF ≥2.16."""
    global _keras_modules
    if _keras_modules:
        return _keras_modules
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
    except (ImportError, AttributeError):
        from keras.applications import MobileNetV2
        from keras.applications.mobilenet_v2 import preprocess_input
        from keras.utils import load_img, img_to_array
    _keras_modules = {
        "MobileNetV2": MobileNetV2,
        "preprocess_input": preprocess_input,
        "load_img": load_img,
        "img_to_array": img_to_array,
    }
    return _keras_modules


def _get_mobilenet():
    """Lazy-load MobileNetV2 to save memory."""
    global _mobilenet_model
    if _mobilenet_model is None:
        km = _load_keras_modules()
        _mobilenet_model = km["MobileNetV2"](
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        )
    return _mobilenet_model


def _extract_generic_embedding(image_path: str) -> np.ndarray:
    """Extract embedding from any image using MobileNetV2 → project to 128-D."""
    km = _load_keras_modules()

    model = _get_mobilenet()

    img = km["load_img"](image_path, target_size=(224, 224))
    img_array = km["img_to_array"](img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = km["preprocess_input"](img_array)

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


# ── Frequent Binary Pattern (FBP) Functions ─────────────────────────

def extract_frequent_patterns(binary_template: np.ndarray, window_length: int = 6) -> list:
    """
    Extract frequent binary sub-patterns from a binary template using a sliding window.
    Returns a list of pattern strings ranked by frequency (most frequent first).
    Only patterns occurring more than once are included.
    """
    from itertools import product as iter_product

    # Generate all possible K-length binary combinations
    K_combinations = {}
    for c in enumerate(iter_product(range(2), repeat=window_length)):
        K_combinations[str(c[1])] = 0

    # Slide window across binary template and count pattern occurrences
    i = 0
    while i < (len(binary_template) - window_length) or (len(binary_template) - i >= window_length):
        candidate = binary_template[i:i + window_length]
        candidate_str = str(tuple(map(int, candidate)))
        if candidate_str in K_combinations:
            K_combinations[candidate_str] += 1
        i += 1

    # Sort by frequency (descending)
    sorted_patterns = dict(sorted(K_combinations.items(), key=lambda x: x[1], reverse=True))

    # Keep only patterns with frequency > 1
    frequent = [k for k, v in sorted_patterns.items() if v > 1]

    # Clean pattern strings: "(0, 1, 1, 0, 1, 0)" -> "011010"
    cleaned = []
    for code in frequent:
        code = code.strip('()')
        parts = code.split(',')
        pattern = ''.join(p.strip() for p in parts)
        cleaned.append(pattern)

    return cleaned


def rank_patterns_across_modalities(pattern_lists: list) -> list:
    """
    Rank frequent patterns across 2 or 3 modalities.
    Finds patterns common to all modalities and ranks them by combined rank (lower = better).
    Returns a list of pattern strings ordered by rank.
    """
    if len(pattern_lists) < 2:
        return pattern_lists[0] if pattern_lists else []

    # Find patterns common to ALL modalities
    common_patterns = set(pattern_lists[0])
    for plist in pattern_lists[1:]:
        common_patterns = common_patterns.intersection(set(plist))

    if not common_patterns:
        # Fallback: if no common patterns, use patterns from first modality
        return pattern_lists[0][:15]

    # Rank by sum of positions across all modalities (lower = more frequent in all)
    ranked = []
    for pattern in common_patterns:
        total_rank = 0
        for plist in pattern_lists:
            if pattern in plist:
                total_rank += plist.index(pattern)
            else:
                total_rank += len(plist)
        ranked.append((pattern, total_rank))

    # Sort by combined rank (ascending = best first)
    ranked.sort(key=lambda x: x[1])

    return [p[0] for p in ranked]


def fbp_similarity(probe_patterns: list, enrolled_patterns: list, top_n: int = 15) -> float:
    """
    Compute similarity between probe and enrolled frequent patterns.
    Uses Jaccard-like measure: overlap of top-N patterns / top-N.
    Returns similarity as a float between 0.0 and 1.0.
    """
    probe_set = set(probe_patterns[:top_n])
    enrolled_set = set(enrolled_patterns[:top_n])

    if not probe_set or not enrolled_set:
        return 0.0

    overlap = len(probe_set.intersection(enrolled_set))
    union = len(probe_set.union(enrolled_set))

    if union == 0:
        return 0.0

    return overlap / union


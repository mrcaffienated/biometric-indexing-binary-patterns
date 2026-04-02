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
from collections import Counter

# ── TensorFlow CPU performance optimizations (helps Windows machines) ──
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # suppress TF info/warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # avoid oneDNN variability
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 4))

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


# Cache for the 1280→128 random projection (avoids regenerating on every call)
_projection_128 = None


def _get_projection_128(input_dim: int = 1280) -> np.ndarray:
    """Return a cached random projection matrix (input_dim → 128)."""
    global _projection_128
    if _projection_128 is None or _projection_128.shape[0] != input_dim:
        rng = np.random.RandomState(99)          # deterministic, isolated RNG
        _projection_128 = rng.randn(input_dim, 128)
    return _projection_128


def _extract_generic_embedding(image_path: str) -> np.ndarray:
    """Extract embedding from any image using MobileNetV2 → project to 128-D."""
    km = _load_keras_modules()

    model = _get_mobilenet()

    img = km["load_img"](image_path, target_size=(224, 224))
    img_array = km["img_to_array"](img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = km["preprocess_input"](img_array)

    features = model.predict(img_array, verbose=0).flatten()  # 1280-D

    # Project down to 128-D using a CACHED random projection
    proj = _get_projection_128(len(features))
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

    Optimized: uses integer packing + Counter instead of string conversions.
    """
    template = binary_template.astype(np.uint8)
    n = len(template)
    if n < window_length:
        return []

    # Pack each window into a single integer for fast hashing / counting
    # e.g. [0,1,1,0,1,0] → 0b011010 = 26
    powers = 1 << np.arange(window_length - 1, -1, -1, dtype=np.uint32)

    counts = Counter()
    for i in range(n - window_length + 1):
        key = int(template[i:i + window_length] @ powers)
        counts[key] += 1

    # Keep only patterns with frequency > 1, sorted descending
    frequent = sorted(
        ((pat, freq) for pat, freq in counts.items() if freq > 1),
        key=lambda x: x[1],
        reverse=True,
    )

    # Convert integer back to binary string of fixed width
    fmt = f"{{:0{window_length}b}}"
    return [fmt.format(pat) for pat, _ in frequent]


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


def per_trait_fbp_match(probe_per_mod: dict, enrolled_per_mod: dict,
                        threshold: float = 0.65, top_n: int = 15) -> dict:
    """
    Compare each biometric trait independently and require >= 2 matching traits.

    Args:
        probe_per_mod:    dict mapping modality name -> list of FBP pattern strings
        enrolled_per_mod: dict mapping modality name -> list of FBP pattern strings
        threshold:        similarity threshold for a single trait to count as matched
        top_n:            number of top patterns to consider per trait

    Returns:
        {
            "trait_scores":   {"face": 0.72, "iris": 0.68, ...},
            "trait_matched":  {"face": True,  "iris": True,  ...},
            "matched_count":  2,
            "total_compared": 3,
            "decision":       "Valid Match" | "Strong Match" | "Rejected",
            "reason":         human-readable explanation
        }
    """
    trait_scores = {}
    trait_matched = {}

    # Find traits present in BOTH probe and enrolled
    shared_traits = sorted(set(probe_per_mod.keys()) & set(enrolled_per_mod.keys()))

    for trait in shared_traits:
        score = fbp_similarity(
            probe_per_mod[trait][:top_n],
            enrolled_per_mod[trait][:top_n],
            top_n,
        )
        trait_scores[trait] = round(score, 4)
        trait_matched[trait] = score >= threshold

    matched_count = sum(1 for v in trait_matched.values() if v)
    total_compared = len(shared_traits)

    # ── Strict decision rule ──
    if matched_count >= 3:
        decision = "Strong Match"
        reason = f"All {matched_count} traits matched — strong biometric corroboration."
    elif matched_count == 2:
        decision = "Valid Match"
        reason = f"2 traits matched — minimum multi-trait requirement satisfied."
    elif matched_count == 1:
        decision = "Rejected"
        reason = "Only 1 trait matched — single-trait matches are NOT accepted."
    else:
        decision = "Rejected"
        reason = "No traits matched the similarity threshold."

    return {
        "trait_scores": trait_scores,
        "trait_matched": trait_matched,
        "matched_count": matched_count,
        "total_compared": total_compared,
        "decision": decision,
        "reason": reason,
    }


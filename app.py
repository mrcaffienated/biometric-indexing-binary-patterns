import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 4))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from feature_utils import (
    extract_face_embedding,
    extract_iris_embedding,
    extract_fingerprint_embedding,
    create_binary_template,
    hamming_similarity,
    get_or_create_projection_matrix,
    extract_frequent_patterns,
    rank_patterns_across_modalities,
    fbp_similarity,
    per_trait_fbp_match,
)
import json

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multi-Biometric Indexing System",
    page_icon="🔐",
    layout="centered",
)

st.markdown("""
<style>
    /* ── Clean dark theme overrides ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #e0e0e0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #8892b0;
        font-size: 0.95rem;
        margin-top: -0.5rem;
    }

    /* Cards */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    /* Status badges */
    .badge-ok {
        display: inline-block;
        background: #1b5e20;
        color: #a5d6a7;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-missing {
        display: inline-block;
        background: #4a1a1a;
        color: #ef9a9a;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-warn {
        display: inline-block;
        background: #4a3800;
        color: #ffe082;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Score display */
    .score-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .score-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #64ffda;
    }
    .score-label {
        font-size: 0.8rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Result banners */
    .result-pass {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-pass h2 { color: #fff; margin: 0; }
    .result-pass p { color: #c8e6c9; margin: 0.3rem 0 0 0; }

    .result-fail {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-fail h2 { color: #fff; margin: 0; }
    .result-fail p { color: #ffcdd2; margin: 0.3rem 0 0 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Divider */
    .clean-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    /* Binary template grid */
    .template-grid {
        display: grid;
        grid-template-columns: repeat(16, 1fr);
        gap: 2px;
        max-width: 100%;
        margin: 0.5rem 0;
    }
    .bit-1 {
        aspect-ratio: 1;
        background: #64ffda;
        border-radius: 3px;
    }
    .bit-0 {
        aspect-ratio: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

MODALITIES = ["face", "iris", "fingerprint"]
MODALITY_ICONS = {"face": "👤", "iris": "👁️", "fingerprint": "🖐️"}
MATCH_THRESHOLD = 0.65
FBP_WINDOW_LENGTH = 6
FBP_TOP_N = 15
BASE_DATASET = "lfw_subset"
FEATURES_DIR = "test_features"


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ═══════════════════════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="main-header">
        <h1>🔐 Multi-Biometric Indexing System</h1>
        <p>Privacy-Preserving Authentication with Frequent Binary Patterns</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Admin Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Login", use_container_width=True, type="primary"):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# MAIN APP (after login)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔐 Multi-Biometric Indexing System</h1>
    <p>Privacy-Preserving Authentication with Frequent Binary Patterns</p>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

tab1, tab2, tab3 = st.tabs(["👤 Enroll", "🧠 Train", "🔍 Test"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — ENROLLMENT
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enroll New Person")
    st.markdown("Upload or capture biometric data for **at least 2 of 3** modalities.")

    person_name = st.text_input("Person Name", placeholder="e.g. John Doe")

    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

    enroll_data = {}  # mod -> list of image bytes
    cols = st.columns(3)

    for i, mod in enumerate(MODALITIES):
        with cols[i]:
            st.markdown(f"**{MODALITY_ICONS[mod]} {mod.title()}**")

            enroll_data[mod] = []

            if mod == "face":
                input_method = st.radio(
                    "Input", ["📁 Upload", "📷 Camera"],
                    key=f"enroll_method_{mod}",
                    horizontal=True,
                )

                if input_method == "📁 Upload":
                    uploaded = st.file_uploader(
                        f"Upload {mod} images",
                        type=["jpg", "png", "jpeg"],
                        accept_multiple_files=True,
                        key=f"enroll_{mod}",
                    )
                    if uploaded:
                        enroll_data[mod] = uploaded
                else:
                    cam_img = st.camera_input(f"Capture {mod}", key=f"enroll_cam_{mod}")
                    if cam_img is not None:
                        enroll_data[mod] = [cam_img]
            else:
                uploaded = st.file_uploader(
                    f"Upload {mod} images",
                    type=["jpg", "png", "jpeg"],
                    accept_multiple_files=True,
                    key=f"enroll_{mod}",
                )
                if uploaded:
                    enroll_data[mod] = uploaded

    # Count how many modalities have data
    provided = [mod for mod in MODALITIES if enroll_data[mod]]

    if len(provided) > 0:
        st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
        status_html = ""
        for mod in MODALITIES:
            if enroll_data[mod]:
                status_html += f'<span class="badge-ok">✓ {mod.title()} ({len(enroll_data[mod])} files)</span> '
            else:
                status_html += f'<span class="badge-missing">✗ {mod.title()}</span> '
        st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)

    if st.button("Save Enrollment", use_container_width=True, type="primary"):

        if not person_name:
            st.warning("Please enter a person name.")
        elif len(provided) < 2:
            st.warning("Please provide data for **at least 2** modalities.")
        else:
            saved_count = 0
            for mod in provided:
                save_dir = os.path.join(BASE_DATASET, person_name, mod)
                os.makedirs(save_dir, exist_ok=True)

                for idx, img_file in enumerate(enroll_data[mod]):
                    filepath = os.path.join(save_dir, f"{person_name}_{mod}_{idx}.jpg")
                    with open(filepath, "wb") as f:
                        f.write(img_file.read())
                    saved_count += 1

            st.success(f"Enrolled **{person_name}** — {saved_count} images saved across {len(provided)} modalities.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING (FBP Indexing)
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Generate Frequent Binary Pattern Index")
    st.markdown("Extracts **frequent binary patterns** from each modality, ranks them across modalities, and stores a **compact index** per person.")

    if st.button("Start Training", use_container_width=True, type="primary"):

        if not os.path.exists(BASE_DATASET):
            st.error("No enrolled data found. Please enroll at least one person first.")
        else:
            persons = [
                p for p in os.listdir(BASE_DATASET)
                if os.path.isdir(os.path.join(BASE_DATASET, p))
            ]

            if len(persons) == 0:
                st.error("No enrolled persons found.")
            else:
                progress = st.progress(0, text="Loading AI models (first run may take a moment)...")

                extractors = {
                    "face": extract_face_embedding,
                    "iris": extract_iris_embedding,
                    "fingerprint": extract_fingerprint_embedding,
                }

                # ── Pre-load models so subsequent calls are fast ──
                # This is the slow part (TF model init); show it clearly
                try:
                    from feature_utils import _get_mobilenet, _load_keras_modules
                    _load_keras_modules()
                    _get_mobilenet()
                except Exception:
                    pass  # models will load lazily on first use
                progress.progress(0.05, text="Models loaded. Processing enrollments...")

                combined_dir = os.path.join(FEATURES_DIR, "combined_templates")
                os.makedirs(combined_dir, exist_ok=True)

                person_results = {}

                for pi, person in enumerate(persons):
                    progress.progress(
                        (pi) / len(persons),
                        text=f"Processing {person}..."
                    )

                    person_templates = {}
                    available_mods = []

                    for mod in MODALITIES:
                        mod_dir = os.path.join(BASE_DATASET, person, mod)
                        if not os.path.isdir(mod_dir):
                            continue

                        images = [
                            f for f in os.listdir(mod_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                        ]
                        if not images:
                            continue

                        R = get_or_create_projection_matrix(
                            f"projection_matrix_{mod}.npy"
                        )

                        img_path = os.path.join(mod_dir, images[0])
                        try:
                            embedding = extractors[mod](img_path)
                            binary_tmpl = create_binary_template(embedding, R)
                            person_templates[mod] = binary_tmpl
                            available_mods.append(mod)
                        except Exception as e:
                            st.caption(f"⚠️ Skipped {person}/{mod}: {e}")

                    if len(available_mods) >= 2:
                        # ── FBP: Extract frequent patterns per modality ──
                        pattern_lists = []
                        per_mod_patterns = {}
                        for mod in available_mods:
                            patterns = extract_frequent_patterns(
                                person_templates[mod], FBP_WINDOW_LENGTH
                            )
                            pattern_lists.append(patterns)
                            per_mod_patterns[mod] = patterns[:FBP_TOP_N]

                        # ── Rank patterns across modalities ──
                        ranked_patterns = rank_patterns_across_modalities(pattern_lists)
                        top_patterns = ranked_patterns[:FBP_TOP_N]

                        # ── Save FBP index ──
                        person_out = os.path.join(combined_dir, person)
                        os.makedirs(person_out, exist_ok=True)

                        fbp_data = {
                            "modalities": available_mods,
                            "ranked_patterns": top_patterns,
                            "per_modality_patterns": per_mod_patterns,
                            "window_length": FBP_WINDOW_LENGTH,
                        }
                        with open(os.path.join(person_out, "fbp_index.json"), "w") as f:
                            json.dump(fbp_data, f)

                        with open(os.path.join(person_out, "modalities.txt"), "w") as f:
                            f.write(",".join(available_mods))

                        person_results[person] = {
                            "mods": available_mods,
                            "num_patterns": len(top_patterns),
                            "top_patterns": top_patterns[:5],
                        }
                    else:
                        st.warning(f"⚠️ {person}: only {len(available_mods)} modality — need ≥2, skipped.")

                progress.progress(1.0, text="Complete!")

                st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
                st.markdown("#### Training Summary")

                for person, info in person_results.items():
                    mod_badges = " + ".join(
                        f"{MODALITY_ICONS[m]} {m.title()}" for m in info["mods"]
                    )
                    pattern_preview = ", ".join(info["top_patterns"][:3])
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{person}</strong><br>
                        <span style="color:#64ffda">{mod_badges}</span><br>
                        <span style="color:#8892b0">FBP Index: {info["num_patterns"]} frequent patterns (window={FBP_WINDOW_LENGTH})</span><br>
                        <span style="color:#5a6a8a;font-size:0.8rem">Top patterns: <code>{pattern_preview}…</code></span>
                    </div>
                    """, unsafe_allow_html=True)

                st.success(f"Training completed! {len(person_results)} FBP indexes generated.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TESTING (Strict Multi-Trait FBP Authentication)
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Biometric Authentication")
    st.markdown(
        "Provide **at least 2 of 3** biometric samples. "
        "Each trait is compared **independently** — a match is accepted only when **≥ 2 traits** pass the threshold."
    )

    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

    probe_paths = {}
    test_cols = st.columns(3)

    for i, mod in enumerate(MODALITIES):
        with test_cols[i]:
            st.markdown(f"**{MODALITY_ICONS[mod]} {mod.title()}**")

            if mod == "face":
                test_method = st.radio(
                    "Input", ["📁 Upload", "📷 Camera"],
                    key=f"test_method_{mod}",
                    horizontal=True,
                )

                if test_method == "📁 Upload":
                    uploaded = st.file_uploader(
                        f"Upload {mod} image",
                        type=["jpg", "png", "jpeg"],
                        key=f"test_{mod}",
                    )
                    if uploaded is not None:
                        temp_path = f"temp_probe_{mod}.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded.read())
                        probe_paths[mod] = temp_path
                        st.image(temp_path, use_container_width=True)
                else:
                    cam_img = st.camera_input(f"Capture {mod}", key=f"test_cam_{mod}")
                    if cam_img is not None:
                        temp_path = f"temp_probe_{mod}.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(cam_img.getbuffer())
                        probe_paths[mod] = temp_path
            else:
                uploaded = st.file_uploader(
                    f"Upload {mod} image",
                    type=["jpg", "png", "jpeg"],
                    key=f"test_{mod}",
                )
                if uploaded is not None:
                    temp_path = f"temp_probe_{mod}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.read())
                    probe_paths[mod] = temp_path
                    st.image(temp_path, use_container_width=True)

    provided_probes = sorted(probe_paths.keys(), key=lambda m: MODALITIES.index(m))

    if len(provided_probes) > 0:
        status_html = ""
        for mod in MODALITIES:
            if mod in probe_paths:
                status_html += f'<span class="badge-ok">✓ {mod.title()}</span> '
            else:
                status_html += f'<span class="badge-missing">✗ {mod.title()}</span> '
        st.markdown(f"**Provided:** {status_html}", unsafe_allow_html=True)

    if len(provided_probes) < 2:
        if len(provided_probes) == 1:
            st.markdown(
                '<span class="badge-warn">⚠ Upload at least 1 more modality</span>',
                unsafe_allow_html=True,
            )
    else:
        if st.button("🔍 Authenticate", use_container_width=True, type="primary"):

            st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

            extractors = {
                "face": extract_face_embedding,
                "iris": extract_iris_embedding,
                "fingerprint": extract_fingerprint_embedding,
            }

            combined_dir = os.path.join(FEATURES_DIR, "combined_templates")

            if not os.path.exists(combined_dir):
                st.error("No FBP indexes found. Run **Training** first.")
            else:
                with st.spinner("Extracting per-trait frequent binary patterns from probe..."):

                    # ── Build per-modality probe FBP patterns ──
                    probe_per_mod_patterns = {}   # mod -> list of pattern strings
                    probe_binary = {}

                    for mod in provided_probes:
                        proj_path = f"projection_matrix_{mod}.npy"
                        if not os.path.exists(proj_path):
                            st.error(f"No projection matrix for {mod}. Run training first.")
                            st.stop()

                        R = np.load(proj_path)
                        embedding = extractors[mod](probe_paths[mod])
                        binary_tmpl = create_binary_template(embedding, R)
                        probe_binary[mod] = binary_tmpl

                        # Extract frequent patterns for THIS modality
                        patterns = extract_frequent_patterns(binary_tmpl, FBP_WINDOW_LENGTH)
                        probe_per_mod_patterns[mod] = patterns[:FBP_TOP_N]

                    # ── Per-trait comparison against all enrolled persons ──
                    best_match_count = -1
                    best_avg_score = -1.0
                    best_person = "Unknown"
                    best_result = None
                    compared_count = 0

                    for person in os.listdir(combined_dir):
                        person_dir = os.path.join(combined_dir, person)
                        if not os.path.isdir(person_dir):
                            continue

                        fbp_file = os.path.join(person_dir, "fbp_index.json")
                        if not os.path.exists(fbp_file):
                            continue

                        with open(fbp_file) as f:
                            fbp_data = json.load(f)

                        enrolled_mods = fbp_data["modalities"]

                        # Check that probe modalities are covered by enrolled data
                        if not all(m in enrolled_mods for m in provided_probes):
                            continue

                        enrolled_per_mod = fbp_data.get("per_modality_patterns", {})
                        if not enrolled_per_mod:
                            continue

                        # ── Per-trait matching with strict >= 2 rule ──
                        result = per_trait_fbp_match(
                            probe_per_mod_patterns,
                            enrolled_per_mod,
                            threshold=MATCH_THRESHOLD,
                            top_n=FBP_TOP_N,
                        )
                        compared_count += 1

                        # Pick best person: most matched traits first, then highest avg score
                        avg_score = (
                            sum(result["trait_scores"].values()) / max(len(result["trait_scores"]), 1)
                        )
                        if (
                            result["matched_count"] > best_match_count
                            or (
                                result["matched_count"] == best_match_count
                                and avg_score > best_avg_score
                            )
                        ):
                            best_match_count = result["matched_count"]
                            best_avg_score = avg_score
                            best_person = person
                            best_result = result

                # ══════════════════════════════════════════════════
                # DISPLAY RESULTS
                # ══════════════════════════════════════════════════
                st.markdown("#### 🔐 Multi-Trait Authentication Result")

                if compared_count == 0 or best_result is None:
                    st.markdown("""
                    <div class="result-fail">
                        <h2>⚠️ NO MATCH POSSIBLE</h2>
                        <p>No enrolled person has a matching modality combination. Re-train first.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    accepted = best_result["decision"] in ("Valid Match", "Strong Match")

                    # ── Per-Trait Score Cards ──
                    st.markdown("##### Per-Trait Similarity Scores")
                    trait_cols = st.columns(len(best_result["trait_scores"]))
                    for idx, (trait, score) in enumerate(best_result["trait_scores"].items()):
                        with trait_cols[idx]:
                            score_pct = round(score * 100, 1)
                            is_pass = best_result["trait_matched"][trait]
                            color = "#64ffda" if is_pass else "#ef9a9a"
                            icon = MODALITY_ICONS.get(trait, "🔬")
                            status_badge = (
                                '<span class="badge-ok">✓ PASS</span>'
                                if is_pass
                                else '<span class="badge-missing">✗ FAIL</span>'
                            )
                            st.markdown(f"""
                            <div class="score-box">
                                <div style="font-size:1.5rem">{icon}</div>
                                <div class="score-label">{trait.upper()}</div>
                                <div class="score-value" style="color:{color}">{score_pct}%</div>
                                <div style="margin-top:0.3rem">{status_badge}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Matched Count Summary ──
                    matched_count = best_result["matched_count"]
                    total_compared_traits = best_result["total_compared"]
                    count_color = "#64ffda" if accepted else "#ef9a9a"
                    st.markdown(f"""
                    <div class="info-card" style="text-align:center">
                        <div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">
                            Traits Matched
                        </div>
                        <div style="font-size:2rem;font-weight:700;color:{count_color}">
                            {matched_count} / {total_compared_traits}
                        </div>
                        <div style="color:#8892b0;font-size:0.85rem">
                            Threshold per trait: {int(MATCH_THRESHOLD * 100)}% &nbsp;|&nbsp;
                            Minimum required: 2 traits
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Final Decision Banner ──
                    if accepted:
                        decision_label = best_result["decision"]
                        decision_emoji = "✅" if decision_label == "Valid Match" else "🟢"
                        st.markdown(f"""
                        <div class="result-pass">
                            <h2>{decision_emoji} AUTHENTICATED — {decision_label.upper()}</h2>
                            <p>Identity: <strong>{best_person}</strong> — {matched_count} of {total_compared_traits} traits matched</p>
                            <p style="font-size:0.85rem">{best_result["reason"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-fail">
                            <h2>❌ REJECTED</h2>
                            <p>Best candidate: <strong>{best_person}</strong> — only {matched_count} of {total_compared_traits} traits matched</p>
                            <p style="font-size:0.85rem">{best_result["reason"]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Technical Details ──
                    st.markdown(f"""
                    <div class="info-card">
                        <div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">
                            Technical Details
                        </div>
                        <div style="color:#e0e0e0;font-size:0.85rem;margin-top:0.5rem">
                            Method: Per-Trait Frequent Binary Pattern (FBP) Matching<br>
                            Strict Rule: Match accepted ONLY if ≥ 2 traits independently pass threshold<br>
                            Window length: {FBP_WINDOW_LENGTH} bits (2<sup>{FBP_WINDOW_LENGTH}</sup> = {2**FBP_WINDOW_LENGTH} possible patterns)<br>
                            Top patterns per trait: {FBP_TOP_N}<br>
                            Per-trait threshold: {int(MATCH_THRESHOLD * 100)}%<br>
                            Persons compared: {compared_count}<br>
                            Decision: <strong>{best_result["decision"]}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Decision Matrix Reference ──
                    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
                    st.markdown("##### 📋 Decision Matrix")
                    st.markdown("""
                    | Traits Matched | Decision |
                    |:-:|:-:|
                    | **0** | ❌ Rejected |
                    | **1** | ❌ Rejected — Single trait not accepted |
                    | **2** | ✅ Valid Match |
                    | **3** | ✅ Strong Match |

                    > ⚠️ **This system strictly rejects matches based on a single trait under any circumstance.**
                    """)

                    # ════════════════════════════════════════════════
                    # PER-TRAIT BINARY TEMPLATE PREVIEW
                    # ════════════════════════════════════════════════
                    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
                    st.markdown("#### 🧬 Per-Trait Binary Template Preview")

                    MAX_PATTERN_LEN = 30

                    for mod in provided_probes:
                        probe_pats = probe_per_mod_patterns.get(mod, [])
                        enrolled_pats = best_result["trait_scores"].get(mod) is not None

                        if not probe_pats:
                            continue

                        probe_concat = "".join(probe_pats)[:MAX_PATTERN_LEN]

                        icon = MODALITY_ICONS.get(mod, "🔬")
                        is_pass = best_result["trait_matched"].get(mod, False)
                        score_pct = round(best_result["trait_scores"].get(mod, 0) * 100, 1)
                        badge = (
                            f'<span class="badge-ok">✓ {score_pct}%</span>'
                            if is_pass
                            else f'<span class="badge-missing">✗ {score_pct}%</span>'
                        )

                        # Build colored bits
                        bits_html = ""
                        for ch in probe_concat:
                            color = "#64ffda" if ch == "1" else "rgba(255,255,255,0.2)"
                            bits_html += f'<span style="color:{color}">{ch}</span>'

                        st.markdown(f"""
                        <div class="info-card">
                            <div style="margin-bottom:0.3rem">
                                <strong>{icon} {mod.title()}</strong> &nbsp; {badge}
                            </div>
                            <div style="font-family:monospace;font-size:1rem;letter-spacing:2px">
                                {bits_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


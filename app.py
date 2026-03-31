import streamlit as st
import os
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
)

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
# TAB 2 — TRAINING
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Generate Combined Protected Templates")
    st.markdown("Creates **fused binary templates** by concatenating all enrolled modalities per person.")

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
                progress = st.progress(0, text="Initializing...")

                extractors = {
                    "face": extract_face_embedding,
                    "iris": extract_iris_embedding,
                    "fingerprint": extract_fingerprint_embedding,
                }

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
                        combined = np.concatenate(
                            [person_templates[mod] for mod in available_mods]
                        )

                        mod_key = "_".join(available_mods)
                        person_out = os.path.join(combined_dir, person)
                        os.makedirs(person_out, exist_ok=True)
                        np.save(
                            os.path.join(person_out, f"combined_{mod_key}.npy"),
                            combined
                        )
                        with open(os.path.join(person_out, "modalities.txt"), "w") as f:
                            f.write(",".join(available_mods))

                        person_results[person] = {
                            "mods": available_mods,
                            "bits": len(combined),
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
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{person}</strong><br>
                        <span style="color:#64ffda">{mod_badges}</span><br>
                        <span style="color:#8892b0">Combined template: {info["bits"]}-bit</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.success(f"Training completed! {len(person_results)} combined templates generated.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TESTING
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Biometric Authentication")
    st.markdown("Provide **at least 2 of 3** biometric samples. They are **fused into one combined pattern** for matching.")

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
                st.error("No combined templates found. Run **Training** first.")
            else:
                with st.spinner("Generating fused binary probe..."):

                    # ── Build combined probe template ──
                    probe_parts = {}
                    for mod in provided_probes:
                        proj_path = f"projection_matrix_{mod}.npy"
                        if not os.path.exists(proj_path):
                            st.error(f"No projection matrix for {mod}. Run training first.")
                            st.stop()

                        R = np.load(proj_path)
                        embedding = extractors[mod](probe_paths[mod])
                        probe_parts[mod] = create_binary_template(embedding, R)

                    combined_probe = np.concatenate(
                        [probe_parts[mod] for mod in provided_probes]
                    )

                    # ── Compare against enrolled combined templates ──
                    best_score = -1.0
                    best_person = "Unknown"
                    compared_count = 0

                    for person in os.listdir(combined_dir):
                        person_dir = os.path.join(combined_dir, person)
                        if not os.path.isdir(person_dir):
                            continue

                        mod_file = os.path.join(person_dir, "modalities.txt")
                        if not os.path.exists(mod_file):
                            continue

                        with open(mod_file) as f:
                            enrolled_mods = f.read().strip().split(",")

                        if not all(m in enrolled_mods for m in provided_probes):
                            continue

                        for tmpl_file in os.listdir(person_dir):
                            if not tmpl_file.endswith(".npy"):
                                continue

                            db_template = np.load(os.path.join(person_dir, tmpl_file))
                            enrolled_mod_key = tmpl_file.replace("combined_", "").replace(".npy", "")
                            enrolled_mod_list = enrolled_mod_key.split("_")

                            # Extract matching chunks in probe order
                            chunk_size = 128
                            extracted_parts = []
                            for pm in provided_probes:
                                if pm in enrolled_mod_list:
                                    idx = enrolled_mod_list.index(pm)
                                    extracted_parts.append(
                                        db_template[idx * chunk_size : (idx + 1) * chunk_size]
                                    )
                            if len(extracted_parts) == len(provided_probes):
                                db_subset = np.concatenate(extracted_parts)
                                score = hamming_similarity(combined_probe, db_subset)
                                compared_count += 1
                                if score > best_score:
                                    best_score = score
                                    best_person = person

                matched = best_score > MATCH_THRESHOLD

                # ── Display Result ──
                st.markdown("#### Combined Fusion Result")

                col_score, col_decision = st.columns([1, 2])

                with col_score:
                    score_pct = round(best_score * 100, 1) if best_score >= 0 else 0
                    color = "#64ffda" if matched else "#ef9a9a"
                    mod_label = " + ".join(
                        f"{MODALITY_ICONS[m]} {m.title()}" for m in provided_probes
                    )
                    st.markdown(f"""
                    <div class="score-box">
                        <div class="score-label">Fused Score</div>
                        <div class="score-value" style="color:{color}">{score_pct}%</div>
                        <div style="color:#8892b0;font-size:0.85rem">
                            {mod_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_decision:
                    if compared_count == 0:
                        st.markdown("""
                        <div class="result-fail">
                            <h2>⚠️ NO MATCH POSSIBLE</h2>
                            <p>No enrolled person has a matching modality combination. Re-train first.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif matched:
                        st.markdown(f"""
                        <div class="result-pass">
                            <h2>✅ AUTHENTICATED</h2>
                            <p>Identity: <strong>{best_person}</strong> — Fused similarity: {score_pct}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-fail">
                            <h2>❌ REJECTED</h2>
                            <p>Best match: {best_person} ({score_pct}%) — below {int(MATCH_THRESHOLD*100)}% threshold</p>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Technical Details ──
                st.markdown(f"""
                <div class="info-card">
                    <div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Technical Details</div>
                    <div style="color:#e0e0e0;font-size:0.85rem;margin-top:0.5rem">
                        Modalities fused: {len(provided_probes)} ({", ".join(provided_probes)})<br>
                        Combined template size: {len(combined_probe)}-bit ({len(provided_probes)} × 128)<br>
                        Persons compared: {compared_count}<br>
                        Threshold: {int(MATCH_THRESHOLD*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ════════════════════════════════════════════════════════
                # BINARY TEMPLATE VISUALIZER — COMBINED ONLY
                # ════════════════════════════════════════════════════════
                st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
                st.markdown("#### 🧬 Combined Binary Template")
                st.caption(
                    f"Fused {len(provided_probes)}-modality template "
                    f"({len(combined_probe)}-bit = {len(provided_probes)} × 128 bits)"
                )

                st.markdown("**🔗 Full Combined Pattern**")
                full_str = "".join(str(int(b)) for b in combined_probe)
                st.code(full_str, language=None)
                st.caption(f"Total: {int(np.sum(combined_probe))} ones / {len(combined_probe)} bits")


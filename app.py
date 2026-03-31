import streamlit as st
import os
import numpy as np
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
    st.markdown("Upload biometric data for **at least 2 of 3** modalities.")

    person_name = st.text_input("Person Name", placeholder="e.g. John Doe")

    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

    uploads = {}
    cols = st.columns(3)

    for i, mod in enumerate(MODALITIES):
        with cols[i]:
            st.markdown(f"**{MODALITY_ICONS[mod]} {mod.title()}**")
            uploads[mod] = st.file_uploader(
                f"Upload {mod} images",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                key=f"enroll_{mod}",
            )

    # Count how many modalities have uploads
    provided = [mod for mod in MODALITIES if uploads[mod]]

    if len(provided) > 0:
        st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
        status_html = ""
        for mod in MODALITIES:
            if uploads[mod]:
                status_html += f'<span class="badge-ok">✓ {mod.title()} ({len(uploads[mod])} files)</span> '
            else:
                status_html += f'<span class="badge-missing">✗ {mod.title()}</span> '
        st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)

    if st.button("Save Enrollment", use_container_width=True, type="primary"):

        if not person_name:
            st.warning("Please enter a person name.")
        elif len(provided) < 2:
            st.warning("Please upload images for **at least 2** modalities.")
        else:
            saved_count = 0
            for mod in provided:
                save_dir = os.path.join(BASE_DATASET, person_name, mod)
                os.makedirs(save_dir, exist_ok=True)

                for idx, img_file in enumerate(uploads[mod]):
                    filepath = os.path.join(save_dir, f"{person_name}_{mod}_{idx}.jpg")
                    with open(filepath, "wb") as f:
                        f.write(img_file.read())
                    saved_count += 1

            st.success(f"Enrolled **{person_name}** — {saved_count} images saved across {len(provided)} modalities.")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Generate Protected Templates")
    st.markdown("Creates cancelable binary templates for all enrolled modalities.")

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
                total_steps = 0
                completed = 0

                # Count total work
                for person in persons:
                    for mod in MODALITIES:
                        mod_dir = os.path.join(BASE_DATASET, person, mod)
                        if os.path.isdir(mod_dir):
                            total_steps += len([
                                f for f in os.listdir(mod_dir)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                            ])

                if total_steps == 0:
                    st.warning("No images found in enrolled data.")
                else:
                    extractors = {
                        "face": extract_face_embedding,
                        "iris": extract_iris_embedding,
                        "fingerprint": extract_fingerprint_embedding,
                    }

                    modality_stats = {mod: 0 for mod in MODALITIES}

                    for person in persons:
                        for mod in MODALITIES:
                            mod_dir = os.path.join(BASE_DATASET, person, mod)
                            if not os.path.isdir(mod_dir):
                                continue

                            # Get or create projection matrix for this modality
                            R = get_or_create_projection_matrix(
                                f"projection_matrix_{mod}.npy"
                            )

                            output_dir = os.path.join(FEATURES_DIR, f"Mix_{mod}_grp", person)
                            os.makedirs(output_dir, exist_ok=True)

                            for filename in os.listdir(mod_dir):
                                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                                    continue

                                img_path = os.path.join(mod_dir, filename)
                                try:
                                    embedding = extractors[mod](img_path)
                                    binary_tmpl = create_binary_template(embedding, R)

                                    save_name = filename.rsplit('.', 1)[0] + ".npy"
                                    np.save(os.path.join(output_dir, save_name), binary_tmpl)
                                    modality_stats[mod] += 1

                                except Exception as e:
                                    st.caption(f"⚠️ Skipped {filename}: {e}")

                                completed += 1
                                progress.progress(
                                    completed / total_steps,
                                    text=f"Processing {person} / {mod}..."
                                )

                    progress.progress(1.0, text="Complete!")

                    # Summary
                    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
                    st.markdown("#### Training Summary")
                    summary_cols = st.columns(3)
                    for i, mod in enumerate(MODALITIES):
                        with summary_cols[i]:
                            count = modality_stats[mod]
                            icon = MODALITY_ICONS[mod]
                            st.metric(f"{icon} {mod.title()}", f"{count} templates")

                    st.success("Training completed successfully!")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — TESTING
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Biometric Authentication")
    st.markdown("Provide **at least 2 of 3** biometric samples to authenticate.")

    st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)

    probe_paths = {}
    test_cols = st.columns(3)

    for i, mod in enumerate(MODALITIES):
        with test_cols[i]:
            st.markdown(f"**{MODALITY_ICONS[mod]} {mod.title()}**")
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

    provided_probes = list(probe_paths.keys())

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

            modality_results = {}

            with st.spinner("Analyzing biometric data..."):
                for mod in provided_probes:
                    proj_path = f"projection_matrix_{mod}.npy"
                    features_base = os.path.join(FEATURES_DIR, f"Mix_{mod}_grp")

                    if not os.path.exists(proj_path):
                        modality_results[mod] = {
                            "status": "error",
                            "message": "No projection matrix. Run training first.",
                        }
                        continue

                    if not os.path.exists(features_base):
                        modality_results[mod] = {
                            "status": "error",
                            "message": "No templates found. Run training first.",
                        }
                        continue

                    R = np.load(proj_path)
                    embedding = extractors[mod](probe_paths[mod])
                    binary_probe = create_binary_template(embedding, R)

                    best_score = -1.0
                    best_person = "Unknown"

                    for person in os.listdir(features_base):
                        person_dir = os.path.join(features_base, person)
                        if not os.path.isdir(person_dir):
                            continue
                        for tmpl_file in os.listdir(person_dir):
                            if not tmpl_file.endswith(".npy"):
                                continue
                            db_template = np.load(os.path.join(person_dir, tmpl_file))
                            score = hamming_similarity(binary_probe, db_template)
                            if score > best_score:
                                best_score = score
                                best_person = person

                    matched = best_score > MATCH_THRESHOLD
                    modality_results[mod] = {
                        "status": "match" if matched else "no_match",
                        "score": best_score,
                        "person": best_person,
                        "matched": matched,
                    }

            # ── Display per-modality results ──
            st.markdown("#### Per-Modality Results")
            result_cols = st.columns(len(provided_probes))

            matches_count = 0
            matched_person = "Unknown"

            for i, mod in enumerate(provided_probes):
                res = modality_results[mod]
                with result_cols[i]:
                    st.markdown(f"**{MODALITY_ICONS[mod]} {mod.title()}**")

                    if res["status"] == "error":
                        st.error(res["message"])
                    else:
                        score_pct = round(res["score"] * 100, 1)
                        color = "#64ffda" if res["matched"] else "#ef9a9a"
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-label">{mod.title()} Score</div>
                            <div class="score-value" style="color:{color}">{score_pct}%</div>
                            <div style="color:#8892b0;font-size:0.85rem">
                                Best match: {res["person"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if res["matched"]:
                            matches_count += 1
                            matched_person = res["person"]

            # ── Final Decision ──
            st.markdown("<hr class='clean-divider'>", unsafe_allow_html=True)
            st.markdown("#### Final Decision")

            col_decision, col_stats = st.columns([2, 1])

            with col_decision:
                if matches_count >= 2:
                    st.markdown(f"""
                    <div class="result-pass">
                        <h2>✅ AUTHENTICATED</h2>
                        <p>Identity: <strong>{matched_person}</strong> — {matches_count}/{len(provided_probes)} modalities matched</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fail">
                        <h2>❌ REJECTED</h2>
                        <p>Only {matches_count}/{len(provided_probes)} modalities matched (minimum 2 required)</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_stats:
                st.markdown(f"""
                <div class="info-card">
                    <div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Technical Details</div>
                    <div style="color:#e0e0e0;font-size:0.85rem;margin-top:0.5rem">
                        Modalities tested: {len(provided_probes)}<br>
                        Matches found: {matches_count}<br>
                        Threshold: {int(MATCH_THRESHOLD*100)}%<br>
                        Template size: 128-bit
                    </div>
                </div>
                """, unsafe_allow_html=True)
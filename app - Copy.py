import streamlit as st
import os
import numpy as np
from deepface import DeepFace
from pathlib import Path

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Biometric System", layout="wide")

# ===============================
# SESSION LOGIN STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===============================
# LOGIN PAGE
# ===============================
if not st.session_state.logged_in:

    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()

# ===============================
# MAIN SYSTEM
# ===============================
st.title("🔐 Privacy-Preserving Biometric System")

tab1, tab2, tab3 = st.tabs(["👤 Enroll", "🧠 Train", "🔍 Test"])

# ======================================================
# TAB 1 — ENROLLMENT
# ======================================================
with tab1:

    st.subheader("Enroll New Person")

    person_name = st.text_input("Enter Person Name")

    uploaded_images = st.file_uploader(
        "Upload 3-5 Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Save Enrollment"):

        if person_name and uploaded_images:

            save_path = os.path.join("lfw_subset", person_name)
            os.makedirs(save_path, exist_ok=True)

            for idx, img in enumerate(uploaded_images):
                with open(os.path.join(save_path, f"{person_name}_{idx}.jpg"), "wb") as f:
                    f.write(img.read())

            st.success("Enrollment completed successfully!")

        else:
            st.warning("Please provide name and images.")

# ======================================================
# TAB 2 — TRAINING
# ======================================================
with tab2:

    st.subheader("Generate Protected Templates")

    if st.button("Start Training"):

        st.info("Generating projection matrix and templates...")

        os.makedirs("test_features/Mix_face_grp", exist_ok=True)

        # Generate projection matrix if not exists
        if not os.path.exists("projection_matrix.npy"):
            st.write("Creating Projection Matrix...")
            dummy_embedding = np.random.rand(128)
            R = np.random.randn(128, 128)
            np.save("projection_matrix.npy", R)
        else:
            R = np.load("projection_matrix.npy")

        base_dataset = "lfw_subset"
        output_base = "test_features/Mix_face_grp"

        for person in os.listdir(base_dataset):

            person_path = os.path.join(base_dataset, person)

            if os.path.isdir(person_path):

                output_person = os.path.join(output_base, person)
                os.makedirs(output_person, exist_ok=True)

                for file in os.listdir(person_path):

                    img_path = os.path.join(person_path, file)

                    try:
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet",
                            enforce_detection=False
                        )[0]["embedding"]

                        embedding = np.array(embedding)

                        protected = np.dot(embedding, R)

                        threshold = np.median(protected)
                        binary_template = (protected > threshold).astype(int)

                        save_name = file.replace(".jpg", ".npy").replace(".png", ".npy")
                        np.save(os.path.join(output_person, save_name), binary_template)

                    except:
                        continue

        st.success("Training Completed Successfully!")

# ======================================================
# TAB 3 — TESTING
# ======================================================
with tab3:

    st.subheader("Test Biometric Authentication")

    uploaded_probe = st.file_uploader(
        "Upload Probe Image",
        type=["jpg", "png", "jpeg"],
        key="probe"
    )

    if uploaded_probe is not None:

        with open("temp_probe.jpg", "wb") as f:
            f.write(uploaded_probe.read())

        st.image("temp_probe.jpg", width=300)

        R = np.load("projection_matrix.npy")

        embedding = DeepFace.represent(
            img_path="temp_probe.jpg",
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        embedding = np.array(embedding)

        protected = np.dot(embedding, R)
        threshold = np.median(protected)
        binary_probe = (protected > threshold).astype(int)

        st.subheader("Binary Template Preview")
        st.code(binary_probe[:32])

        def hamming_similarity(a, b):
            return np.sum(a == b) / len(a)

        best_score = -1
        best_person = "Unknown"

        base_path = "test_features/Mix_face_grp"

        for person in os.listdir(base_path):

            person_path = os.path.join(base_path, person)

            if os.path.isdir(person_path):

                for file in os.listdir(person_path):

                    db_template = np.load(os.path.join(person_path, file))

                    score = hamming_similarity(binary_probe, db_template)

                    if score > best_score:
                        best_score = score
                        best_person = person

        threshold_decision = 0.65

        st.subheader("Recognition Result")
        st.write("Similarity Score:", round(float(best_score), 3))

        if best_score > threshold_decision:
            st.success(f"Genuine Match: {best_person}")
        else:
            st.error("No Enrolled Identity Found (Impostor)")
        st.divider()
        st.subheader("📊 Performance & Technical Details")

        colA, colB = st.columns(2)

        with colA:
            st.write("Embedding Dimension:", len(embedding))
            st.write("Projection Matrix Shape:", R.shape)
            st.write("Binary Template Length:", len(binary_probe))

        with colB:
            st.write("Number of 1s in Template:", int(np.sum(binary_probe)))
            st.write("Number of 0s in Template:",
                     int(len(binary_probe) - np.sum(binary_probe)))
            st.write("Decision Threshold:", threshold_decision)
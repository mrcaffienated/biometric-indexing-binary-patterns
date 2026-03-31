# 🔐 Multi-Biometric Indexing System
### Privacy-Preserving Authentication with Frequent Binary Patterns

---

## 📋 Prerequisites

- **Python 3.10** (required — other versions may cause compatibility issues)
- **pip** (comes with Python)
- **A web browser** (Chrome, Firefox, Safari, Edge)

### Check if Python 3.10 is installed:
```bash
python3 --version
```
If not installed, download from: https://www.python.org/downloads/release/python-3100/

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Extract the ZIP file
```bash
unzip biometric-indexing-binary-patterns.zip
cd "biomeric indexing binary patterns/indexing"
```

### Step 2: Create a virtual environment
```bash
python3 -m venv biometrics_env
```

### Step 3: Activate the virtual environment

**On macOS / Linux:**
```bash
source biometrics_env/bin/activate
```

**On Windows (Command Prompt):**
```cmd
biometrics_env\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
biometrics_env\Scripts\Activate.ps1
```

> ✅ You should see `(biometrics_env)` at the beginning of your terminal prompt.

### Step 4: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ This may take 5-10 minutes (downloads TensorFlow, DeepFace, etc.)

### Step 5: Run the application
```bash
streamlit run app.py
```

### Step 6: Open in browser
The terminal will show:
```
Local URL: http://localhost:8501
```
Open that URL in your browser.

### Step 7: Login
```
Username: admin
Password: 1234
```

---

## 📖 How to Use

### 1. Enroll (Tab 1)
- Enter a person's name
- Upload images for **at least 2 of 3** modalities:
  - 👤 **Face** — photo of face (upload or camera)
  - 👁️ **Iris** — iris scan image (upload only)
  - 🖐️ **Fingerprint** — fingerprint image (upload only)
- Click **Save Enrollment**

### 2. Train (Tab 2)
- Click **Start Training**
- The system generates **combined binary templates** by fusing all enrolled modalities per person
- Each person gets one concatenated binary template (256-bit for 2 mods, 384-bit for 3 mods)

### 3. Test (Tab 3)
- Upload biometric samples for **at least 2 modalities**
- Click **🔍 Authenticate**
- The system:
  1. Extracts embeddings from each provided modality
  2. Converts to binary templates (128-bit each)
  3. **Concatenates** them into one fused template
  4. Compares against enrolled combined templates using **Hamming similarity**
  5. Shows AUTHENTICATED (≥65%) or REJECTED (<65%)
- View the **Combined Binary Template** (1s and 0s) at the bottom

---

## 🔧 Troubleshooting

### "command not found: streamlit"
You forgot to activate the virtual environment. Run Step 3 again.

### "No module named 'tensorflow'"
Run `pip install -r requirements.txt` again inside the activated venv.

### "ImportError: libgthread / libGL"
On Linux, install system dependencies:
```bash
sudo apt-get install libglib2.0-0 libgl1 libsm6 libxext6
```

### App crashes with memory error
TensorFlow + DeepFace need ~2GB RAM. Close other heavy applications.

### Port already in use
Run on a different port:
```bash
streamlit run app.py --server.port 8503
```

---

## 📁 Project Structure
```
indexing/
├── app.py                  # Main Streamlit application
├── feature_utils.py        # Embedding extraction & binary template generation
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux system dependencies (for Streamlit Cloud)
├── runtime.txt             # Python version specification
├── README.md               # This file
├── lfw_subset/             # Enrolled biometric data (created after enrollment)
│   └── {person_name}/
│       ├── face/
│       ├── iris/
│       └── fingerprint/
└── test_features/          # Generated templates (created after training)
    └── combined_templates/
        └── {person_name}/
            ├── combined_{mods}.npy
            └── modalities.txt
```

---

## 📦 Dependencies
| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| deepface | Face embedding extraction (FaceNet) |
| tensorflow-cpu==2.13.1 | Deep learning backend |
| opencv-python-headless | Image processing |
| numpy | Numerical operations |
| Pillow | Image handling |
| matplotlib | Visualization |

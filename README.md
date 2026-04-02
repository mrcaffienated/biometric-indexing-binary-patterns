# 🔐 Multi-Biometric Indexing System
### Privacy-Preserving Authentication with Frequent Binary Patterns

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **Python** | **3.10** (required — other versions may cause TensorFlow / DeepFace issues) |
| **pip** | Comes bundled with Python |
| **RAM** | ≥ 2 GB free (TensorFlow + DeepFace models load into memory) |
| **Browser** | Chrome, Firefox, Safari, or Edge |

### Check if Python 3.10 is installed

**macOS / Linux:**
```bash
python3 --version
```

**Windows (Command Prompt or PowerShell):**
```cmd
python --version
```

> If the output is **not** `Python 3.10.x`, download it from:
> https://www.python.org/downloads/release/python-3100/
>
> ⚠️ **Windows users:** During installation tick **"Add Python to PATH"**.

---

## 🚀 First-Time Setup (One-Time Only)

Run these steps **once** when you first download the project.

### Step 1 — Open a terminal in the project folder

**macOS:** Open **Terminal** → `cd` into the project folder
```bash
cd ~/Desktop/biometric-indexing-binary-patterns
```

**Windows (Command Prompt):**
```cmd
cd %USERPROFILE%\Desktop\biometric-indexing-binary-patterns
```

**Windows (PowerShell):**
```powershell
cd $env:USERPROFILE\Desktop\biometric-indexing-binary-patterns
```

---

### Step 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv biometrics_env
```

**Windows:**
```cmd
python -m venv biometrics_env
```

---

### Step 3 — Activate the virtual environment

**macOS / Linux:**
```bash
source biometrics_env/bin/activate
```

**Windows (Command Prompt):**
```cmd
biometrics_env\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
biometrics_env\Scripts\Activate.ps1
```

> ✅ You should see `(biometrics_env)` at the beginning of your terminal prompt.

> ⚠️ **PowerShell note:** If you get an "execution policy" error, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

### Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ This may take **5–10 minutes** (downloads TensorFlow, DeepFace, OpenCV, etc.)

---

### Step 5 — Run the application

```bash
streamlit run app.py
```

The terminal will print:
```
Local URL: http://localhost:8501
```

Open that URL in your browser. 🎉

---

### Step 6 — Login

```
Username: admin
Password: 1234
```

---

## 🔄 Re-Opening the App (Every Time After First Setup)

Once the first-time setup is done you only need **3 quick commands** to start the app again.

### macOS / Linux

```bash
# 1. Open Terminal and go to the project folder
cd ~/Desktop/biometric-indexing-binary-patterns

# 2. Activate the virtual environment
source biometrics_env/bin/activate

# 3. Run the app
streamlit run app.py
```

### Windows — Command Prompt

```cmd
:: 1. Open Command Prompt and go to the project folder
cd %USERPROFILE%\Desktop\biometric-indexing-binary-patterns

:: 2. Activate the virtual environment
biometrics_env\Scripts\activate.bat

:: 3. Run the app
streamlit run app.py
```

### Windows — PowerShell

```powershell
# 1. Open PowerShell and go to the project folder
cd $env:USERPROFILE\Desktop\biometric-indexing-binary-patterns

# 2. Activate the virtual environment
biometrics_env\Scripts\Activate.ps1

# 3. Run the app
streamlit run app.py
```

> After running `streamlit run app.py`, open **http://localhost:8501** in your browser.
> To **stop** the server, press `Ctrl + C` in the terminal.

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
- The system extracts **Frequent Binary Patterns (FBP)** from each modality, ranks them across modalities, and stores a **compact FBP index** per person (top 15 most discriminative patterns)

### 3. Test (Tab 3)
- Upload biometric samples for **at least 2 modalities**
- Click **🔍 Authenticate**
- The system:
  1. Extracts embeddings from each provided modality
  2. Converts to binary templates (128-bit each)
  3. Extracts **Frequent Binary Patterns** using a sliding window
  4. Ranks patterns across modalities
  5. Compares against enrolled FBP indexes using **Jaccard similarity**
  6. Shows **AUTHENTICATED** (≥65%) or **REJECTED** (<65%)
- View the **Binary Template Preview** at the bottom

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `command not found: streamlit` | You forgot to activate the virtual environment. Run the activate command from Step 3. |
| `No module named 'tensorflow'` | Run `pip install -r requirements.txt` inside the activated venv. |
| `ImportError: libgthread / libGL` | **Linux only** — run `sudo apt-get install libglib2.0-0 libgl1 libsm6 libxext6` |
| App crashes with memory error | TensorFlow + DeepFace need ~2 GB RAM. Close other heavy applications. |
| Port `8501` already in use | Run on a different port: `streamlit run app.py --server.port 8503` |
| PowerShell "execution policy" error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `python3` not recognized (Windows) | Use `python` instead of `python3` on Windows. |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `deepface` | Face embedding extraction (FaceNet) |
| `tensorflow` | Deep learning backend |
| `tf-keras` | Keras compatibility layer |
| `opencv-python-headless` | Image processing |
| `numpy` | Numerical operations |
| `Pillow` | Image handling |
| `matplotlib` | Visualization |

---

## 📁 Project Structure

```
biometric-indexing-binary-patterns/
├── app.py                  # Main Streamlit application
├── feature_utils.py        # Embedding extraction, binary templates & FBP functions
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux system dependencies (for Streamlit Cloud)
├── runtime.txt             # Python version specification
├── README.md               # This file
├── lfw_subset/             # Enrolled biometric data (created after enrollment)
│   └── {person_name}/
│       ├── face/
│       ├── iris/
│       └── fingerprint/
└── test_features/          # Generated FBP indexes (created after training)
    └── combined_templates/
        └── {person_name}/
            ├── fbp_index.json
            └── modalities.txt
```

---

## ⚡ Quick Reference Card

| Action | Command |
|---|---|
| Activate venv (Mac) | `source biometrics_env/bin/activate` |
| Activate venv (Win CMD) | `biometrics_env\Scripts\activate.bat` |
| Activate venv (Win PS) | `biometrics_env\Scripts\Activate.ps1` |
| Start the app | `streamlit run app.py` |
| Stop the app | `Ctrl + C` in terminal |
| Open in browser | `http://localhost:8501` |
| Deactivate venv | `deactivate` |

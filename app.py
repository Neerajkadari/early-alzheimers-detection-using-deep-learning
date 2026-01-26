from flask import Flask, request, render_template_string, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import requests

# =====================================================
# MODEL LOADING (RENDER + GITHUB RELEASES)
# =====================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SCREENING_MODEL_FILE = "alzheimers_oasis_early_ad.keras"
V2_MODEL_FILE = "alzheimer_cnn_v2.keras"

SCREENING_MODEL_URL = (
    "https://github.com/Neerjakadari/early-alzheimers-detection-using-deep-learning/"
    "releases/download/v1.0.0/alzheimers_oasis_early_ad.keras"
)

V2_MODEL_URL = (
    "https://github.com/Neerjakadari/early-alzheimers-detection-using-deep-learning/"
    "releases/download/v1.0.0/alzheimer_cnn_v2.keras"
)

SCREENING_MODEL_PATH = os.path.join(MODEL_DIR, SCREENING_MODEL_FILE)
V2_MODEL_PATH = os.path.join(MODEL_DIR, V2_MODEL_FILE)

def download_if_missing(url, path):
    if os.path.exists(path):
        return
    print(f"⬇️ Downloading {os.path.basename(path)}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded {os.path.basename(path)}")

download_if_missing(SCREENING_MODEL_URL, SCREENING_MODEL_PATH)
download_if_missing(V2_MODEL_URL, V2_MODEL_PATH)

screening_model = tf.keras.models.load_model(SCREENING_MODEL_PATH)
v2_model = tf.keras.models.load_model(V2_MODEL_PATH)


# =====================================================
# SOUND PATHS (YOUR EXACT LOCATIONS)
# =====================================================
GOOD_SOUND_PATH = r"chime-alert-demo-309545.mp3"
BAD_SOUND_PATH  = r"wrong-answer-126515.mp3"

# =====================================================
# LOAD MODELS
# =====================================================
screening_model = tf.keras.models.load_model(SCREENING_MODEL_PATH)
v2_model = tf.keras.models.load_model(V2_MODEL_PATH)

IMG_SIZE = (180, 180)
CLASS_NAMES = ["AD", "CN", "MCI"]

# =====================================================
# BACKEND PREDICTION (LOGIC ONLY)
# =====================================================
def backend_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # ---- Stage 1 ----
    p1 = screening_model.predict(img, verbose=0)[0]
    cn_prob = p1[CLASS_NAMES.index("CN")]

    if cn_prob >= 0.6:
        return "🟢 Normal (No Cognitive Risk Detected)", "good"

    # ---- Stage 2 ----
    p2 = v2_model.predict(img, verbose=0)[0]
    ad_prob = p2[CLASS_NAMES.index("AD")]

    if ad_prob >= 0.5:
        return "🔴 Advanced Cognitive Impairment (AD Risk)", "bad"
    else:
        return "🟡 Early Cognitive Impairment (Early AD)", "bad"

# =====================================================
# SOUND ROUTES (SERVE LOCAL FILES SAFELY)
# =====================================================
@app.route("/sound/good")
def good_sound():
    return send_file(GOOD_SOUND_PATH, mimetype="audio/mpeg")

@app.route("/sound/bad")
def bad_sound():
    return send_file(BAD_SOUND_PATH, mimetype="audio/mpeg")

# =====================================================
# UI (INLINE HTML)
# =====================================================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Alzheimer’s MRI Screening</title>
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(120deg, #e3f2fd, #ffffff);
    padding: 40px;
}
.card {
    background: white;
    max-width: 780px;
    margin: auto;
    padding: 35px;
    border-radius: 14px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}
h1 { color: #0d47a1; }
button {
    background: #1976d2;
    color: white;
    border: none;
    padding: 12px 22px;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
}
button:hover { background: #0d47a1; }
.result {
    margin-top: 30px;
    padding: 20px;
    font-size: 22px;
    border-radius: 10px;
    text-align: center;
}
.good { background: #e8f5e9; color: #1b5e20; }
.bad  { background: #ffebee; color: #b71c1c; }
.footer {
    margin-top: 25px;
    font-size: 13px;
    color: #555;
}
</style>
</head>

<body>
<div class="card">
<h1>🧠 Alzheimer’s MRI Screening System</h1>
<p>AI-assisted screening tool for early cognitive risk detection</p>

<form method="post" enctype="multipart/form-data">
    <input type="file" name="image" required>
    <br><br>
    <button type="submit">Analyze MRI</button>
</form>

{% if result %}
<div class="result {{ sound }}">
    <strong>Final Result:</strong><br>
    {{ result }}
</div>

<audio autoplay>
{% if sound == "good" %}
<source src="/sound/good" type="audio/mpeg">
{% else %}
<source src="/sound/bad" type="audio/mpeg">
{% endif %}
</audio>
{% endif %}

<div class="footer">
⚠️ This system is for screening support only and not a medical diagnosis.
</div>
</div>
</body>
</html>
"""

# =====================================================
# MAIN ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    sound = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            result, sound = backend_predict(path)

    return render_template_string(HTML, result=result, sound=sound)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)



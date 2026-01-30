from flask import Flask, request, render_template_string, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import keras
import sys
from types import ModuleType
import inspect

# =====================================================
# KERAS COMPATIBILITY PATCH (DO NOT TOUCH)
# =====================================================
for name in dir(keras.layers):
    attr = getattr(keras.layers, name)
    if inspect.isclass(attr) and hasattr(attr, "__init__"):
        orig_init = attr.__init__
        def make_wrapper(o):
            def wrapper(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                return o(self, *args, **kwargs)
            return wrapper
        attr.__init__ = make_wrapper(orig_init)

from keras.src.saving import serialization_lib
_original_deserialize = serialization_lib.deserialize_keras_object

def patched_deserialize(config, custom_objects=None, **kwargs):
    def clean(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for v in obj.values():
                clean(v)
        elif isinstance(obj, list):
            for i in obj:
                clean(i)
    if isinstance(config, dict):
        clean(config)
    return _original_deserialize(config, custom_objects, **kwargs)

serialization_lib.deserialize_keras_object = patched_deserialize

# =====================================================
# FAKE keras.src (FOR OLD MODELS)
# =====================================================
if "keras.src" not in sys.modules:
    keras_src = ModuleType("keras.src")
    keras_src_models = ModuleType("keras.src.models")
    keras_src_models.functional = keras.models
    keras_src.models = keras_src_models
    keras_src.layers = keras.layers
    keras_src.initializers = keras.initializers
    keras_src.optimizers = keras.optimizers

    sys.modules["keras.src"] = keras_src
    sys.modules["keras.src.models"] = keras_src_models
    sys.modules["keras.src.models.functional"] = keras.models
    sys.modules["keras.src.layers"] = keras.layers
    sys.modules["keras.src.initializers"] = keras.initializers
    sys.modules["keras.src.optimizers"] = keras.optimizers

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (180, 180)

# =====================================================
# MODEL PATHS
# =====================================================
SCREENING_MODEL_PATH = "alzheimers_oasis_early_ad.h5"
V2_MODEL_PATH = "alzheimer_cnn_v2.h5"

# =====================================================
# IMAGE (AS-IS, OUTSIDE FOLDERS)
# =====================================================
REFERENCE_IMAGE_PATH = "0b9504f9-1bac-47b9-a6c1-e54887aa2980.jpg"

# =====================================================
# SOUND FILES
# =====================================================
GOOD_SOUND_PATH = "chime-alert-demo-309545.mp3"
BAD_SOUND_PATH  = "wrong-answer-126515.mp3"

# =====================================================
# LOAD MODELS
# =====================================================
screening_model = tf.keras.models.load_model(SCREENING_MODEL_PATH, compile=False)
v2_model = tf.keras.models.load_model(V2_MODEL_PATH, compile=False)

screening_model.compile(optimizer="adam", loss="categorical_crossentropy")
v2_model.compile(optimizer="adam", loss="categorical_crossentropy")

# =====================================================
# BACKEND PREDICTION (STABLE & FINAL)
# =====================================================
def backend_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # ==============================
    # Stage 1: Screening model
    # Order: [AD, CN, MCI]
    # ==============================
    p1 = screening_model.predict(img, verbose=0)[0]
    p1_ad  = float(p1[0])
    p1_cn  = float(p1[1])
    p1_mci = float(p1[2])

    # ==============================
    # Stage 2: AD model
    # ==============================
    p2 = v2_model.predict(img, verbose=0)[0]
    p2_ad = float(p2[0])

    # =================================================
    # 🟢 NORMAL — AD-ABSENCE BASED (ONLY CHANGE)
    # =================================================
    if (
        p2_ad <= 0.18 and        # almost no AD evidence
        p1_ad <= 0.20            # screening also sees no AD
    ):
        return "🟢 Brain Scan Appears Normal", "good"

    # =================================================
    # 🔴 AD — UNCHANGED
    # =================================================
    if p2_ad >= 0.62:
        return "🔴 Alzheimer’s Disease Detected", "bad"

    # =================================================
    # 🟡 EARLY AD — UNCHANGED
    # =================================================
    if 0.18 < p2_ad < 0.62:
        return "🟡 Early Alzheimer’s (Mild Cognitive Impairment)", "bad"

    # =================================================
    # 🔒 SAFETY
    # =================================================
    return "🟡 Early Alzheimer’s (Mild Cognitive Impairment)", "bad"


# =====================================================
# ROUTES FOR IMAGE & SOUND (AS-IS FILES)
# =====================================================
@app.route("/reference-image")
def reference_image():
    return send_file(REFERENCE_IMAGE_PATH, mimetype="image/jpeg")

@app.route("/sound/good")
def good_sound():
    return send_file(GOOD_SOUND_PATH, mimetype="audio/mpeg")

@app.route("/sound/bad")
def bad_sound():
    return send_file(BAD_SOUND_PATH, mimetype="audio/mpeg")

# =====================================================
# UI (CLEAN, HOSPITAL STYLE)
# =====================================================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Early Alzheimer’s MRI Screening</title>
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
    padding: 22px;
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
img {
    width: 120px;          /* very small */
    max-width: 120px;
    height: auto;
    display: block;
    margin: 10px auto;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

</style>
</head>
<body>
<div class="card">

<h1>🧠 Alzheimer’s MRI Screening System</h1>
<p>AI-assisted screening tool for cognitive risk detection</p>

<img src="/reference-image" alt="Reference MRI Image">

<form method="post" enctype="multipart/form-data">
    <input type="file" name="image" required>
    <br><br>
    <button type="submit">Analyze MRI</button>
</form>

{% if result %}
<div class="result {{ sound }}">
    <strong>Final Result:</strong><br><br>
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

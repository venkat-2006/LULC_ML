# app.py
import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ── Load all saved models ─────────────────────────────
pca        = joblib.load("pca_model.pkl")
model      = joblib.load("lulc_best_model.pkl")
le         = joblib.load("label_encoder.pkl")
results    = joblib.load("all_results.pkl")
best_name  = joblib.load("best_model_name.pkl")

IMAGE_SIZE = 64

# Class display info (emoji + color per class)
CLASS_INFO = {
    "annualcrop"           : {"emoji": "🌾", "color": "#d97706"},
    "forest"               : {"emoji": "🌲", "color": "#16a34a"},
    "herbaceousvegetation" : {"emoji": "🌿", "color": "#65a30d"},
    "highway"              : {"emoji": "🛣️",  "color": "#6b7280"},
    "industrial"           : {"emoji": "🏭", "color": "#7c3aed"},
    "pasture"              : {"emoji": "🐄", "color": "#84cc16"},
    "permanentcrop"        : {"emoji": "🍇", "color": "#b45309"},
    "residential"          : {"emoji": "🏘️",  "color": "#3b82f6"},
    "river"                : {"emoji": "🏞️",  "color": "#0ea5e9"},
    "sealake"              : {"emoji": "🌊", "color": "#0c4a6e"},
}

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = image.flatten().reshape(1, -1)   # (1, 12288)
    return pca.transform(image)              # (1, 200)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    # Preprocess & predict
    features       = preprocess_image(save_path)
    pred_code      = model.predict(features)[0]
    pred_label     = le.inverse_transform([pred_code])[0]
    pred_label_key = pred_label.lower().replace(" ", "")
    info           = CLASS_INFO.get(pred_label_key, {"emoji": "🌍", "color": "#38bdf8"})

    os.remove(save_path)

    # Build model comparison list (sorted by accuracy)
    model_table = sorted(
        [{"name": name,
          "accuracy" : r["accuracy"],
          "precision": r["precision"],
          "recall"   : r["recall"],
          "f1"       : r["f1"],
          "is_best"  : (name == best_name)}
         for name, r in results.items()],
        key=lambda x: x["accuracy"],
        reverse=True
    )

    return jsonify({
        "prediction" : pred_label,
        "emoji"      : info["emoji"],
        "color"      : info["color"],
        "best_model" : best_name,
        "model_table": model_table,
    })

if __name__ == "__main__":
    app.run(debug=True)
# app.py
import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

scaler    = joblib.load("scaler.pkl")
pca       = joblib.load("pca_model.pkl")
model     = joblib.load("lulc_best_model.pkl")
le        = joblib.load("label_encoder.pkl")
results   = joblib.load("all_results.pkl")
best_name = joblib.load("best_model_name.pkl")

IMAGE_SIZE = 64

CLASS_INFO = {
    "annualcrop"           : {"emoji": "🌾", "color": "#d97706",
                               "desc": "Seasonal farmland"},
    "forest"               : {"emoji": "🌲", "color": "#16a34a",
                               "desc": "Dense tree cover"},
    "herbaceousvegetation" : {"emoji": "🌿", "color": "#65a30d",
                               "desc": "Grass & shrubs"},
    "highway"              : {"emoji": "🛣️",  "color": "#6b7280",
                               "desc": "Roads & highways"},
    "industrial"           : {"emoji": "🏭", "color": "#7c3aed",
                               "desc": "Factories & warehouses"},
    "pasture"              : {"emoji": "🐄", "color": "#84cc16",
                               "desc": "Grazing land"},
    "permanentcrop"        : {"emoji": "🍇", "color": "#b45309",
                               "desc": "Orchards & vineyards"},
    "residential"          : {"emoji": "🏘️",  "color": "#3b82f6",
                               "desc": "Housing areas"},
    "river"                : {"emoji": "🏞️",  "color": "#0ea5e9",
                               "desc": "Rivers & streams"},
    "sealake"              : {"emoji": "🌊", "color": "#0c4a6e",
                               "desc": "Ocean & lakes"},
}

def extract_features(image):
    features = []

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(3):
        bins = 32 if i == 0 else 16
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.append([
        np.mean(gray),
        np.std(gray),
        np.mean(np.abs(np.diff(gray.astype(float)))),
    ])

    gray_f = gray.astype(np.float32)
    lbp    = np.zeros_like(gray_f)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),
                   (0,1),(1,-1),(1,0),(1,1)]:
        shifted = np.roll(np.roll(gray_f, dy, axis=0), dx, axis=1)
        lbp    += (gray_f > shifted).astype(np.float32)
    lbp_hist, _ = np.histogram(lbp, bins=9, range=(0, 9))
    lbp_hist     = lbp_hist / (lbp_hist.sum() + 1e-6)
    features.append(lbp_hist)

    sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_hist, _ = np.histogram(magnitude, bins=16, range=(0, 300))
    edge_hist     = edge_hist / (edge_hist.sum() + 1e-6)
    features.append(edge_hist)

    small = cv2.resize(image, (16, 16)) / 255.0
    features.append(small.flatten())

    return np.concatenate([np.array(f).flatten() for f in features])


def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    feat  = extract_features(image).reshape(1, -1)
    feat  = scaler.transform(feat)
    feat  = pca.transform(feat)
    return feat


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

    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    try:
        features   = preprocess_image(save_path)
        pred_code  = model.predict(features)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        key        = pred_label.lower().replace(" ", "")
        info       = CLASS_INFO.get(
            key, {"emoji": "🌍", "color": "#38bdf8",
                  "desc": "Land cover"})
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    model_table = sorted(
        [{"name"     : name,
          "accuracy" : round(r["accuracy"]  * 100, 1),
          "precision": round(r["precision"] * 100, 1),
          "recall"   : round(r["recall"]    * 100, 1),
          "f1"       : round(r["f1"]        * 100, 1),
          "is_best"  : (name == best_name)}
         for name, r in results.items()],
        key=lambda x: x["accuracy"], reverse=True
    )

    return jsonify({
        "prediction" : pred_label,
        "emoji"      : info["emoji"],
        "color"      : info["color"],
        "desc"       : info["desc"],
        "best_model" : best_name,
        "model_table": model_table,
    })


if __name__ == "__main__":
    app.run(debug=True)
# # app.py
# import os
# import cv2
# import numpy as np
# import joblib
# from flask import Flask, request, render_template, jsonify

# app = Flask(__name__)

# scaler    = joblib.load("scaler.pkl")
# pca       = joblib.load("pca_model.pkl")
# model     = joblib.load("lulc_best_model.pkl")
# le        = joblib.load("label_encoder.pkl")
# results   = joblib.load("all_results.pkl")
# best_name = joblib.load("best_model_name.pkl")

# IMAGE_SIZE = 64

# CLASS_INFO = {
#     "annualcrop"           : {"emoji": "🌾", "color": "#d97706",
#                                "desc": "Seasonal farmland"},
#     "forest"               : {"emoji": "🌲", "color": "#16a34a",
#                                "desc": "Dense tree cover"},
#     "herbaceousvegetation" : {"emoji": "🌿", "color": "#65a30d",
#                                "desc": "Grass & shrubs"},
#     "highway"              : {"emoji": "🛣️",  "color": "#6b7280",
#                                "desc": "Roads & highways"},
#     "industrial"           : {"emoji": "🏭", "color": "#7c3aed",
#                                "desc": "Factories & warehouses"},
#     "pasture"              : {"emoji": "🐄", "color": "#84cc16",
#                                "desc": "Grazing land"},
#     "permanentcrop"        : {"emoji": "🍇", "color": "#b45309",
#                                "desc": "Orchards & vineyards"},
#     "residential"          : {"emoji": "🏘️",  "color": "#3b82f6",
#                                "desc": "Housing areas"},
#     "river"                : {"emoji": "🏞️",  "color": "#0ea5e9",
#                                "desc": "Rivers & streams"},
#     "sealake"              : {"emoji": "🌊", "color": "#0c4a6e",
#                                "desc": "Ocean & lakes"},
# }

# def extract_features(image):
#     features = []

#     for i in range(3):
#         hist = cv2.calcHist([image], [i], None, [32], [0, 256])
#         hist = cv2.normalize(hist, hist).flatten()
#         features.append(hist)

#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     for i in range(3):
#         bins = 32 if i == 0 else 16
#         hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
#         hist = cv2.normalize(hist, hist).flatten()
#         features.append(hist)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     features.append([
#         np.mean(gray),
#         np.std(gray),
#         np.mean(np.abs(np.diff(gray.astype(float)))),
#     ])

#     gray_f = gray.astype(np.float32)
#     lbp    = np.zeros_like(gray_f)
#     for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),
#                    (0,1),(1,-1),(1,0),(1,1)]:
#         shifted = np.roll(np.roll(gray_f, dy, axis=0), dx, axis=1)
#         lbp    += (gray_f > shifted).astype(np.float32)
#     lbp_hist, _ = np.histogram(lbp, bins=9, range=(0, 9))
#     lbp_hist     = lbp_hist / (lbp_hist.sum() + 1e-6)
#     features.append(lbp_hist)

#     sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     magnitude = np.sqrt(sobelx**2 + sobely**2)
#     edge_hist, _ = np.histogram(magnitude, bins=16, range=(0, 300))
#     edge_hist     = edge_hist / (edge_hist.sum() + 1e-6)
#     features.append(edge_hist)

#     small = cv2.resize(image, (16, 16)) / 255.0
#     features.append(small.flatten())

#     return np.concatenate([np.array(f).flatten() for f in features])


# def preprocess_image(path):
#     image = cv2.imread(path)
#     image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
#     feat  = extract_features(image).reshape(1, -1)
#     feat  = scaler.transform(feat)
#     feat  = pca.transform(feat)
#     return feat


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     os.makedirs("uploads", exist_ok=True)
#     save_path = os.path.join("uploads", file.filename)
#     file.save(save_path)

#     try:
#         features   = preprocess_image(save_path)
#         pred_code  = model.predict(features)[0]
#         pred_label = le.inverse_transform([pred_code])[0]
#         key        = pred_label.lower().replace(" ", "")
#         info       = CLASS_INFO.get(
#             key, {"emoji": "🌍", "color": "#38bdf8",
#                   "desc": "Land cover"})
#     finally:
#         if os.path.exists(save_path):
#             os.remove(save_path)

#     model_table = sorted(
#         [{"name"     : name,
#           "accuracy" : round(r["accuracy"]  * 100, 1),
#           "precision": round(r["precision"] * 100, 1),
#           "recall"   : round(r["recall"]    * 100, 1),
#           "f1"       : round(r["f1"]        * 100, 1),
#           "is_best"  : (name == best_name)}
#          for name, r in results.items()],
#         key=lambda x: x["accuracy"], reverse=True
#     )

#     return jsonify({
#         "prediction" : pred_label,
#         "emoji"      : info["emoji"],
#         "color"      : info["color"],
#         "desc"       : info["desc"],
#         "best_model" : best_name,
#         "model_table": model_table,
#     })


# if __name__ == "__main__":
#     app.run(debug=True)
# app.py
import os
import cv2
import uuid
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify, send_file
from preprocess import extract_features

app = Flask(__name__)

# ─────────────────────────────────────────
# 🔹 Load all saved objects
# ─────────────────────────────────────────
scaler    = joblib.load("scaler.pkl")
pca       = joblib.load("pca_model.pkl")
model     = joblib.load("lulc_best_model.pkl")
le        = joblib.load("label_encoder.pkl")
results   = joblib.load("all_results.pkl")
best_name = joblib.load("best_model_name.pkl")

# 🔥 Load all models
ALL_MODELS = {}
for name in results.keys():
    filename = name.replace(" ", "_").lower() + ".pkl"
    try:
        ALL_MODELS[name] = joblib.load(filename)
    except Exception as e:
        print(f"⚠️ Could not load {filename}: {e}")

IMAGE_SIZE = 64

# ─────────────────────────────────────────
# 🔹 Class Info
# ─────────────────────────────────────────
CLASS_INFO = {
    "annualcrop": {"emoji": "🌾", "color": "#d97706", "desc": "Seasonal farmland"},
    "forest": {"emoji": "🌲", "color": "#16a34a", "desc": "Dense tree cover"},
    "herbaceousvegetation": {"emoji": "🌿", "color": "#65a30d", "desc": "Grass & shrubs"},
    "highway": {"emoji": "🛣️", "color": "#6b7280", "desc": "Roads & highways"},
    "industrial": {"emoji": "🏭", "color": "#7c3aed", "desc": "Factories & warehouses"},
    "pasture": {"emoji": "🐄", "color": "#84cc16", "desc": "Grazing land"},
    "permanentcrop": {"emoji": "🍇", "color": "#b45309", "desc": "Orchards & vineyards"},
    "residential": {"emoji": "🏘️", "color": "#3b82f6", "desc": "Housing areas"},
    "river": {"emoji": "🏞️", "color": "#0ea5e9", "desc": "Rivers & streams"},
    "sealake": {"emoji": "🌊", "color": "#0c4a6e", "desc": "Ocean & lakes"},
}

# ─────────────────────────────────────────
# 🔹 Preprocess Function
# ─────────────────────────────────────────
def preprocess_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError("Invalid image file")

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    feat  = extract_features(image)

    feat  = scaler.transform(feat.reshape(1, -1))
    feat  = pca.transform(feat)

    return feat


@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────
# 🔥 PREDICT ROUTE
# ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs("uploads", exist_ok=True)

    # ✅ safe filename
    filename = str(uuid.uuid4()) + ".jpg"
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    try:
        # 🔹 Preprocess
        features = preprocess_image(save_path)

        # 🔥 Best model prediction
        probs = model.predict_proba(features)[0]
        pred_code  = np.argmax(probs)
        pred_label = le.inverse_transform([pred_code])[0]
        confidence = round(probs[pred_code] * 100, 2)

        # 🔥 Top 3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = []
        for i in top3_idx:
            label = le.inverse_transform([i])[0]
            top3.append({
                "label": label,
                "confidence": round(probs[i] * 100, 2)
            })

        # 🔥 All model predictions
        model_predictions = {}
        for name, m in ALL_MODELS.items():
            try:
                pred = m.predict(features)[0]
                label = le.inverse_transform([pred])[0]
                model_predictions[name] = label
            except Exception:
                model_predictions[name] = "N/A"

        # 🔥 Agreement %
        pred_values = list(model_predictions.values())
        valid_preds = [p for p in pred_values if p != "N/A"]

        if len(valid_preds) > 0:
            agreement = valid_preds.count(pred_label) / len(valid_preds)
        else:
            agreement = 0

        agreement = round(agreement * 100, 2)

        # 🔹 Class info
        key = pred_label.lower().replace(" ", "")
        info = CLASS_INFO.get(
            key,
            {"emoji": "🌍", "color": "#38bdf8", "desc": "Land cover"}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    # 🔹 Model table
    model_table = sorted(
        [{
            "name": name,
            "accuracy": round(r["accuracy"] * 100, 1),
            "precision": round(r["precision"] * 100, 1),
            "recall": round(r["recall"] * 100, 1),
            "f1": round(r["f1"] * 100, 1),
            "is_best": (name == best_name)
        } for name, r in results.items()],
        key=lambda x: x["accuracy"],
        reverse=True
    )

    # 🔥 Final response
    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
        "confidence_raw": float(probs[pred_code]),
        "top3": top3,
        "agreement": agreement,
        "model_predictions": dict(sorted(model_predictions.items())),
        "emoji": info["emoji"],
        "color": info["color"],
        "desc": info["desc"],
        "best_model": best_name,
        "model_table": model_table,
        "total_models": len(results),
        "feature_size": int(pca.n_components_),
    })


# ─────────────────────────────────────────
# 🔥 CHANGE DETECTION
# ─────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Upload two images"}), 400

    os.makedirs("uploads", exist_ok=True)

    path1 = os.path.join("uploads", str(uuid.uuid4()) + ".jpg")
    path2 = os.path.join("uploads", str(uuid.uuid4()) + ".jpg")

    request.files["image1"].save(path1)
    request.files["image2"].save(path2)

    try:
        f1 = preprocess_image(path1)
        f2 = preprocess_image(path2)

        p1 = le.inverse_transform([model.predict(f1)[0]])[0]
        p2 = le.inverse_transform([model.predict(f2)[0]])[0]

        change = "No Change"
        impact = "Stable land cover"

        if p1 != p2:
            change = f"{p1} → {p2}"

            if p1 == "Forest" and p2 == "Residential":
                impact = "⚠️ Deforestation / Urbanization"
            elif p2 == "Industrial":
                impact = "⚠️ Industrial expansion"
            else:
                impact = "Land transformation detected"

        return jsonify({
            "image1": p1,
            "image2": p2,
            "change": change,
            "impact": impact
        })

    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)


# ─────────────────────────────────────────
# 🔥 CONFUSION MATRIX VIEW
# ─────────────────────────────────────────
@app.route("/confusion")
def confusion():
    return send_file("confusion_matrix.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
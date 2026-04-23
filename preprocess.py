# preprocess.py
import os
import cv2
import numpy as np

DATASET_PATH = "dataset"
IMAGE_SIZE   = 64

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


# ─────────────────────────────────────────
# 🔹 Feature Extraction
# ─────────────────────────────────────────
def extract_features(image):
    if image is None:
        raise ValueError("Invalid image passed to extract_features")

    features = []

    # ── 1. Color histograms (32 bins × 3 channels = 96) ──
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    # ── 2. HSV histograms ──
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(3):
        bins = 32 if i == 0 else 16
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    # ── 3. Grayscale texture stats ──
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.append([
        np.mean(gray),
        np.std(gray),
        np.mean(np.abs(np.diff(gray.astype(float))))
    ])

    # ── 4. LBP-like texture ──
    gray_f = gray.astype(np.float32)
    lbp    = np.zeros_like(gray_f)

    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        shifted = np.roll(np.roll(gray_f, dy, axis=0), dx, axis=1)
        lbp += (gray_f > shifted).astype(np.float32)

    lbp_hist, _ = np.histogram(lbp, bins=9, range=(0, 9))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    features.append(lbp_hist)

    # ── 5. Sobel edge histogram ──
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    edge_hist, _ = np.histogram(magnitude, bins=16, range=(0, 300))
    edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
    features.append(edge_hist)

    # ── 6. Spatial features ──
    small = cv2.resize(image, (16, 16)) / 255.0
    features.append(small.flatten())

    # ✅ final vector
    return np.concatenate([
        np.array(f, dtype=np.float32).flatten()
        for f in features
    ])


# ─────────────────────────────────────────
# 🔹 Dataset Loader
# ─────────────────────────────────────────
def load_dataset():
    feature_list = []
    labels       = []
    skipped      = 0

    print("=" * 52)
    print("  Loading EuroSAT Dataset")
    print("=" * 52)

    for class_name in CLASSES:
        class_folder = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_folder):
            print(f"  ⚠️ WARNING: {class_name} folder missing")
            continue

        files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        loaded = 0

        for fname in files:
            fpath = os.path.join(class_folder, fname)

            image = cv2.imread(fpath)
            if image is None:
                skipped += 1
                continue

            try:
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                feat  = extract_features(image)

                feature_list.append(feat)
                labels.append(class_name)
                loaded += 1

            except Exception:
                skipped += 1

        print(f"  {class_name:<28}: {loaded} images")

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(labels)

    print("=" * 52)
    print(f"  Total images   : {X.shape[0]}")
    print(f"  Feature size   : {X.shape[1]}")
    print(f"  Skipped        : {skipped}")
    print("=" * 52)

    return X, y


# ─────────────────────────────────────────
# 🔹 Debug run
# ─────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_dataset()
    print(f"\nSample feature vector size: {X.shape[1]}")
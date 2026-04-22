# preprocess.py
import os
import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────
DATASET_PATH = "dataset"
IMAGE_SIZE   = 64   # EuroSAT images are 64x64 natively
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# ── MAIN FUNCTION ─────────────────────────────────────
def load_dataset():
    features = []   # pixel arrays
    labels   = []   # class names
    skipped  = 0

    print("=" * 50)
    print("  Loading EuroSAT Dataset")
    print("=" * 50)

    for class_name in CLASSES:
        class_folder = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_folder):
            print(f"  ⚠️  Missing folder: {class_name}")
            continue

        image_files = os.listdir(class_folder)
        loaded = 0

        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)

            # Step 1: Read image
            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue

            # Step 2: Resize to 64×64
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

            # Step 3: Normalize 0–255 → 0.0–1.0
            image = image / 255.0

            # Step 4: Flatten → 64×64×3 = 12,288 values
            flat = image.flatten()

            features.append(flat)
            labels.append(class_name)
            loaded += 1

        print(f"  ✅ {class_name:<25}: {loaded} images loaded")

    features = np.array(features)   # shape: (N, 12288)
    labels   = np.array(labels)     # shape: (N,)

    print("\n" + "=" * 50)
    print(f"  Feature matrix : {features.shape}")
    print(f"  Labels array   : {labels.shape}")
    print(f"  Skipped        : {skipped} corrupt images")
    print("=" * 50)

    return features, labels

# ── TEST RUN ──────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_dataset()
# explore.py
import os

DATASET_PATH = "dataset"
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

print("=" * 45)
print("  EuroSAT Dataset Explorer")
print("=" * 45)

total = 0
for cls in CLASSES:
    folder = os.path.join(DATASET_PATH, cls)
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        total += count
        print(f"  {cls:<25}: {count} images")
    else:
        print(f"  {cls:<25}: ❌ FOLDER NOT FOUND")

print("-" * 45)
print(f"  {'TOTAL':<25}: {total} images")
print("=" * 45)
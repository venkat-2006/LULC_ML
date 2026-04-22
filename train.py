# # train.py
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.metrics import (accuracy_score, precision_score,
#                              recall_score, f1_score, confusion_matrix)
# from preprocess import load_dataset

# # ── CONFIG ────────────────────────────────────────────
# # Set to None to use ALL images (slower but better accuracy)
# # Set to 5000 for a quick test run first
# # MAX_SAMPLES = 5000
# MAX_SAMPLES = None

# # ═══════════════════════════════════════════════════════
# # STEP 1 — Load Data
# # ═══════════════════════════════════════════════════════
# print("\n" + "═"*55)
# print("  STEP 1: Loading dataset")
# print("═"*55)

# X, y = load_dataset()

# # Optional: use subset for speed
# if MAX_SAMPLES and MAX_SAMPLES < len(X):
#     print(f"\n  ⚡ Using {MAX_SAMPLES} samples (out of {len(X)}) for speed")
#     idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
#     X, y = X[idx], y[idx]

# # ═══════════════════════════════════════════════════════
# # STEP 2 — Encode Labels  (text → numbers)
# # ═══════════════════════════════════════════════════════
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

# # ═══════════════════════════════════════════════════════
# # STEP 3 — Train / Test Split  80% / 20%
# # ═══════════════════════════════════════════════════════
# print("\n" + "═"*55)
# print("  STEP 3: Splitting 80/20")
# print("═"*55)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded      # keeps class balance
# )
# print(f"  Train : {len(X_train)} samples")
# print(f"  Test  : {len(X_test)} samples")

# # ═══════════════════════════════════════════════════════
# # STEP 4 — PCA  (12,288 → 200 features)
# # ═══════════════════════════════════════════════════════
# print("\n" + "═"*55)
# print("  STEP 4: Applying PCA")
# print("═"*55)

# N_COMPONENTS = min(300, X_train.shape[0]-1, X_train.shape[1])
# pca = PCA(n_components=N_COMPONENTS, random_state=42)

# X_train_pca = pca.fit_transform(X_train)  # learn from train only
# X_test_pca  = pca.transform(X_test)       # apply same to test

# var_explained = sum(pca.explained_variance_ratio_) * 100
# print(f"  Features before PCA : {X_train.shape[1]}")
# print(f"  Features after PCA  : {X_train_pca.shape[1]}")
# print(f"  Variance retained   : {var_explained:.1f}%")

# # Save PCA + Label Encoder immediately
# joblib.dump(pca, "pca_model.pkl")
# joblib.dump(le,  "label_encoder.pkl")
# print("\n  ✅ pca_model.pkl saved")
# print("  ✅ label_encoder.pkl saved")

# # ═══════════════════════════════════════════════════════
# # STEP 5 — Define All 7 Models
# # ═══════════════════════════════════════════════════════
# models = {
#     "Decision Tree"      : DecisionTreeClassifier(
#                                max_depth=20,
#                                random_state=42),

#     "Naive Bayes"        : GaussianNB(),

#     "Logistic Regression": LogisticRegression(
#                                max_iter=2000,
#                                C=1.0,
#                                random_state=42),

#     "KNN"                : KNeighborsClassifier(
#                                n_neighbors=7,
#                                metric='euclidean'),

#     "SVM"                : SVC(
#                                kernel='rbf',
#                                C=10,
#                                gamma='scale',
#                                random_state=42,
#                                probability=True),

#     "Gradient Boosting"  : GradientBoostingClassifier(
#                                n_estimators=200,
#                                learning_rate=0.1,
#                                max_depth=5,
#                                random_state=42),

#     "Random Forest"      : RandomForestClassifier(
#                                n_estimators=200,
#                                max_depth=20,
#                                random_state=42),
# }
# # ═══════════════════════════════════════════════════════
# # STEP 6 — Train & Evaluate Every Model
# # ═══════════════════════════════════════════════════════
# print("\n" + "═"*55)
# print("  STEP 6: Training & Evaluating 7 Models")
# print("═"*55)

# results    = {}
# trained    = {}

# for name, model in models.items():
#     print(f"\n  🔄 {name}...", end=" ", flush=True)

#     model.fit(X_train_pca, y_train)
#     y_pred = model.predict(X_test_pca)

#     acc  = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#     rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#     f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

#     results[name] = {
#         "accuracy" : round(acc,  4),
#         "precision": round(prec, 4),
#         "recall"   : round(rec,  4),
#         "f1"       : round(f1,   4),
#     }
#     trained[name] = {"model": model, "y_pred": y_pred}

#     print(f"✅  Accuracy: {acc:.4f}")

# # ═══════════════════════════════════════════════════════
# # STEP 7 — Print Results Table
# # ═══════════════════════════════════════════════════════
# print("\n\n" + "═"*72)
# print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
# print("═"*72)
# for name, r in results.items():
#     print(f"  {name:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
#           f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
# print("═"*72)

# # ═══════════════════════════════════════════════════════
# # STEP 8 — Select Best Model (weighted formula)
# # ═══════════════════════════════════════════════════════
# print("\n" + "═"*55)
# print("  STEP 8: Selecting Best Model")
# print("═"*55)

# best_name  = None
# best_score = -1

# for name, r in results.items():
#     score = (0.4 * r["accuracy"]  +
#              0.3 * r["precision"] +
#              0.2 * r["recall"]    +
#              0.1 * r["f1"])
#     print(f"  {name:<22}  Combined Score: {score:.4f}")
#     if score > best_score:
#         best_score = score
#         best_name  = name

# print(f"\n  🏆 Best Model : {best_name}")
# print(f"  🏆 Best Score : {best_score:.4f}")

# # ═══════════════════════════════════════════════════════
# # STEP 9 — Save Best Model + All Results
# # ═══════════════════════════════════════════════════════
# joblib.dump(trained[best_name]["model"], "lulc_best_model.pkl")
# joblib.dump(results,                     "all_results.pkl")
# joblib.dump(best_name,                   "best_model_name.pkl")

# print(f"\n  ✅ lulc_best_model.pkl saved")
# print(f"  ✅ all_results.pkl saved")
# print(f"  ✅ best_model_name.pkl saved")

# # ═══════════════════════════════════════════════════════
# # STEP 10 — Plot 1: Accuracy Bar Chart
# # ═══════════════════════════════════════════════════════
# names      = list(results.keys())
# accuracies = [results[n]["accuracy"] for n in names]
# colors     = ["#0ea5e9" if n != best_name else "#22c55e" for n in names]

# plt.figure(figsize=(12, 5))
# bars = plt.bar(names, accuracies, color=colors, edgecolor='white')
# plt.xticks(rotation=25, ha='right')
# plt.ylim(0, 1.12)
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy Comparison — EuroSAT LULC Classification")

# for bar, val in zip(bars, accuracies):
#     plt.text(bar.get_x() + bar.get_width()/2,
#              bar.get_height() + 0.01,
#              f"{val:.2f}", ha='center', fontsize=9)

# plt.tight_layout()
# plt.savefig("model_comparison.png", dpi=150)
# plt.show()
# print("\n  ✅ model_comparison.png saved")

# # ═══════════════════════════════════════════════════════
# # STEP 11 — Plot 2: Confusion Matrix (Best Model)
# # ═══════════════════════════════════════════════════════
# cm = confusion_matrix(y_test, trained[best_name]["y_pred"])

# plt.figure(figsize=(12, 9))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=le.classes_,
#             yticklabels=le.classes_)
# plt.title(f"Confusion Matrix — {best_name} (EuroSAT)")
# plt.ylabel("Actual Class")
# plt.xlabel("Predicted Class")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=150)
# plt.show()
# print("  ✅ confusion_matrix.png saved")

# print("\n" + "═"*55)
# print("  🎉 TRAINING COMPLETE!")
# print("  Files saved:")
# print("    • pca_model.pkl")
# print("    • label_encoder.pkl")
# print("    • lulc_best_model.pkl")
# print("    • all_results.pkl")
# print("    • model_comparison.png")
# print("    • confusion_matrix.png")
# print("═"*55)
# train.py — FINAL VERSION (Full Dataset, Best Parameters)
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from preprocess import load_dataset

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
# None = use ALL 27,000 images (best accuracy, ~1-2 hrs)
# 10000 = faster test (~20-30 mins, ~85% accuracy)
MAX_SAMPLES = None

# ═══════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 1: Loading dataset")
print("═"*55)

X, y = load_dataset()

if MAX_SAMPLES and MAX_SAMPLES < len(X):
    print(f"\n  ⚡ Using {MAX_SAMPLES} samples out of {len(X)}")
    idx  = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
else:
    print(f"\n  ✅ Using FULL dataset: {len(X)} images")

# ═══════════════════════════════════════════════════════
# STEP 2 — Encode Labels
# ═══════════════════════════════════════════════════════
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

# ═══════════════════════════════════════════════════════
# STEP 3 — Train / Test Split  80% / 20%
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 3: Splitting 80% train / 20% test")
print("═"*55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_encoded
)
print(f"  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)}  samples")

# ═══════════════════════════════════════════════════════
# STEP 4 — PCA  (12,288 → 300 features)
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 4: Applying PCA")
print("═"*55)

N_COMPONENTS  = min(300, X_train.shape[0]-1, X_train.shape[1])
pca           = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca   = pca.fit_transform(X_train)
X_test_pca    = pca.transform(X_test)

var_explained = sum(pca.explained_variance_ratio_) * 100
print(f"  Features before PCA : {X_train.shape[1]}")
print(f"  Features after PCA  : {X_train_pca.shape[1]}")
print(f"  Variance retained   : {var_explained:.1f}%")

joblib.dump(pca, "pca_model.pkl")
joblib.dump(le,  "label_encoder.pkl")
print("\n  ✅ pca_model.pkl saved")
print("  ✅ label_encoder.pkl saved")

# ═══════════════════════════════════════════════════════
# STEP 5 — Define All 7 Models (Optimized)
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 5: Models defined (optimized parameters)")
print("═"*55)

models = {
    "Decision Tree"      : DecisionTreeClassifier(
                               max_depth    = 20,
                               random_state = 42),

    "Naive Bayes"        : GaussianNB(),

    "Logistic Regression": LogisticRegression(
                               max_iter     = 2000,
                               C            = 1.0,
                               random_state = 42),

    "KNN"                : KNeighborsClassifier(
                               n_neighbors  = 7,
                               metric       = 'euclidean'),

    "SVM"                : SVC(
                               kernel       = 'rbf',
                               C            = 10,
                               gamma        = 'scale',
                               random_state = 42,
                               probability  = True),

    "Gradient Boosting"  : GradientBoostingClassifier(
                               n_estimators  = 200,
                               learning_rate = 0.1,
                               max_depth     = 5,
                               random_state  = 42),

    "Random Forest"      : RandomForestClassifier(
                               n_estimators = 200,
                               max_depth    = 20,
                               random_state = 42),
}

for name in models:
    print(f"  ✅ {name}")

# ═══════════════════════════════════════════════════════
# STEP 6 — Train & Evaluate Every Model
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 6: Training & Evaluating (this takes time...)")
print("═"*55)

results = {}
trained = {}

for name, model in models.items():
    print(f"\n  🔄 Training {name}...", end=" ", flush=True)

    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

    results[name] = {
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
    }
    trained[name] = {"model": model, "y_pred": y_pred}

    print(f"✅  Accuracy: {acc:.4f}")

# ═══════════════════════════════════════════════════════
# STEP 7 — Print Results Table
# ═══════════════════════════════════════════════════════
print("\n\n" + "═"*72)
print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("═"*72)
for name, r in results.items():
    print(f"  {name:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
print("═"*72)

# ═══════════════════════════════════════════════════════
# STEP 8 — Select Best Model
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 8: Selecting Best Model")
print("═"*55)

best_name  = None
best_score = -1

for name, r in results.items():
    score = (0.4 * r["accuracy"]  +
             0.3 * r["precision"] +
             0.2 * r["recall"]    +
             0.1 * r["f1"])
    print(f"  {name:<22}  Combined Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_name  = name

print(f"\n  🏆 Best Model : {best_name}")
print(f"  🏆 Best Score : {best_score:.4f}")

# ═══════════════════════════════════════════════════════
# STEP 9 — Save Everything
# ═══════════════════════════════════════════════════════
joblib.dump(trained[best_name]["model"], "lulc_best_model.pkl")
joblib.dump(results,                     "all_results.pkl")
joblib.dump(best_name,                   "best_model_name.pkl")

print(f"\n  ✅ lulc_best_model.pkl saved")
print(f"  ✅ all_results.pkl saved")
print(f"  ✅ best_model_name.pkl saved")

# ═══════════════════════════════════════════════════════
# STEP 10 — Plot 1: Accuracy Bar Chart
# ═══════════════════════════════════════════════════════
names      = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
colors     = ["#22c55e" if n == best_name else "#0ea5e9" for n in names]

plt.figure(figsize=(12, 5))
bars = plt.bar(names, accuracies, color=colors, edgecolor='white', linewidth=0.5)
plt.xticks(rotation=25, ha='right')
plt.ylim(0, 1.12)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison — EuroSAT LULC Classification")
plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.4, label='90% line')
plt.legend()

for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.2f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("\n  ✅ model_comparison.png saved")

# ═══════════════════════════════════════════════════════
# STEP 11 — Plot 2: Confusion Matrix
# ═══════════════════════════════════════════════════════
cm = confusion_matrix(y_test, trained[best_name]["y_pred"])

plt.figure(figsize=(13, 10))
sns.heatmap(cm,
            annot         = True,
            fmt           = 'd',
            cmap          = 'Blues',
            xticklabels   = le.classes_,
            yticklabels   = le.classes_,
            linewidths    = 0.5)
plt.title(f"Confusion Matrix — {best_name}  |  EuroSAT (10 Classes)", fontsize=13)
plt.ylabel("Actual Class",    fontsize=11)
plt.xlabel("Predicted Class", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("  ✅ confusion_matrix.png saved")

# ═══════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  🎉 TRAINING COMPLETE!")
print(f"  🏆 Best Model   : {best_name}")
print(f"  🏆 Best Score   : {best_score:.4f}")
print(f"  🏆 Best Accuracy: {results[best_name]['accuracy']*100:.1f}%")
print("\n  Files saved:")
print("    • pca_model.pkl")
print("    • label_encoder.pkl")
print("    • lulc_best_model.pkl")
print("    • all_results.pkl")
print("    • best_model_name.pkl")
print("    • model_comparison.png")
print("    • confusion_matrix.png")
print("═"*55)
print("\n  ✅ Now run:  python app.py")
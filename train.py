# # train.py — Maximum accuracy version
# import numpy as np
# import joblib
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import (GradientBoostingClassifier,
#                                RandomForestClassifier,
#                                ExtraTreesClassifier,
#                                VotingClassifier)
# from sklearn.metrics import (accuracy_score, precision_score,
#                               recall_score, f1_score, confusion_matrix)
# from preprocess import load_dataset

# # ═══════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════
# MAX_SAMPLES  = None    # use ALL 27,000
# N_COMPONENTS = 200     # PCA components

# # ═══════════════════════════════════════
# # STEP 1 — Load
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 1 — Loading Dataset")
# print("═"*52)
# X, y = load_dataset()
# print(f"\n  Using ALL {len(X)} images")

# # ═══════════════════════════════════════
# # STEP 2 — Encode Labels
# # ═══════════════════════════════════════
# le        = LabelEncoder()
# y_encoded = le.fit_transform(y)
# print(f"  Classes : {list(le.classes_)}")

# # ═══════════════════════════════════════
# # STEP 3 — Split
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 3 — Train/Test Split 80/20")
# print("═"*52)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded,
#     test_size=0.2, random_state=42, stratify=y_encoded
# )
# print(f"  Train : {len(X_train)}")
# print(f"  Test  : {len(X_test)}")

# # ═══════════════════════════════════════
# # STEP 4 — Scale + PCA
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 4 — Scale + PCA")
# print("═"*52)

# scaler = StandardScaler()
# print("  Fitting StandardScaler...", end=" ", flush=True)
# X_train_s = scaler.fit_transform(X_train)
# X_test_s  = scaler.transform(X_test)
# print("done")

# n   = min(N_COMPONENTS, X_train_s.shape[0]-1, X_train_s.shape[1])
# pca = PCA(n_components=n, random_state=42)
# print("  Fitting PCA...", end=" ", flush=True)
# X_train_pca = pca.fit_transform(X_train_s)
# X_test_pca  = pca.transform(X_test_s)
# var = sum(pca.explained_variance_ratio_) * 100
# print(f"done  |  {X_train.shape[1]} → {n}  |  {var:.1f}% variance kept")

# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(pca,    "pca_model.pkl")
# joblib.dump(le,     "label_encoder.pkl")
# print("  scaler.pkl saved")
# print("  pca_model.pkl saved")
# print("  label_encoder.pkl saved")

# # ═══════════════════════════════════════
# # STEP 5 — Models
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 5 — Models")
# print("═"*52)

# rf  = RandomForestClassifier(
#         n_estimators=500, max_depth=None,
#         min_samples_leaf=1, n_jobs=-1, random_state=42)

# et  = ExtraTreesClassifier(
#         n_estimators=500, max_depth=None,
#         min_samples_leaf=1, n_jobs=-1, random_state=42)

# svm = SVC(kernel='rbf', C=10, gamma='scale',
#           random_state=42, probability=True)

# gb  = GradientBoostingClassifier(
#         n_estimators=300, learning_rate=0.08,
#         max_depth=5, subsample=0.8,
#         min_samples_leaf=10, random_state=42)

# lr  = LogisticRegression(
#         max_iter=2000, C=1.0,
#         solver='lbfgs',
#         random_state=42)

# models = {
#     "Decision Tree"      : DecisionTreeClassifier(
#                                max_depth=20, random_state=42),
#     "Naive Bayes"        : GaussianNB(),
#     "Logistic Regression": lr,
#     "KNN"                : KNeighborsClassifier(
#                                n_neighbors=7, metric='euclidean',
#                                algorithm='ball_tree', n_jobs=-1),
#     "SVM"                : svm,
#     "Gradient Boosting"  : gb,
#     "Random Forest"      : rf,
#     "Extra Trees"        : et,
#     "Voting Ensemble"    : VotingClassifier(
#                                estimators=[
#                                    ('rf',  rf),
#                                    ('et',  et),
#                                    ('svm', svm),
#                                    ('gb',  gb),
#                                ],
#                                voting='soft',
#                                n_jobs=-1),
# }

# times = {
#     "Decision Tree"      : "~2 mins",
#     "Naive Bayes"        : "~1 min",
#     "Logistic Regression": "~5 mins",
#     "KNN"                : "~5 mins",
#     "SVM"                : "~20 mins",
#     "Gradient Boosting"  : "~40 mins",
#     "Random Forest"      : "~8 mins",
#     "Extra Trees"        : "~8 mins",
#     "Voting Ensemble"    : "~30 mins",
# }

# for name in models:
#     print(f"  {name:<22} (est. {times[name]})")

# # ═══════════════════════════════════════
# # STEP 6 — Train & Evaluate
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 6 — Training All Models")
# print("  Total estimated time: ~1.5–2 hrs")
# print("═"*52)

# results = {}
# trained = {}

# for name, model in models.items():
#     print(f"\n  {name} (est. {times[name]})...", end=" ", flush=True)
#     model.fit(X_train_pca, y_train)
#     y_pred = model.predict(X_test_pca)

#     acc  = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred,
#                            average='weighted', zero_division=0)
#     rec  = recall_score(y_test, y_pred,
#                         average='weighted', zero_division=0)
#     f1   = f1_score(y_test, y_pred,
#                     average='weighted', zero_division=0)

#     results[name] = {
#         "accuracy" : round(acc,  4),
#         "precision": round(prec, 4),
#         "recall"   : round(rec,  4),
#         "f1"       : round(f1,   4),
#     }
#     trained[name] = {"model": model, "y_pred": y_pred}
#     print(f"Accuracy: {acc*100:.2f}%")

# # ═══════════════════════════════════════
# # STEP 7 — Results Table
# # ═══════════════════════════════════════
# print("\n\n" + "═"*72)
# print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10}"
#       f" {'Recall':>10} {'F1':>10}")
# print("═"*72)
# for name, r in results.items():
#     print(f"  {name:<22} {r['accuracy']*100:>9.2f}%"
#           f" {r['precision']*100:>9.2f}%"
#           f" {r['recall']*100:>9.2f}%"
#           f" {r['f1']*100:>9.2f}%")
# print("═"*72)

# # ═══════════════════════════════════════
# # STEP 8 — Best Model
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  STEP 8 — Best Model Selection")
# print("═"*52)
# best_name  = None
# best_score = -1

# for name, r in results.items():
#     score = (0.4*r["accuracy"]  + 0.3*r["precision"] +
#              0.2*r["recall"]    + 0.1*r["f1"])
#     print(f"  {name:<22}  Score: {score:.4f}")
#     if score > best_score:
#         best_score = score
#         best_name  = name

# print(f"\n  Best Model : {best_name}")
# print(f"  Accuracy   : {results[best_name]['accuracy']*100:.2f}%")

# # ═══════════════════════════════════════
# # STEP 9 — Save
# # ═══════════════════════════════════════
# joblib.dump(trained[best_name]["model"], "lulc_best_model.pkl")
# joblib.dump(results,                     "all_results.pkl")
# joblib.dump(best_name,                   "best_model_name.pkl")
# print(f"\n  lulc_best_model.pkl saved")
# print(f"  all_results.pkl saved")
# print(f"  best_model_name.pkl saved")

# # ═══════════════════════════════════════
# # STEP 10 — Bar Chart
# # ═══════════════════════════════════════
# names      = list(results.keys())
# accuracies = [results[n]["accuracy"]*100 for n in names]
# colors     = ["#22c55e" if n == best_name else "#0ea5e9" for n in names]

# plt.figure(figsize=(14, 6))
# bars = plt.bar(names, accuracies, color=colors,
#                edgecolor='white', linewidth=0.6)
# plt.xticks(rotation=30, ha='right', fontsize=9)
# plt.ylim(0, 115)
# plt.ylabel("Accuracy (%)", fontsize=11)
# plt.title("Model Accuracy — EuroSAT LULC (10 Classes)", fontsize=13)
# plt.axhline(y=90, color='red',    linestyle='--',
#             alpha=0.5, label='90% target')
# plt.axhline(y=80, color='orange', linestyle='--',
#             alpha=0.5, label='80% baseline')
# plt.legend(fontsize=9)
# for bar, val in zip(bars, accuracies):
#     plt.text(bar.get_x() + bar.get_width()/2,
#              bar.get_height() + 1,
#              f"{val:.1f}%", ha='center',
#              fontsize=8, fontweight='bold')
# plt.tight_layout()
# plt.savefig("model_comparison.png", dpi=150)
# print("  model_comparison.png saved")

# # ═══════════════════════════════════════
# # STEP 11 — Confusion Matrix
# # ═══════════════════════════════════════
# cm = confusion_matrix(y_test, trained[best_name]["y_pred"])
# plt.figure(figsize=(14, 11))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=le.classes_,
#             yticklabels=le.classes_,
#             linewidths=0.4)
# plt.title(f"Confusion Matrix — {best_name} | EuroSAT",
#           fontsize=13)
# plt.ylabel("Actual",    fontsize=11)
# plt.xlabel("Predicted", fontsize=11)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=150)
# print("  confusion_matrix.png saved")

# # ═══════════════════════════════════════
# # DONE
# # ═══════════════════════════════════════
# print("\n" + "═"*52)
# print("  TRAINING COMPLETE!")
# print(f"  Best Model : {best_name}")
# print(f"  Accuracy   : {results[best_name]['accuracy']*100:.2f}%")
# print("\n  Files saved:")
# print("    scaler.pkl            pca_model.pkl")
# print("    label_encoder.pkl     lulc_best_model.pkl")
# print("    all_results.pkl       best_model_name.pkl")
# print("    model_comparison.png  confusion_matrix.png")
# print("═"*52)
# print("\n  Next: python app.py")
# train.py — Balanced version (good accuracy + reasonable time)
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from preprocess import load_dataset

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
MAX_SAMPLES  = None   # ALL 27,000 images
N_COMPONENTS = 150    # PCA components

# ═══════════════════════════════════════════════
# STEP 1 — Load
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 1 — Loading Dataset")
print("═"*55)

X, y = load_dataset()
print(f"\n  ✅ Full dataset: {len(X)} images")
print(f"  ✅ Feature vector size: {X.shape[1]} (from preprocess.py)")

# ═══════════════════════════════════════════════
# STEP 2 — Encode Labels
# ═══════════════════════════════════════════════
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

# ═══════════════════════════════════════════════
# STEP 3 — Train/Test Split 80/20
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 3 — Train/Test Split 80/20")
print("═"*55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_encoded
)
print(f"  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)}  samples")

# ═══════════════════════════════════════════════
# STEP 4 — Scale + PCA
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 4 — StandardScaler + PCA")
print("═"*55)

# Scale first (important for SVM + LR + PCA)
scaler = StandardScaler()
print("  Fitting StandardScaler...", end=" ", flush=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("✅")

# PCA
n   = min(N_COMPONENTS, X_train_s.shape[0]-1, X_train_s.shape[1])
pca = PCA(n_components=n, random_state=42)
print("  Fitting PCA...", end=" ", flush=True)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
var = sum(pca.explained_variance_ratio_) * 100
print(f"✅  {X_train.shape[1]} → {n} features  |  {var:.1f}% variance kept")

# Save preprocessing objects
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca,    "pca_model.pkl")
joblib.dump(le,     "label_encoder.pkl")
print("\n  ✅ scaler.pkl        saved")
print("  ✅ pca_model.pkl     saved")
print("  ✅ label_encoder.pkl saved")

# ═══════════════════════════════════════════════
# STEP 5 — Define 7 Models
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 5 — 7 Models Defined")
print("═"*55)

models = {
    "Decision Tree"      : DecisionTreeClassifier(
                               max_depth    = 20,
                               random_state = 42),

    "Naive Bayes"        : GaussianNB(),

    "Logistic Regression": LogisticRegression(
                           max_iter     = 300,   # enough to converge
                           C            = 1.0,
                           solver       = 'saga', # fastest for large data
                           n_jobs       = -1,     # all CPU cores
                           random_state = 42),

    "KNN"                : KNeighborsClassifier(
                               n_neighbors  = 7,
                               metric       = 'euclidean',
                               algorithm    = 'ball_tree',
                               n_jobs       = -1),

    "SVM"                : SVC(
                               kernel       = 'rbf',
                               C            = 10,
                               gamma        = 'scale',
                               random_state = 42,
                               probability  = True),
"Gradient Boosting"  : HistGradientBoostingClassifier(
                           max_iter      = 100,
                           learning_rate = 0.1,
                           max_depth     = 6,
                           random_state  = 42),

    "Random Forest"      : RandomForestClassifier(
                               n_estimators  = 300,
                               max_depth     = None,
                               min_samples_leaf = 1,
                               n_jobs        = -1,
                               random_state  = 42),
}

times = {
    "Decision Tree"      : "~1 min",
    "Naive Bayes"        : "~30 sec",
    "Logistic Regression": "~3 mins",
    "KNN"                : "~4 mins",
    "SVM"                : "~25 mins",
    "Gradient Boosting"  : "~25 mins",
    "Random Forest"      : "~5 mins",
}

for name in models:
    print(f"  ✅ {name:<22}  (est. {times[name]})")

print("\n  ⏱️  Total estimated: ~60-70 mins")

# ═══════════════════════════════════════════════
# STEP 6 — Train & Evaluate All Models
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 6 — Training All 7 Models")
print("═"*55)

results = {}
trained = {}

for name, model in models.items():
    print(f"\n  🔄 {name}  (est. {times[name]})...",
          end=" ", flush=True)

    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,
                           average='weighted', zero_division=0)
    rec  = recall_score(y_test,   y_pred,
                        average='weighted', zero_division=0)
    f1   = f1_score(y_test,       y_pred,
                    average='weighted', zero_division=0)

    results[name] = {
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
    }
    trained[name] = {"model": model, "y_pred": y_pred}
    print(f"✅  Accuracy: {acc*100:.2f}%")

# ═══════════════════════════════════════════════
# STEP 7 — Results Table
# ═══════════════════════════════════════════════
print("\n\n" + "═"*72)
print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10}"
      f" {'Recall':>10} {'F1':>10}")
print("═"*72)
for name, r in results.items():
    acc_str = f"{r['accuracy']*100:.2f}%"
    pre_str = f"{r['precision']*100:.2f}%"
    rec_str = f"{r['recall']*100:.2f}%"
    f1_str  = f"{r['f1']*100:.2f}%"
    print(f"  {name:<22} {acc_str:>10} {pre_str:>10}"
          f" {rec_str:>10} {f1_str:>10}")
print("═"*72)

# ═══════════════════════════════════════════════
# STEP 8 — Select Best Model
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  STEP 8 — Best Model Selection")
print("  Formula: 0.4×Acc + 0.3×Prec + 0.2×Rec + 0.1×F1")
print("═"*55)

best_name  = None
best_score = -1

for name, r in results.items():
    score = (0.4 * r["accuracy"]  +
             0.3 * r["precision"] +
             0.2 * r["recall"]    +
             0.1 * r["f1"])
    print(f"  {name:<22}  Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_name  = name

print(f"\n  🏆 Best Model    : {best_name}")
print(f"  🏆 Combined Score: {best_score:.4f}")
print(f"  🏆 Accuracy      : {results[best_name]['accuracy']*100:.2f}%")
print(f"  🏆 F1 Score      : {results[best_name]['f1']*100:.2f}%")

# ═══════════════════════════════════════════════
# STEP 9 — Save All Files
# ═══════════════════════════════════════════════
joblib.dump(trained[best_name]["model"], "lulc_best_model.pkl")
joblib.dump(results,                     "all_results.pkl")
joblib.dump(best_name,                   "best_model_name.pkl")

print(f"\n  ✅ lulc_best_model.pkl  saved")
print(f"  ✅ all_results.pkl      saved")
print(f"  ✅ best_model_name.pkl  saved")

# ═══════════════════════════════════════════════
# STEP 10 — Bar Chart
# ═══════════════════════════════════════════════
names      = list(results.keys())
accuracies = [results[n]["accuracy"] * 100 for n in names]
colors     = ["#22c55e" if n == best_name else "#0ea5e9" for n in names]

plt.figure(figsize=(13, 6))
bars = plt.bar(names, accuracies, color=colors,
               edgecolor='white', linewidth=0.6)
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.ylim(0, 115)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.title("Model Accuracy Comparison — EuroSAT LULC (10 Classes)",
          fontsize=13, fontweight='bold')
plt.axhline(y=90, color='red',    linestyle='--',
            alpha=0.5, label='90% target')
plt.axhline(y=75, color='orange', linestyle='--',
            alpha=0.5, label='75% baseline')
plt.legend(fontsize=9)

for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{val:.1f}%", ha='center',
             fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("\n  ✅ model_comparison.png saved")

# ═══════════════════════════════════════════════
# STEP 11 — Confusion Matrix
# ═══════════════════════════════════════════════
cm = confusion_matrix(y_test, trained[best_name]["y_pred"])

plt.figure(figsize=(14, 11))
sns.heatmap(cm,
            annot       = True,
            fmt         = 'd',
            cmap        = 'Blues',
            xticklabels = le.classes_,
            yticklabels = le.classes_,
            linewidths  = 0.4)
plt.title(f"Confusion Matrix — {best_name}  |  EuroSAT 10 Classes",
          fontsize=13, fontweight='bold')
plt.ylabel("Actual Class",    fontsize=11)
plt.xlabel("Predicted Class", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("  ✅ confusion_matrix.png saved")

# ═══════════════════════════════════════════════
# COMPLETE
# ═══════════════════════════════════════════════
print("\n" + "═"*55)
print("  🎉 TRAINING COMPLETE!")
print(f"  🏆 Best Model : {best_name}")
print(f"  🏆 Accuracy   : {results[best_name]['accuracy']*100:.2f}%")
print(f"  🏆 F1 Score   : {results[best_name]['f1']*100:.2f}%")
print("\n  All saved files:")
print("    scaler.pkl           label_encoder.pkl")
print("    pca_model.pkl        lulc_best_model.pkl")
print("    all_results.pkl      best_model_name.pkl")
print("    model_comparison.png confusion_matrix.png")
print("═"*55)
print("\n  ▶️  Next → python app.py")
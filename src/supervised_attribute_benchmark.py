"""
AutoVision - Supervised Multi-Attribute Benchmark
Evaluates whether the 512-D embeddings contain linear and non-linear separability
for the target vehicle attributes when trained with supervision:
1. Body Type (4 classes)
2. Brand / Make (7 classes)
3. Model (10 classes)
4. Year / Generation (5 classes)

Compares:
- Logistic Regression (L2)
- Linear SVM (LinearSVC)
- Multi-Layer Perceptron (MLP)
- Random Forest
Against Unsupervised Clustering Baselines.
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BODY_TYPES = {
    "impala": "Sedan",
    "grand caravan": "Minivan",
    "explorer": "SUV",
    "mustang": "Coupe",
    "taurus": "Sedan",
    "accord": "Sedan",
    "odyssey": "Minivan",
    "altima": "Sedan",
    "camry": "Sedan",
    "jetta": "Sedan",
}


def parse_label(raw_class):
    parts = raw_class.split("_")
    brand = parts[0].capitalize()
    year = parts[-1]
    model = "_".join(parts[1:-1]).replace("_", " ").title()
    body_type = BODY_TYPES.get(model.lower(), "Sedan")
    return brand, model, year, body_type


def load_dataset():
    raw_classes = []
    with open("train_data.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_classes.append(row[0].strip())

    brands, models, years, body_types = [], [], [], []
    for rc in raw_classes:
        b, m, y, t = parse_label(rc)
        brands.append(b)
        models.append(m)
        years.append(y)
        body_types.append(t)

    features = np.loadtxt("train_features.csv", delimiter=",", dtype=np.float32)
    X_l2 = normalize(features, norm="l2")

    return X_l2, np.array(brands), np.array(models), np.array(years), np.array(body_types)


def benchmark_attribute(X, y_labels, target_name):
    """Runs 5-Fold Stratified Cross-Validation across multiple classifiers."""
    print(f"\n[EVALUATING] Target Attribute: {target_name} ({len(np.unique(y_labels))} classes)...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, C=0.5, max_iter=2000),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="f1_macro")
        dt = time.time() - t0
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        mean_f1 = np.mean(f1_scores)
        results[name] = {"accuracy": mean_acc, "std": std_acc, "f1_macro": mean_f1}
        print(f"  - {name:20s}: Accuracy = {mean_acc*100:5.2f}% (+/- {std_acc*100:4.2f}%) | Macro F1 = {mean_f1:.4f} (took {dt:.1f}s)")

    return results


def main():
    total_start = time.time()
    print("=" * 80)
    print("AUTORecognition: SUPERVISED ATTRIBUTE CLASSIFICATION BENCHMARK")
    print("Assessing feature discriminative power with ground truth supervision")
    print("=" * 80)

    X_l2, brands, models, years, body_types = load_dataset()

    targets = [
        ("Body Type", body_types),
        ("Brand (Make)", brands),
        ("Model", models),
        ("Year (Era)", years),
    ]

    benchmark_records = []
    target_results = {}

    for name, labels in targets:
        res = benchmark_attribute(X_l2, labels, name)
        target_results[name] = res
        for clf_name, metrics_data in res.items():
            benchmark_records.append({
                "Target Attribute": name,
                "Classifier": clf_name,
                "Accuracy": round(float(metrics_data["accuracy"]), 4),
                "Std": round(float(metrics_data["std"]), 4),
                "Macro F1": round(float(metrics_data["f1_macro"]), 4),
            })

    # Save to CSV
    os.makedirs("reports/figures", exist_ok=True)
    csv_path = "reports/supervised_attribute_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Target Attribute", "Classifier", "Accuracy", "Std", "Macro F1"])
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"\n[REPORT] Saved benchmark table to '{csv_path}'")

    # Plot Comparison: Unsupervised Purity vs Supervised Accuracy
    print("[PLOTTING] Generating Unsupervised vs Supervised Comparison Plot...")
    # Unsupervised purities from previous Spectral/K-Means run:
    unsupervised_purities = {
        "Body Type": 69.1,
        "Brand (Make)": 32.0,
        "Model": 34.8,
        "Year (Era)": 48.3,
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    attr_names = list(unsupervised_purities.keys())
    unsuper_vals = [unsupervised_purities[a] for a in attr_names]
    super_lr_vals = [target_results[a]["Logistic Regression"]["accuracy"] * 100 for a in attr_names]
    super_mlp_vals = [target_results[a]["MLP (Neural Net)"]["accuracy"] * 100 for a in attr_names]

    x = np.arange(len(attr_names))
    width = 0.25

    r1 = ax.bar(x - width, unsuper_vals, width, label="Unsupervised Clustering Purity (%)", color="#9B9B9B")
    r2 = ax.bar(x, super_lr_vals, width, label="Supervised Logistic Regression (%)", color="#4A90E2")
    r3 = ax.bar(x + width, super_mlp_vals, width, label="Supervised Neural Net (MLP) (%)", color="#50E3C2")

    ax.set_title("AutoVision: Unsupervised Clustering vs. Supervised Classification by Attribute",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attr_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy / Purity (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = "reports/figures/unsupervised_vs_supervised.png"
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTTING] Saved figure to '{chart_path}'")

    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY (LOGISTIC REGRESSION vs CLUSTERING):")
    print("=" * 80)
    for a in attr_names:
        un = unsupervised_purities[a]
        sup = target_results[a]["Logistic Regression"]["accuracy"] * 100
        diff = sup - un
        print(f"  - {a:18s}: Unsupervised = {un:5.1f}%  ->  Supervised = {sup:5.1f}%  (Gain: +{diff:5.1f}%)")
    print("=" * 80)
    print(f"\n[DONE] Execution completed in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()

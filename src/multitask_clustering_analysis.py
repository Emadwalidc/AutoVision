"""
AutoVision - Attribute-Specific Clustering Analysis
Evaluates unsupervised feature clustering specifically along the 5 project targets:
1. Brand (Make)       -> 7 classes
2. Body Type          -> 4 classes
3. Model              -> 10 classes
4. Year (Generation)  -> 5 classes
5. Color (Visual)     -> Extracted dominant car color
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import normalize
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
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


def detect_quick_color(img_path):
    """Fast dominant color detection for central vehicle ROI."""
    if not os.path.exists(img_path):
        return "Unknown"
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        cropped = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))).resize((40, 40))
        pixels = np.array(cropped).reshape(-1, 3) / 255.0
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        val = max_c
        sat = np.where(max_c == 0, 0, diff / (max_c + 1e-6))

        hue = np.zeros_like(max_c)
        mask_r = (max_c == r) & (diff > 0)
        mask_g = (max_c == g) & (diff > 0)
        mask_b = (max_c == b) & (diff > 0)

        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

        color_votes = []
        for h_val, s_val, v_val in zip(hue, sat, val):
            if v_val < 0.22:
                color_votes.append("Black")
            elif s_val < 0.16 and v_val > 0.70:
                color_votes.append("White")
            elif s_val < 0.20:
                color_votes.append("Silver/Gray")
            else:
                if (h_val >= 340) or (h_val < 15):
                    color_votes.append("Red")
                elif 15 <= h_val < 45:
                    color_votes.append("Brown" if v_val < 0.6 else "Yellow")
                elif 45 <= h_val < 70:
                    color_votes.append("Yellow")
                elif 70 <= h_val < 165:
                    color_votes.append("Green")
                elif 165 <= h_val < 260:
                    color_votes.append("Blue")
                else:
                    color_votes.append("Red")

        from collections import Counter
        return Counter(color_votes).most_common(1)[0][0]
    except Exception:
        return "Unknown"


def calculate_purity(y_true, y_pred):
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(contingency, axis=0)) / np.sum(contingency))


def main():
    print("=" * 80)
    print("AUTORecognition: MULTI-ATTRIBUTE CLUSTERING EVALUATION")
    print("Evaluating how features cluster across: Brand, Model, Year, Body Type, & Color")
    print("=" * 80)

    # 1. Load data
    raw_classes = []
    filenames = []
    with open("train_data.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_classes.append(row[0].strip())
                filenames.append(row[1].strip())

    brands = []
    models = []
    years = []
    body_types = []

    for rc in raw_classes:
        b, m, y, t = parse_label(rc)
        brands.append(b)
        models.append(m)
        years.append(y)
        body_types.append(t)

    brands = np.array(brands)
    models = np.array(models)
    years = np.array(years)
    body_types = np.array(body_types)

    # Load features
    print("\n[1/4] Loading features and computing Spectral Manifold representations...")
    features = np.loadtxt("train_features.csv", delimiter=",", dtype=np.float32)
    X_l2 = normalize(features, norm="l2")

    # Compute Spectral Manifold (best performing embedding from Exp 3)
    se = SpectralEmbedding(n_components=10, affinity="nearest_neighbors", n_neighbors=25, random_state=RANDOM_STATE)
    X_spectral = se.fit_transform(X_l2)
    X_spectral = normalize(X_spectral, norm="l2")

    # Detect color for 1,000 samples to analyze color clustering
    print("[2/4] Extracting vehicle color attributes for color-correlation analysis...")
    colors = []
    color_indices = list(range(1000))
    for i in color_indices:
        img_p = os.path.join("train_images", raw_classes[i], filenames[i])
        colors.append(detect_quick_color(img_p))
    colors = np.array(colors)

    # 3. Benchmark Clustering across each target attribute
    print("\n[3/4] Benchmarking clustering per attribute:")
    targets = [
        ("Body Type", body_types, len(np.unique(body_types))),
        ("Brand (Make)", brands, len(np.unique(brands))),
        ("Model", models, len(np.unique(models))),
        ("Year (Era)", years, len(np.unique(years))),
    ]

    results = []

    for name, ground_truth, k in targets:
        # Standard L2 K-Means
        km_l2 = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15)
        pred_l2 = km_l2.fit_predict(X_l2)

        # Spectral Manifold K-Means
        km_spec = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15)
        pred_spec = km_spec.fit_predict(X_spectral)

        purity_l2 = calculate_purity(ground_truth, pred_l2)
        ari_l2 = metrics.adjusted_rand_score(ground_truth, pred_l2)
        nmi_l2 = metrics.normalized_mutual_info_score(ground_truth, pred_l2)

        purity_spec = calculate_purity(ground_truth, pred_spec)
        ari_spec = metrics.adjusted_rand_score(ground_truth, pred_spec)
        nmi_spec = metrics.normalized_mutual_info_score(ground_truth, pred_spec)
        sil_spec = metrics.silhouette_score(X_spectral, pred_spec)

        results.append({
            "Target Attribute": name,
            "Classes (k)": k,
            "Baseline ARI": round(float(ari_l2), 4),
            "Spectral ARI": round(float(ari_spec), 4),
            "Baseline NMI": round(float(nmi_l2), 4),
            "Spectral NMI": round(float(nmi_spec), 4),
            "Baseline Purity": round(float(purity_l2), 4),
            "Spectral Purity": round(float(purity_spec), 4),
            "Spectral Silhouette": round(float(sil_spec), 4)
        })

        print(f"\n  -> Target: {name:15s} (k={k} classes):")
        print(f"     Baseline L2    : ARI = {ari_l2:.4f} | NMI = {nmi_l2:.4f} | Purity = {purity_l2*100:.1f}%")
        print(f"     Spectral (SOTA): ARI = {ari_spec:.4f} | NMI = {nmi_spec:.4f} | Purity = {purity_spec*100:.1f}% | Sil = {sil_spec:.4f}")

    # Color correlation analysis on the 1000 samples
    km_color = KMeans(n_clusters=len(np.unique(colors)), random_state=RANDOM_STATE, n_init=15)
    pred_color = km_color.fit_predict(X_l2[:1000])
    color_nmi = metrics.normalized_mutual_info_score(colors, pred_color)
    color_purity = calculate_purity(colors, pred_color)
    print(f"\n  -> Target: Vehicle Color (k={len(np.unique(colors))} palette groups):")
    print(f"     Color Agreement: NMI = {color_nmi:.4f} | Purity = {color_purity*100:.1f}%")

    # 4. Save results to CSV & Figure
    print("\n[4/4] Generating Attribute-Specific Clustering Dashboard...")
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = "reports/attribute_clustering_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Plot multi-attribute comparison chart
    attr_names = [r["Target Attribute"] for r in results]
    spectral_aris = [r["Spectral ARI"] for r in results]
    spectral_nmis = [r["Spectral NMI"] for r in results]
    spectral_purities = [r["Spectral Purity"] * 100 for r in results]

    x = np.arange(len(attr_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    r1 = ax.bar(x - width, spectral_purities, width, label="Cluster Purity (%)", color="#4A90E2")
    r2 = ax.bar(x, [v * 100 for v in spectral_nmis], width, label="NMI (x100)", color="#50E3C2")
    r3 = ax.bar(x + width, [v * 100 for v in spectral_aris], width, label="ARI (x100)", color="#F5A623")

    ax.set_title("AutoVision: Clustering Performance by Target Vehicle Attribute", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attr_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(figures_dir, "attribute_clustering_comparison.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {chart_path}")
    print(f"      - Saved: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


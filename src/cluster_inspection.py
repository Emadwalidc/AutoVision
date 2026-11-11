"""
AutoVision - Visual Error Analysis & PCA-Denoised Clustering Inspection
Phase 1: Task 5 - Visual Cluster Inspection & Prototypical Analysis

This script:
1. Tests PCA-denoised clustering (512-D vs 129-D vs 68-D).
2. Computes cluster centroids and identifies prototypical vehicle images (nearest to centroid).
3. Produces a 10-cluster prototype montage (reports/figures/cluster_prototypes_montage.png).
4. Conducts a visual error analysis on confused classes (reports/figures/confusion_case_study.png).
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_dataset(data_path="train_data.csv", features_path="train_features.csv"):
    """Load annotations and pre-extracted feature embeddings."""
    print("[1/5] Loading data and feature vectors...")
    labels = []
    filenames = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                labels.append(row[0].strip())
                filenames.append(row[1].strip())

    labels = np.array(labels)
    features = np.loadtxt(features_path, delimiter=",", dtype=np.float32)
    unique_classes = np.unique(labels)
    return labels, filenames, features, unique_classes


def evaluate_pca_kmeans(X_l2, y, output_dir):
    """Compare K-Means on Full 512-D vs 129-D (90% var) vs 68-D (80% var)."""
    print("[2/5] Evaluating PCA-denoised clustering...")
    dims = [512, 129, 68, 30]
    results = []

    pca_models = {}
    X_projections = {512: X_l2}

    for d in [129, 68, 30]:
        pca = PCA(n_components=d, random_state=RANDOM_STATE)
        X_proj = pca.fit_transform(X_l2)
        X_proj = normalize(X_proj, norm="l2")
        X_projections[d] = X_proj
        pca_models[d] = pca

    for d in dims:
        X_data = X_projections[d]
        km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
        preds = km.fit_predict(X_data)

        sil = metrics.silhouette_score(X_data, preds)
        ch = metrics.calinski_harabasz_score(X_data, preds)
        db = metrics.davies_bouldin_score(X_data, preds)
        ari = metrics.adjusted_rand_score(y, preds)
        nmi = metrics.normalized_mutual_info_score(y, preds)

        results.append({
            "Dimensions": d,
            "Silhouette": round(float(sil), 4),
            "Calinski_Harabasz": round(float(ch), 2),
            "Davies_Bouldin": round(float(db), 4),
            "ARI": round(float(ari), 4),
            "NMI": round(float(nmi), 4)
        })
        print(f"      - Dim {d:3d}: ARI={ari:.4f} | NMI={nmi:.4f} | Silhouette={sil:.4f}")

    # Plot comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    dim_labels = [f"{r['Dimensions']}D" for r in results]
    aris = [r["ARI"] for r in results]
    nmis = [r["NMI"] for r in results]

    x = np.arange(len(dim_labels))
    width = 0.35

    ax1.bar(x - width/2, aris, width, label="ARI", color="royalblue")
    ax1.bar(x + width/2, nmis, width, label="NMI", color="darkorange")
    ax1.set_title("External Agreement vs Feature Dimension", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dim_labels)
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    sils = [r["Silhouette"] for r in results]
    ax2.bar(dim_labels, sils, color="teal", width=0.4)
    ax2.set_title("Silhouette Score vs Feature Dimension", fontweight="bold")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "pca_kmeans_comparison.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {chart_path}")
    return results


def build_cluster_prototypes_montage(labels, filenames, X_l2, unique_classes, output_dir, images_dir="train_images"):
    """Find images closest to each cluster center and build a 10x5 visual montage."""
    print("[3/5] Computing cluster prototypes from closest centroid samples...")
    km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    preds = km.fit_predict(X_l2)
    centroids = km.cluster_centers_

    # Determine dominant class for each cluster
    contingency = metrics.cluster.contingency_matrix(labels, preds)
    dominant_class_idx = np.argmax(contingency, axis=0)

    # 10 clusters x 5 prototypical images
    n_prototypes = 5
    fig, axes = plt.subplots(10, n_prototypes, figsize=(15, 22))

    for cluster_id in range(10):
        c_mask = np.where(preds == cluster_id)[0]
        c_samples = X_l2[c_mask]

        # Compute Euclidean distance to cluster centroid
        dist_to_center = np.linalg.norm(c_samples - centroids[cluster_id], axis=1)
        sorted_order = np.argsort(dist_to_center)
        top_indices = c_mask[sorted_order[:n_prototypes]]

        dom_class = unique_classes[dominant_class_idx[cluster_id]].replace("_", " ")

        for col_idx, sample_idx in enumerate(top_indices):
            ax = axes[cluster_id, col_idx]
            img_class = labels[sample_idx]
            img_name = filenames[sample_idx]
            img_path = os.path.join(images_dir, img_class, img_name)

            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")

            title_text = f"True: {img_class.split('_')[1]}"
            ax.set_title(title_text, fontsize=8)
            ax.axis("off")

            if col_idx == 0:
                ax.text(-0.25, 0.5, f"Cluster {cluster_id}\n(Dom: {dom_class.split(' ')[1]})",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        va="center", ha="right")

    plt.suptitle("AutoVision: Top-5 Prototypical Vehicle Images per K-Means Cluster",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    montage_path = os.path.join(output_dir, "cluster_prototypes_montage.png")
    fig.savefig(montage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {montage_path}")
    return km, preds


def build_confusion_case_study(labels, filenames, preds, unique_classes, output_dir, images_dir="train_images"):
    """Examine confused sedan classes vs clear minivan/SUV classes."""
    print("[4/5] Building visual confusion case study...")
    # Two sedans that are frequently confused: Honda Accord vs Toyota Camry
    class_a = "honda_accord_2005-2006"
    class_b = "toyota_camry_2002-2003"
    # A distinct class: Ford Mustang
    class_c = "ford_mustang_2000-2001"

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))

    def show_samples(row_idx, class_name, row_title):
        indices = np.where(labels == class_name)[0][:4]
        for col_idx, idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            img_path = os.path.join(images_dir, class_name, filenames[idx])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(f"Assigned Cluster: {preds[idx]}", fontsize=9)
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.15, 0.5, row_title, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="right")

    show_samples(0, class_a, "Honda Accord\n(2005-06)")
    show_samples(1, class_b, "Toyota Camry\n(2002-03)")
    show_samples(2, class_c, "Ford Mustang\n(2000-01)")

    plt.suptitle("Visual Error & Silhouette Comparison: Why Sedans Confuse vs Distinct Shapes",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    case_path = os.path.join(output_dir, "confusion_case_study.png")
    fig.savefig(case_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {case_path}")


def main():
    start_time = time.time()
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    labels, filenames, features, unique_classes = load_dataset()
    X_l2 = normalize(features, norm="l2")

    # 1. Evaluate PCA dimensionality reduction effect on clustering
    pca_results = evaluate_pca_kmeans(X_l2, labels, figures_dir)

    # 2. Build cluster prototype montage
    km, preds = build_cluster_prototypes_montage(labels, filenames, X_l2, unique_classes, figures_dir)

    # 3. Build confusion case study
    build_confusion_case_study(labels, filenames, preds, unique_classes, figures_dir)

    # 4. Save PCA metrics
    with open("reports/pca_denoised_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pca_results[0].keys()))
        writer.writeheader()
        writer.writerows(pca_results)

    elapsed = time.time() - start_time
    print(f"[5/5] Visual inspection and PCA analysis finished in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()


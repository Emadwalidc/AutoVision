"""
AutoVision - High-Performance Clustering Pipeline (Phase 1)
Self-contained, robust implementation using NumPy, Scikit-Learn, and Matplotlib.
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def calculate_purity(y_true, y_pred):
    """Calculate cluster purity."""
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def load_dataset(data_path="train_data.csv", features_path="train_features.csv"):
    """Load annotations and feature embeddings."""
    print(f"[1/7] Loading dataset from '{data_path}' and '{features_path}'...")
    labels = []
    filenames = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                labels.append(row[0].strip())
                filenames.append(row[1].strip())

    labels = np.array(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)

    print("      - Loading 512-dim feature embeddings...")
    features = np.loadtxt(features_path, delimiter=",", dtype=np.float32)

    print(f"      - Total samples: {features.shape[0]}")
    print(f"      - Feature dimensions: {features.shape[1]}")
    print(f"      - Distinct classes: {len(unique_classes)} (500 samples each)")
    return labels, filenames, features, unique_classes


def evaluate_model(X, y_true, y_pred, name):
    """Compute internal and external evaluation metrics."""
    mask = y_pred != -1
    n_clusters = len(set(y_pred[mask]))

    if n_clusters < 2:
        return {
            "Algorithm": name,
            "Clusters": n_clusters,
            "Silhouette": 0.0,
            "Calinski_Harabasz": 0.0,
            "Davies_Bouldin": 0.0,
            "ARI": 0.0,
            "NMI": 0.0,
            "Homogeneity": 0.0,
            "Completeness": 0.0,
            "V_Measure": 0.0,
            "Purity": 0.0,
        }

    sil = metrics.silhouette_score(X[mask], y_pred[mask])
    ch = metrics.calinski_harabasz_score(X[mask], y_pred[mask])
    db = metrics.davies_bouldin_score(X[mask], y_pred[mask])

    ari = metrics.adjusted_rand_score(y_true[mask], y_pred[mask])
    nmi = metrics.normalized_mutual_info_score(y_true[mask], y_pred[mask])
    homo, comp, vm = metrics.homogeneity_completeness_v_measure(y_true[mask], y_pred[mask])
    purity = calculate_purity(y_true[mask], y_pred[mask])

    return {
        "Algorithm": name,
        "Clusters": n_clusters,
        "Silhouette": round(float(sil), 4),
        "Calinski_Harabasz": round(float(ch), 2),
        "Davies_Bouldin": round(float(db), 4),
        "ARI": round(float(ari), 4),
        "NMI": round(float(nmi), 4),
        "Homogeneity": round(float(homo), 4),
        "Completeness": round(float(comp), 4),
        "V_Measure": round(float(vm), 4),
        "Purity": round(float(purity), 4),
    }


def run_pca_profiling(features, output_dir):
    """Profile cumulative variance with PCA."""
    print("[2/7] Profiling PCA explained variance...")
    pca = PCA().fit(features)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    n_80 = int(np.argmax(cum_var >= 0.80) + 1)
    n_90 = int(np.argmax(cum_var >= 0.90) + 1)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)

    print(f"      - 80% variance: {n_80} components")
    print(f"      - 90% variance: {n_90} components")
    print(f"      - 95% variance: {n_95} components")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, color="teal", linewidth=2.5, label="Cumulative Explained Variance")
    ax.axhline(y=0.80, color="grey", linestyle=":", label=f"80% Variance ({n_80} PCs)")
    ax.axhline(y=0.90, color="crimson", linestyle="--", label=f"90% Variance ({n_90} PCs)")
    ax.axhline(y=0.95, color="darkorange", linestyle="-.", label=f"95% Variance ({n_95} PCs)")
    ax.set_title("PCA Cumulative Explained Variance (Vehicle Embeddings)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Principal Components", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance Ratio", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.5)

    pca_path = os.path.join(output_dir, "pca_variance.png")
    fig.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {pca_path}")
    return n_90


def run_elbow_and_silhouette(X, output_dir, k_range=range(2, 21)):
    """Search for optimal k using Elbow (Inertia) and Silhouette Scores."""
    print(f"[3/7] Exploring optimal k across k={list(k_range)}...")
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        preds = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = metrics.silhouette_score(X, preds)
        sil_scores.append(sil)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(k_range, inertias, marker="o", color="royalblue", linewidth=2)
    ax1.set_title("Elbow Method (Inertia / WCSS)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.axvline(x=10, color="crimson", linestyle="--", label="Ground Truth (k=10)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(k_range, sil_scores, marker="s", color="darkorange", linewidth=2)
    ax2.set_title("Silhouette Score vs k", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.axvline(x=10, color="crimson", linestyle="--", label="Ground Truth (k=10)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    elbow_path = os.path.join(output_dir, "elbow_silhouette.png")
    fig.savefig(elbow_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {elbow_path}")


def main():
    total_start = time.time()
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Data
    labels, filenames, features, unique_classes = load_dataset()
    n_classes = len(unique_classes)

    # 2. Normalization: L2 Norm (Unit Hypersphere) & Standardization
    X_l2 = normalize(features, norm="l2")
    X_std = StandardScaler().fit_transform(features)

    # 3. PCA Profiling
    run_pca_profiling(features, figures_dir)

    # 4. Optimal k search
    run_elbow_and_silhouette(X_l2, figures_dir)

    # 5. Model Execution & Evaluation
    print(f"[4/7] Training clustering models at k={n_classes}...")
    models_to_test = [
        ("K-Means (L2-norm)", KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=10), X_l2),
        ("K-Means (StandardScaler)", KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=10), X_std),
        ("MiniBatch K-Means", MiniBatchKMeans(n_clusters=n_classes, random_state=RANDOM_STATE, batch_size=256, n_init=10), X_l2),
        ("Agglomerative (Ward)", AgglomerativeClustering(n_clusters=n_classes, linkage="ward"), X_l2),
        ("Agglomerative (Cosine-Average)", AgglomerativeClustering(n_clusters=n_classes, metric="cosine", linkage="average"), features),
        ("Gaussian Mixture (Diag GMM)", GaussianMixture(n_components=n_classes, covariance_type="diag", random_state=RANDOM_STATE), X_l2),
        ("DBSCAN (eps=0.45, min=10)", DBSCAN(eps=0.45, min_samples=10, metric="euclidean"), X_l2),
    ]

    results = []
    predictions = {}

    for name, model, data in models_to_test:
        t0 = time.time()
        preds = model.fit_predict(data)
        predictions[name] = preds
        metrics_dict = evaluate_model(data, labels, preds, name)
        results.append(metrics_dict)
        dt = time.time() - t0
        print(f"      - {name:30s} done in {dt:.2f}s | ARI={metrics_dict['ARI']:.4f} | NMI={metrics_dict['NMI']:.4f} | Purity={metrics_dict['Purity']:.4f}")

    # Save metrics to CSV
    metrics_csv_path = os.path.join(reports_dir, "clustering_metrics.csv")
    fieldnames = list(results[0].keys())
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[5/7] Metric evaluation table saved to '{metrics_csv_path}'")

    # 6. Dimensionality Reduction (t-SNE) for Cluster Visualization
    print("[6/7] Computing 2D t-SNE projection for 5,000 samples (25-35 seconds)...")
    tsne_start = time.time()
    tsne = TSNE(n_components=2, perplexity=35, random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_l2)
    print(f"      - t-SNE completed in {time.time() - tsne_start:.1f}s")

    print("      - Generating 2x2 comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    # Color palette
    colors = plt.cm.tab10.colors

    def plot_scatter(ax, coords, labels_arr, title, is_ground_truth=False):
        unique_vals = np.unique(labels_arr)
        for i, val in enumerate(unique_vals):
            mask = labels_arr == val
            color = colors[i % len(colors)]
            label_text = val.replace("_", " ") if is_ground_truth else f"Cluster {val}"
            ax.scatter(coords[mask, 0], coords[mask, 1], label=label_text, s=12, alpha=0.7, color=color)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plot_scatter(axes[0, 0], X_tsne, labels, "Ground Truth Vehicle Classes", is_ground_truth=True)
    plot_scatter(axes[0, 1], X_tsne, predictions["K-Means (L2-norm)"], "K-Means Clusters (L2-norm)")
    plot_scatter(axes[1, 0], X_tsne, predictions["Agglomerative (Ward)"], "Agglomerative (Ward Linkage)")
    plot_scatter(axes[1, 1], X_tsne, predictions["Gaussian Mixture (Diag GMM)"], "Gaussian Mixture Model (GMM)")

    plt.tight_layout()
    tsne_fig_path = os.path.join(figures_dir, "tsne_clustering_comparison.png")
    fig.savefig(tsne_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {tsne_fig_path}")

    # 7. Contingency Matrix Heatmap for K-Means (L2-norm)
    print("[7/7] Plotting Contingency Heatmap (Ground Truth vs K-Means)...")
    km_preds = predictions["K-Means (L2-norm)"]
    contingency = metrics.cluster.contingency_matrix(labels, km_preds)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cax = ax.imshow(contingency, cmap="Blues", interpolation="nearest", aspect="auto")
    fig.colorbar(cax)

    class_names = [c.replace("_", " ") for c in unique_classes]
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"Cluster {i}" for i in range(n_classes)], rotation=45, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)

    # Annotate counts inside cells
    for i in range(n_classes):
        for j in range(n_classes):
            val = contingency[i, j]
            color = "white" if val > contingency.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Contingency Matrix: Ground Truth Vehicle Models vs K-Means Clusters", fontsize=12, fontweight="bold")
    ax.set_xlabel("Assigned Cluster ID", fontsize=11)
    ax.set_ylabel("True Vehicle Model", fontsize=11)
    plt.tight_layout()

    contingency_path = os.path.join(figures_dir, "kmeans_contingency_matrix.png")
    fig.savefig(contingency_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {contingency_path}")

    total_time = time.time() - total_start
    print(f"\n[SUCCESS] Pipeline execution finished in {total_time:.1f} seconds!")


if __name__ == "__main__":
    main()

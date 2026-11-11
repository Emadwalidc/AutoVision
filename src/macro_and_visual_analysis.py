"""
AutoVision - Macro-Category Clustering & Visual Prototype Inspection
Phase 1: Diagnostic Deep Dive

1. Maps 10 fine-grained vehicle classes to 4 macro body categories:
   - Sedan (3,000 samples)
   - Minivan (1,000 samples)
   - SUV (500 samples)
   - Coupe / Sports (500 samples)
2. Runs K-Means and Agglomerative Clustering on L2-normalized features at k=4.
3. Computes comprehensive evaluation metrics comparing k=10 (Fine-grained) vs k=4 (Macro).
4. Generates a 4x4 Macro Contingency Matrix Heatmap.
5. Extracts prototypical vehicle images for each cluster and builds a visual montage.
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Mapping from fine-grained class to macro vehicle body category
MACRO_MAPPING = {
    "chevrolet_impala_2004-2005": "Sedan",
    "ford_taurus_2002-2003": "Sedan",
    "honda_accord_2005-2006": "Sedan",
    "nissan_altima_2005-2006": "Sedan",
    "toyota_camry_2002-2003": "Sedan",
    "volkswagen_jetta_2002-2003": "Sedan",
    "dodge_grand caravan_2005-2006": "Minivan",
    "honda_odyssey_2005-2006": "Minivan",
    "ford_explorer_2001-2002": "SUV",
    "ford_mustang_2000-2001": "Coupe",
}


def calculate_purity(y_true, y_pred):
    """Compute cluster purity."""
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def load_data(data_path="train_data.csv", features_path="train_features.csv"):
    """Load annotations, map to macro-categories, and load feature representations."""
    print("[1/5] Loading dataset and mapping to macro body categories...")
    fine_labels = []
    filenames = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                fine_labels.append(row[0].strip())
                filenames.append(row[1].strip())

    fine_labels = np.array(fine_labels)
    macro_labels = np.array([MACRO_MAPPING[label] for label in fine_labels])

    features = np.loadtxt(features_path, delimiter=",", dtype=np.float32)
    X_l2 = normalize(features, norm="l2")

    macro_unique, macro_counts = np.unique(macro_labels, return_counts=True)
    for cat, count in zip(macro_unique, macro_counts):
        print(f"      - Category: {cat:10s} | Samples: {count}")

    return fine_labels, macro_labels, filenames, X_l2, macro_unique


def run_macro_clustering(X_l2, macro_labels, macro_unique, output_dir):
    """Run K-Means and Agglomerative Clustering for k=4 (Macro Categories)."""
    print("[2/5] Running clustering at k=4 (Macro Categories)...")
    k = len(macro_unique)

    # 1. K-Means
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km_preds = km.fit_predict(X_l2)

    # 2. Agglomerative (Ward Linkage)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_preds = agg.fit_predict(X_l2)

    def compute_all_metrics(y_true, y_pred, name):
        sil = metrics.silhouette_score(X_l2, y_pred)
        ch = metrics.calinski_harabasz_score(X_l2, y_pred)
        db = metrics.davies_bouldin_score(X_l2, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        homo, comp, vm = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
        purity = calculate_purity(y_true, y_pred)
        return {
            "Level": "Macro (k=4)",
            "Algorithm": name,
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

    km_metrics = compute_all_metrics(macro_labels, km_preds, "K-Means (L2-norm)")
    agg_metrics = compute_all_metrics(macro_labels, agg_preds, "Agglomerative (Ward)")

    print(f"      - K-Means (Macro k=4)      : ARI={km_metrics['ARI']:.4f} | NMI={km_metrics['NMI']:.4f} | Purity={km_metrics['Purity']:.4f}")
    print(f"      - Agglomerative (Macro k=4): ARI={agg_metrics['ARI']:.4f} | NMI={agg_metrics['NMI']:.4f} | Purity={agg_metrics['Purity']:.4f}")

    # Generate Macro Contingency Heatmap
    contingency = metrics.cluster.contingency_matrix(macro_labels, km_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(contingency, cmap="Blues", interpolation="nearest", aspect="auto")
    fig.colorbar(cax)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Cluster {i}" for i in range(k)], fontsize=10)
    ax.set_yticks(range(len(macro_unique)))
    ax.set_yticklabels(macro_unique, fontsize=10)

    for i in range(len(macro_unique)):
        for j in range(k):
            val = contingency[i, j]
            color = "white" if val > contingency.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title("Contingency Matrix: Macro Body Categories vs K-Means (k=4)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Assigned Cluster ID", fontsize=11)
    ax.set_ylabel("True Macro Vehicle Category", fontsize=11)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "macro_clustering_contingency.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {heatmap_path}")

    return [km_metrics, agg_metrics], km, km_preds


def compare_fine_vs_macro(macro_metrics, output_dir):
    """Plot side-by-side comparison between Fine-Grained (k=10) and Macro (k=4)."""
    print("[3/5] Generating Fine-Grained (k=10) vs Macro (k=4) comparison plot...")
    # Fine-grained values from previous run
    fine_km_ari = 0.1233
    fine_km_nmi = 0.2323
    fine_km_purity = 0.2934

    macro_km = macro_metrics[0]
    macro_km_ari = macro_km["ARI"]
    macro_km_nmi = macro_km["NMI"]
    macro_km_purity = macro_km["Purity"]

    categories = ["Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)", "Cluster Purity"]
    fine_scores = [fine_km_ari, fine_km_nmi, fine_km_purity]
    macro_scores = [macro_km_ari, macro_km_nmi, macro_km_purity]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, fine_scores, width, label="Fine-Grained (10 Specific Models)", color="#4A90E2")
    rects2 = ax.bar(x + width/2, macro_scores, width, label="Macro-Category (4 Body Types)", color="#50E3C2")

    ax.set_title("Clustering Quality: Fine-Grained (k=10) vs. Macro-Body (k=4)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Annotate values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(output_dir, "macro_vs_fine_comparison.png")
    fig.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {comp_path}")


def build_visual_prototype_montage(fine_labels, macro_labels, filenames, X_l2, km_model, km_preds, output_dir, images_dir="train_images"):
    """Extract sample images closest to cluster centroids and display in visual montage."""
    print("[4/5] Extracting visual prototypical images from clusters...")
    centroids = km_model.cluster_centers_
    k = len(centroids)
    n_prototypes = 5

    fig, axes = plt.subplots(k, n_prototypes, figsize=(15, 10))

    for cluster_id in range(k):
        indices = np.where(km_preds == cluster_id)[0]
        c_samples = X_l2[indices]

        # Calculate distance to cluster centroid
        dists = np.linalg.norm(c_samples - centroids[cluster_id], axis=1)
        sorted_order = np.argsort(dists)
        top_indices = indices[sorted_order[:n_prototypes]]

        # Determine dominant macro and fine class
        cat_counts = {}
        for idx in indices:
            cat = macro_labels[idx]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        dom_cat = max(cat_counts, key=cat_counts.get)
        cat_pct = (cat_counts[dom_cat] / len(indices)) * 100

        for col_idx, sample_idx in enumerate(top_indices):
            ax = axes[cluster_id, col_idx]
            img_class = fine_labels[sample_idx]
            img_file = filenames[sample_idx]
            img_path = os.path.join(images_dir, img_class, img_file)

            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")

            short_model = img_class.split("_")[1].capitalize()
            ax.set_title(f"{short_model} ({macro_labels[sample_idx]})", fontsize=8)
            ax.axis("off")

            if col_idx == 0:
                ax.text(-0.25, 0.5, f"Cluster {cluster_id}\nDominant:\n{dom_cat} ({cat_pct:.0f}%)",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        va="center", ha="right")

    plt.suptitle("AutoVision: Prototypical Vehicle Images per Macro Cluster (k=4)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    montage_path = os.path.join(output_dir, "macro_cluster_prototypes.png")
    fig.savefig(montage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      - Saved: {montage_path}")


def main():
    start_time = time.time()
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    fine_labels, macro_labels, filenames, X_l2, macro_unique = load_data()

    # Run macro clustering
    macro_metrics, km_model, km_preds = run_macro_clustering(X_l2, macro_labels, macro_unique, figures_dir)

    # Compare macro vs fine
    compare_fine_vs_macro(macro_metrics, figures_dir)

    # Build visual prototype montage
    build_visual_prototype_montage(fine_labels, macro_labels, filenames, X_l2, km_model, km_preds, figures_dir)

    # Save metrics to CSV
    metrics_path = "reports/macro_clustering_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(macro_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(macro_metrics)
    print(f"[5/5] Saved macro metrics table to '{metrics_path}'")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Macro analysis and visual inspection completed in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()


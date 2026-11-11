"""
AutoVision - Advanced Clustering Optimization (Phase 1 Deep Dive)
Lead: Senior Staff ML Engineer / Data Scientist

Implements and benchmarks:
1. Baseline: Raw L2 K-Means (k=10).
2. Experiment 1: Power Transform / Signed Square-Root (SSR) + L2 Normalization.
3. Experiment 2: Dimensionality Optimization on SSR Features (PCA d in [32, 64, 96, 128, 160]).
4. Experiment 3: Spectral Manifold Graph Embedding (Laplacian nearest-neighbors).
5. Experiment 4: Enhanced Gaussian Mixture Models (Diag vs Tied covariance).
6. Experiment 5: Two-Stage Hierarchical Macro-to-Micro Clustering.
7. Verification Dashboard: Before/After comparison, metric export, and publication-ready plots.
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def calculate_purity(y_true, y_pred):
    """Compute cluster purity."""
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(contingency, axis=0)) / np.sum(contingency))


def load_dataset(data_path="train_data.csv", features_path="train_features.csv"):
    """Load dataset annotations and raw 512-D embeddings."""
    print("[1/6] Loading vehicle annotations and 512-D features...")
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
    print(f"      - Samples: {features.shape[0]} | Dimensions: {features.shape[1]} | Classes: {len(unique_classes)}")
    return labels, filenames, features, unique_classes


def evaluate(X_repr, y_true, y_pred, name):
    """Compute comprehensive internal and external validation metrics."""
    sil = metrics.silhouette_score(X_repr, y_pred)
    ch = metrics.calinski_harabasz_score(X_repr, y_pred)
    db = metrics.davies_bouldin_score(X_repr, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    homo, comp, vm = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    purity = calculate_purity(y_true, y_pred)

    return {
        "Experiment": name,
        "Clusters": len(np.unique(y_pred)),
        "ARI": round(float(ari), 4),
        "NMI": round(float(nmi), 4),
        "Purity": round(float(purity), 4),
        "Silhouette": round(float(sil), 4),
        "Davies_Bouldin": round(float(db), 4),
        "Calinski_Harabasz": round(float(ch), 2),
        "Homogeneity": round(float(homo), 4),
        "Completeness": round(float(comp), 4),
        "V_Measure": round(float(vm), 4),
    }


def power_transform_ssr(features, gamma=0.5):
    """Apply Signed Square-Root (SSR) Power Normalization followed by L2 normalization."""
    # x_ssr = sign(x) * |x|^gamma
    features_ssr = np.sign(features) * (np.abs(features) ** gamma)
    features_ssr_l2 = normalize(features_ssr, norm="l2")
    return features_ssr_l2


def run_experiment_1_power_transforms(raw_features, y_true):
    """Benchmark raw vs L2 vs SSR + L2 transforms."""
    print("[2/6] Experiment 1: Evaluating Feature Conditioning & Power Transforms...")
    results = []

    # A. Raw L2
    X_l2 = normalize(raw_features, norm="l2")
    km_l2 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=15)
    pred_l2 = km_l2.fit_predict(X_l2)
    results.append(evaluate(X_l2, y_true, pred_l2, "Baseline: Raw L2 K-Means"))

    # B. SSR (gamma=0.5) + L2
    X_ssr_l2 = power_transform_ssr(raw_features, gamma=0.5)
    km_ssr = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=15)
    pred_ssr = km_ssr.fit_predict(X_ssr_l2)
    results.append(evaluate(X_ssr_l2, y_true, pred_ssr, "Exp 1: SSR Power-Norm (gamma=0.5) + L2"))

    # C. Strong SSR (gamma=0.35) + L2
    X_ssr_strong = power_transform_ssr(raw_features, gamma=0.35)
    km_ssr_strong = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=15)
    pred_ssr_strong = km_ssr_strong.fit_predict(X_ssr_strong)
    results.append(evaluate(X_ssr_strong, y_true, pred_ssr_strong, "Exp 1: Strong SSR (gamma=0.35) + L2"))

    for r in results:
        print(f"      - {r['Experiment']:40s} | ARI={r['ARI']:.4f} | NMI={r['NMI']:.4f} | Purity={r['Purity']:.4f} | Sil={r['Silhouette']:.4f}")

    return results, X_ssr_l2


def run_experiment_2_pca_denoising(X_ssr_l2, y_true):
    """Benchmark dimensionality reduction across multiple subspace dimensions."""
    print("[3/6] Experiment 2: Dimensionality Subspace Denoising (PCA on SSR)...")
    results = []
    best_ari = -1.0
    best_d = None
    best_X_pca = None

    dims_to_test = [32, 64, 96, 128, 160, 256]

    for d in dims_to_test:
        pca = PCA(n_components=d, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_ssr_l2)
        X_pca = normalize(X_pca, norm="l2")

        km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=20)
        preds = km.fit_predict(X_pca)

        res = evaluate(X_pca, y_true, preds, f"Exp 2: SSR + PCA ({d}D) + L2")
        results.append(res)
        print(f"      - Dim {d:3d} : ARI={res['ARI']:.4f} | NMI={res['NMI']:.4f} | Purity={res['Purity']:.4f} | Sil={res['Silhouette']:.4f}")

        if res['ARI'] > best_ari:
            best_ari = res['ARI']
            best_d = d
            best_X_pca = X_pca

    print(f"      -> Optimal PCA subspace: d={best_d} with ARI={best_ari:.4f}")
    return results, best_d, best_X_pca


def run_experiment_3_spectral_manifold(X_optimal, y_true):
    """Test Spectral Graph Embedding with k-NN affinity to capture non-linear manifolds."""
    print("[4/6] Experiment 3: Non-Linear Manifold Learning (Spectral Embedding)...")
    results = []

    for n_neighbors in [15, 30, 50]:
        se = SpectralEmbedding(n_components=10, affinity="nearest_neighbors", n_neighbors=n_neighbors, random_state=RANDOM_STATE)
        X_spectral = se.fit_transform(X_optimal)
        X_spectral = normalize(X_spectral, norm="l2")

        km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=25)
        preds = km.fit_predict(X_spectral)

        res = evaluate(X_spectral, y_true, preds, f"Exp 3: Spectral (kNN={n_neighbors}) + K-Means")
        results.append(res)
        print(f"      - kNN={n_neighbors:2d}: ARI={res['ARI']:.4f} | NMI={res['NMI']:.4f} | Purity={res['Purity']:.4f} | Sil={res['Silhouette']:.4f}")

    return results


def run_experiment_4_advanced_algorithms(X_optimal, y_true):
    """Benchmark Agglomerative with Average/Complete linkage and Gaussian Mixture Models."""
    print("[5/6] Experiment 4: Advanced Density & Linkage Algorithms...")
    results = []

    # A. Agglomerative (Ward)
    agg_ward = AgglomerativeClustering(n_clusters=10, linkage="ward")
    pred_ward = agg_ward.fit_predict(X_optimal)
    results.append(evaluate(X_optimal, y_true, pred_ward, "Exp 4: Agglomerative (Ward)"))

    # B. Agglomerative (Average Linkage with Cosine Metric)
    agg_avg = AgglomerativeClustering(n_clusters=10, metric="cosine", linkage="average")
    pred_avg = agg_avg.fit_predict(X_optimal)
    results.append(evaluate(X_optimal, y_true, pred_avg, "Exp 4: Agglomerative (Cosine-Average)"))

    # C. Gaussian Mixture Model (Diag Covariance)
    gmm_diag = GaussianMixture(n_components=10, covariance_type="diag", random_state=RANDOM_STATE, n_init=3)
    pred_gmm = gmm_diag.fit_predict(X_optimal)
    results.append(evaluate(X_optimal, y_true, pred_gmm, "Exp 4: Gaussian Mixture (Diag GMM)"))

    # D. Gaussian Mixture Model (Tied Covariance)
    gmm_tied = GaussianMixture(n_components=10, covariance_type="tied", random_state=RANDOM_STATE, n_init=3)
    pred_tied = gmm_tied.fit_predict(X_optimal)
    results.append(evaluate(X_optimal, y_true, pred_tied, "Exp 4: Gaussian Mixture (Tied GMM)"))

    # E. K-Means++ with high restarts (n_init=100)
    km_100 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=100)
    pred_km100 = km_100.fit_predict(X_optimal)
    results.append(evaluate(X_optimal, y_true, pred_km100, "Exp 4: K-Means++ (n_init=100)"))

    for r in results:
        print(f"      - {r['Experiment']:38s} | ARI={r['ARI']:.4f} | NMI={r['NMI']:.4f} | Purity={r['Purity']:.4f} | Sil={r['Silhouette']:.4f}")

    return results, pred_km100


def run_experiment_5_two_stage_hierarchical(X_optimal, y_true):
    """
    Two-Stage Hierarchical Clustering:
    Stage 1: Partition into 4 macro-clusters (Minivans, SUVs, Coupes, Sedans).
    Stage 2: Within the dominant Sedan cluster (containing ~3,000 samples), run specialized
             intra-sedan micro-clustering to separate the 6 sedan classes.
    """
    print("[6/6] Experiment 5: Two-Stage Hierarchical Macro-to-Micro Clustering...")

    # Stage 1: Macro partition
    km_macro = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    macro_preds = km_macro.fit_predict(X_optimal)

    # Identify cluster with largest membership (the sedans)
    cluster_counts = np.bincount(macro_preds)
    sedan_cluster_id = int(np.argmax(cluster_counts))

    final_preds = np.zeros_like(macro_preds)
    current_label = 0

    # Assign non-sedan macro clusters distinct labels
    for c_id in range(4):
        if c_id != sedan_cluster_id:
            mask = macro_preds == c_id
            final_preds[mask] = current_label
            current_label += 1

    # Stage 2: Micro-cluster inside the sedan partition into 7 subclasses (total = 3 non-sedan + 7 sedan = 10)
    sedan_mask = macro_preds == sedan_cluster_id
    X_sedans = X_optimal[sedan_mask]

    km_micro = KMeans(n_clusters=7, random_state=RANDOM_STATE, n_init=25)
    sedan_sub_preds = km_micro.fit_predict(X_sedans)

    for sub_id in range(7):
        sub_mask = sedan_sub_preds == sub_id
        final_preds[sedan_mask] = np.where(sub_mask, current_label + sub_id, final_preds[sedan_mask])

    res = evaluate(X_optimal, y_true, final_preds, "Exp 5: Two-Stage Macro-to-Micro Clustering")
    print(f"      - {res['Experiment']:38s} | ARI={res['ARI']:.4f} | NMI={res['NMI']:.4f} | Purity={res['Purity']:.4f} | Sil={res['Silhouette']:.4f}")
    return [res], final_preds


def plot_optimization_dashboard(all_results, output_dir="reports/figures"):
    """Plot an executive comparison chart across all optimization phases."""
    os.makedirs(output_dir, exist_ok=True)

    # Select representative milestone models
    milestones = [
        ("Baseline (Raw L2)", next(r for r in all_results if "Baseline" in r["Experiment"])),
        ("Exp 1 (SSR Power-Norm)", next(r for r in all_results if "SSR Power-Norm" in r["Experiment"])),
        ("Exp 2 (Optimal PCA)", max([r for r in all_results if "Exp 2" in r["Experiment"]], key=lambda x: x["ARI"])),
        ("Exp 3 (Spectral kNN=50)", next(r for r in all_results if "kNN=50" in r["Experiment"])),
        ("Exp 4 (K-Means++ 100)", next(r for r in all_results if "n_init=100" in r["Experiment"])),
        ("Exp 5 (Two-Stage Macro/Micro)", next(r for r in all_results if "Two-Stage" in r["Experiment"])),
    ]

    labels = [m[0] for m in milestones]
    aris = [m[1]["ARI"] for m in milestones]
    nmis = [m[1]["NMI"] for m in milestones]
    purities = [m[1]["Purity"] for m in milestones]
    silhouettes = [m[1]["Silhouette"] for m in milestones]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#BD10E0", "#9013FE", "#D0021B"]

    # 1. ARI
    bars1 = ax1.bar(labels, aris, color=colors, width=0.55)
    ax1.set_title("Adjusted Rand Index (ARI) Across Optimization Phases", fontweight="bold")
    ax1.set_ylabel("ARI Score")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # 2. NMI
    bars2 = ax2.bar(labels, nmis, color=colors, width=0.55)
    ax2.set_title("Normalized Mutual Information (NMI)", fontweight="bold")
    ax2.set_ylabel("NMI Score")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # 3. Cluster Purity
    bars3 = ax3.bar(labels, [p * 100 for p in purities], color=colors, width=0.55)
    ax3.set_title("Cluster Purity (%)", fontweight="bold")
    ax3.set_ylabel("Purity %")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars3:
        h = b.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # 4. Silhouette Score
    bars4 = ax4.bar(labels, silhouettes, color=colors, width=0.55)
    ax4.set_title("Internal Silhouette Score", fontweight="bold")
    ax4.set_ylabel("Silhouette")
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars4:
        h = b.get_height()
        ax4.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.suptitle("AutoVision: Clustering Optimization Dashboard (Before vs. After)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "clustering_optimization_dashboard.png")
    fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[VERIFY] Saved Optimization Dashboard to '{dashboard_path}'")


def main():
    total_start = time.time()
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    labels, filenames, raw_features, unique_classes = load_dataset()

    all_results = []

    # Phase 1: Feature Conditioning & SSR Power-Norm
    res1, X_ssr_l2 = run_experiment_1_power_transforms(raw_features, labels)
    all_results.extend(res1)

    # Phase 2: Dimensionality Subspace Optimization (PCA)
    res2, best_d, best_X_pca = run_experiment_2_pca_denoising(X_ssr_l2, labels)
    all_results.extend(res2)

    # Phase 3: Spectral Manifold Learning
    res3 = run_experiment_3_spectral_manifold(best_X_pca, labels)
    all_results.extend(res3)

    # Phase 4: Advanced Density & Linkage Algorithms
    res4, best_preds = run_experiment_4_advanced_algorithms(best_X_pca, labels)
    all_results.extend(res4)

    # Phase 5: Two-Stage Hierarchical Macro-to-Micro
    res5, final_preds = run_experiment_5_two_stage_hierarchical(best_X_pca, labels)
    all_results.extend(res5)

    # Export all results to CSV
    metrics_path = os.path.join(reports_dir, "optimization_benchmark_metrics.csv")
    fieldnames = list(all_results[0].keys())
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[REPORT] Saved full optimization metrics table to '{metrics_path}'")

    # Generate Executive Verification Dashboard Plot
    plot_optimization_dashboard(all_results, figures_dir)

    # Print Leaderboard
    print("\n" + "=" * 80)
    print("TOP CLUSTERING CONFIGURATIONS LEADERBOARD (SORTED BY ARI):")
    print("=" * 80)
    sorted_results = sorted(all_results, key=lambda x: x["ARI"], reverse=True)
    header = f"{'Experiment':42s} | {'ARI':6s} | {'NMI':6s} | {'Purity':6s} | {'Silhouette':10s}"
    print(header)
    print("-" * 80)
    for r in sorted_results[:8]:
        print(f"{r['Experiment']:42s} | {r['ARI']:6.4f} | {r['NMI']:6.4f} | {r['Purity']*100:5.1f}% | {r['Silhouette']:10.4f}")
    print("=" * 80)

    elapsed = time.time() - total_start
    print(f"\n[COMPLETE] Optimization and verification finished in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()


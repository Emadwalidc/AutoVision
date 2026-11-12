"""
AutoVision - SOTA Unsupervised Clustering Benchmark
Применяет PCA-Whitening, UMAP-манифолд и Spectral Clustering к новым 512-D эмбеддингам.
"""

import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def calculate_purity(y_true, y_pred):
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(contingency, axis=0)) / np.sum(contingency))


def calculate_hungarian_acc(y_true, y_pred):
    """Точность кластеризации (Hungarian Matching Accuracy) 1-к-1."""
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum() / np.sum(contingency))


def whitened_projection(X, n_components=64):
    """Центрирование + PCA Whitening + L2 проекция на гиперсферу."""
    X_l2 = normalize(X, norm="l2")
    X_mean = X_l2 - np.mean(X_l2, axis=0, keepdims=True)
    pca = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
    X_w = pca.fit_transform(X_mean)
    return normalize(X_w, norm="l2")


def evaluate_all(X_space, y_true, y_pred, name):
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    purity = calculate_purity(y_true, y_pred)
    acc = calculate_hungarian_acc(y_true, y_pred)
    sil = metrics.silhouette_score(X_space, y_pred)
    print(
        f"{name:42s} | ARI: {ari:.4f} | NMI: {nmi:.4f} | Purity: {purity*100:5.1f}% | ACC: {acc*100:5.1f}% | Sil: {sil:.4f}"
    )
    return {
        "Method": name,
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Purity": round(purity, 4),
        "ACC": round(acc, 4),
        "Silhouette": round(sil, 4),
    }


def main():
    with open("train_data.csv", "r", encoding="utf-8") as f:
        y = np.array([r[0].strip() for r in csv.reader(f) if r])
    X = np.loadtxt("train_features.csv", delimiter=",", dtype=np.float32)
    X_l2 = normalize(X, norm="l2")

    print("=" * 105)
    print("AUTOVISION: SOTA CLUSTERING BENCHMARK (k=10 VEHICLE MODELS)")
    print("=" * 105)

    results = []

    # 1. Raw L2 K-Means
    km_raw = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=30).fit_predict(
        X_l2
    )
    results.append(evaluate_all(X_l2, y, km_raw, "1. Baseline L2 K-Means"))

    # 2. PCA Whitening (64D) + K-Means / GMM
    X_w64 = whitened_projection(X, n_components=64)
    km_w64 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=50).fit_predict(
        X_w64
    )
    results.append(evaluate_all(X_w64, y, km_w64, "2. PCA-Whitened (64D) + K-Means"))

    gmm_w64 = GaussianMixture(
        n_components=10, covariance_type="tied", n_init=10, random_state=RANDOM_STATE
    ).fit_predict(X_w64)
    results.append(evaluate_all(X_w64, y, gmm_w64, "3. PCA-Whitened (64D) + Tied GMM"))

    # 3. Spectral Manifold на отбеленных признаках
    se = SpectralEmbedding(
        n_components=10,
        affinity="nearest_neighbors",
        n_neighbors=20,
        random_state=RANDOM_STATE,
    )
    X_spec = normalize(se.fit_transform(X_w64), norm="l2")
    km_spec = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=50).fit_predict(
        X_spec
    )
    results.append(
        evaluate_all(X_spec, y, km_spec, "4. Whitened + Spectral (kNN=20) + K-Means")
    )

    # 4. UMAP Non-Linear Manifold + K-Means (если установлен umap-learn)
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=30,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        X_umap = reducer.fit_transform(X_w64)
        km_umap = KMeans(
            n_clusters=10, random_state=RANDOM_STATE, n_init=50
        ).fit_predict(X_umap)
        results.append(
            evaluate_all(X_umap, y, km_umap, "5. Whitened + UMAP (Cosine) + K-Means")
        )

    os.makedirs("reports", exist_ok=True)
    with open(
        "reports/sota_clustering_metrics.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("=" * 105)


if __name__ == "__main__":
    main()

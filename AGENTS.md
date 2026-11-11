# AutoVision - Agent Guidelines & Project Instructions

Welcome to **AutoVision**, our final year graduation project (Bitirme Projesi). This document serves as the primary instructions and context repository for AI agents (Antigravity, Copilot, etc.) assisting on this codebase.

---

## 1. Project Overview & Repository

- **Repository**: [AutoVision (GitHub)](https://github.com/Emadwalidc/AutoVision.git)
- **Domain**: Computer Vision, Vehicle Analysis, Representation Learning & Detection.
- **Current Development Phase**: **Phase 1: Feature Clustering & Unsupervised Discovery**
- **Future Development Phase**: **Phase 2: YOLO Object Detection & End-to-End Pipeline**

---

## 2. Project Roadmap & Priorities

### Phase 1: Clustering & Unsupervised Feature Analysis (CURRENT FOCUS)
> [!IMPORTANT]
> **Our immediate mission is Clustering.** Do NOT jump to YOLO training or object detection models yet.
> The dataset currently present is a partial subset; more data will be incorporated progressively.

**Objectives & Tasks:**
1. **Feature Exploration**:
   - Analyze embeddings in `train_features.csv` (dimension, distributions, normalization, outliers).
   - Correlate feature vectors with vehicle class annotations in `train_data.csv`.
2. **Dimensionality Reduction**:
   - Implement PCA, t-SNE, and UMAP for 2D/3D visualization of feature spaces.
   - Assess separation and cluster compactness among car makes/models/years.
3. **Clustering Algorithms**:
   - Evaluate multiple clustering algorithms:
     - Partitioning: K-Means, MiniBatch K-Means
     - Density-based: DBSCAN, HDBSCAN
     - Hierarchical: Agglomerative Clustering (Ward, Complete, Average linkage)
     - Probabilistic: Gaussian Mixture Models (GMM)
4. **Evaluation & Metrics**:
   - **Internal metrics** (unsupervised): Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index.
   - **External metrics** (against ground truth `label` from `train_data.csv`): Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Homogeneity, Completeness, V-Measure, Cluster Purity.
5. **Visual Error & Cluster Inspection**:
   - Map clustered samples back to actual images in `train_images/` to visually verify cluster coherence and identify confusion patterns.

---

### Phase 2: YOLO Integration (FUTURE MILESTONE)
> [!NOTE]
> We will use **YOLO** (e.g. YOLOv8 / YOLOv11) for vehicle localization, bounding box detection, and fine-grained classification once the complete dataset is ready.
> When designing Phase 1 scripts and data pipelines, keep them modular so that YOLO dataset generators (e.g. YOLO format annotations, train/val/test splits) can be integrated seamlessly later.

---

## 3. Dataset Specification

Current workspace files:
- `train_data.csv`: CSV with no header (`<class_name>,<image_filename>`). Contains class labels for 5,000 samples across multiple car models (e.g., `chevrolet_impala_2004-2005`, `ford_explorer_2001-2002`).
- `train_features.csv`: Headerless matrix of pre-extracted feature embeddings (5,000 samples).
- `train_images/`: Directory containing subfolders named by vehicle class with individual image files.
- **Data Scaling Note**: More images and features will be added in upcoming project sprints. Keep code dynamic to handle varying sample counts, dimensions, and classes.

---

## 4. Environment & Execution Guidelines for Agents

- **Python Virtual Environment**: `.venv` located at project root (Python 3.14).
  - Executable path: `.\.venv\Scripts\python.exe`
  - Activation: `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.\.venv\Scripts\activate.bat` (CMD)
- **Jupyter Kernel**: Kernel spec `bitirme` (`Python 3.14 (bitirme)`).
- **Dependencies**: All dependencies must be tracked in `requirements.txt`. Before adding new libraries (e.g., `umap-learn`, `hdbscan`), ensure compatibility and append them to `requirements.txt`.
- **Git & Large Files**:
  - Raw images in `train_images/` (~211 MB) and large feature matrices should be handled carefully. Do not commit large binary blobs directly to Git without confirmation; prefer Git LFS or keeping large raw datasets local.
  - Always check `git status` and keep `.gitignore` updated.

---

## 5. Coding Standards & Agent Workflow Rules

1. **Self-Contained & Reproducible**:
   - Seed all random number generators (`np.random.seed(42)`, `random_state=42`) for reproducibility in clustering and dimensionality reduction.
2. **Modular Architecture**:
   - For exploratory analysis, use Jupyter Notebooks (`notebook.ipynb` or `notebooks/`).
   - For reusable logic (feature loaders, clustering pipelines, visualization functions, evaluation metric calculators), write clean Python modules under a dedicated `src/` directory.
3. **Meaningful Visualizations**:
   - Always label axes, include titles, add legends, and save high-resolution figures (e.g., `reports/figures/` or `output/`) when creating cluster plots or confusion matrices.
4. **Git Hygiene**:
   - Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
   - Never commit `.venv/`, `.ipynb_checkpoints/`, or temporary logs.

# Отчёт по извлечению и оценке признаков (AutoVision Feature Baseline)

Оценка выполнена строго по схеме без утечек (`train` → `val` для выбора модели → `test` + `few_shot`).

## 1. Сводная таблица базовых метрик (Test / Few-Shot)

| Backbone | Dim | Задача | Метод | N eval | Top-1 | Top-5 | Macro-F1 | MAE (лет) | Within ±2 г. |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openclip_vitb16` | 512 | Make | Nearest Prototype | 8,885 | 60.57% | 90.10% | 48.92% | — | — |
| `openclip_vitb16` | 512 | Make | Cosine kNN (k=5) | 8,885 | 96.48% | 99.32% | 93.86% | — | — |
| `openclip_vitb16` | 512 | Make | Linear Probe | 8,885 | 91.91% | 99.40% | 88.35% | — | — |
| `openclip_vitb16` | 512 | Model | Nearest Prototype | 8,885 | 64.09% | 89.24% | 55.79% | — | — |
| `openclip_vitb16` | 512 | Model | Cosine kNN (k=5) | 8,885 | 90.78% | 97.67% | 78.46% | — | — |
| `openclip_vitb16` | 512 | Model | Linear Probe | 8,885 | 90.52% | 98.65% | 80.58% | — | — |
| `openclip_vitb16` | 512 | Few-Shot Model | 1-Shot Prototype | 576 | 35.07% | 50.69% | — | — | — |
| `openclip_vitb16` | 512 | Year | Ridge (alpha=0.1) | 8,880 | — | — | — | 2.03 | 58.7% |
| `dinov2_vitb14` | 768 | Make | Nearest Prototype | 8,885 | 17.08% | 60.35% | 18.54% | — | — |
| `dinov2_vitb14` | 768 | Make | Cosine kNN (k=5) | 8,885 | 88.53% | 96.60% | 81.57% | — | — |
| `dinov2_vitb14` | 768 | Make | Linear Probe | 8,885 | 78.62% | 97.55% | 72.59% | — | — |
| `dinov2_vitb14` | 768 | Model | Nearest Prototype | 8,885 | 19.19% | 44.48% | 22.58% | — | — |
| `dinov2_vitb14` | 768 | Model | Cosine kNN (k=5) | 8,885 | 81.45% | 92.40% | 66.70% | — | — |
| `dinov2_vitb14` | 768 | Model | Linear Probe | 8,885 | 82.88% | 96.39% | 70.94% | — | — |
| `dinov2_vitb14` | 768 | Few-Shot Model | 1-Shot Prototype | 576 | 25.00% | 39.76% | — | — | — |
| `dinov2_vitb14` | 768 | Year | Ridge (alpha=0.1) | 8,880 | — | — | — | 1.98 | 61.2% |
| `fusion_openclip_dinov2` | 1280 | Make | Nearest Prototype | 8,885 | 42.97% | 84.57% | 37.34% | — | — |
| `fusion_openclip_dinov2` | 1280 | Make | Cosine kNN (k=5) | 8,885 | 96.09% | 98.95% | 89.39% | — | — |
| `fusion_openclip_dinov2` | 1280 | Make | Linear Probe | 8,885 | 93.81% | 99.63% | 90.87% | — | — |
| `fusion_openclip_dinov2` | 1280 | Model | Nearest Prototype | 8,885 | 45.19% | 78.89% | 43.42% | — | — |
| `fusion_openclip_dinov2` | 1280 | Model | Cosine kNN (k=5) | 8,885 | 90.49% | 97.07% | 77.97% | — | — |
| `fusion_openclip_dinov2` | 1280 | Model | Linear Probe | 8,885 | 93.35% | 99.02% | 84.84% | — | — |
| `fusion_openclip_dinov2` | 1280 | Few-Shot Model | 1-Shot Prototype | 576 | 29.17% | 44.10% | — | — | — |
| `fusion_openclip_dinov2` | 1280 | Year | Ridge (alpha=0.1) | 8,880 | — | — | — | 1.79 | 64.8% |

## 2. Абляция: влияние кропа YOLOv8n vs Сырой кадр (Test subset)

| Backbone | N | Марка Top-1 (YOLO → Raw) | Модель LP Top-1 (YOLO → Raw) | Модель Proto Top-1 (YOLO → Raw) | Год MAE (YOLO → Raw) |
| :--- | ---: | :--- | :--- | :--- | :--- |
| `openclip_vitb16` | 2,000 | 91.60% vs 89.50% | 90.30% vs 87.65% | 62.55% vs 62.05% | 1.99 vs 2.09 |
| `dinov2_vitb14` | 2,000 | 79.00% vs 77.40% | 85.20% vs 80.45% | 19.15% vs 19.35% | 1.98 vs 2.02 |
| `fusion_openclip_dinov2` | 2,000 | 93.50% vs 92.05% | 93.30% vs 91.50% | 44.40% vs 45.70% | 1.78 vs 1.83 |

## 3. Визуализации для диплома (лучший backbone: `fusion_openclip_dinov2`, t-SNE Silhouette=0.021)

- [Кривая объяснённой дисперсии PCA](figures/feature_pca_variance.png)
- [2D t-SNE проекция Топ-10 моделей](figures/feature_tsne_models.png)
- [Регрессия года выпуска (Predicted vs True)](figures/feature_year_regression.png)
- [Визуальный поиск Top-5 похожих авто](figures/feature_retrieval_montage.png) (`openclip_vitb16` — лучший Cosine kNN: 90.78% Model / 96.48% Make)

# AutoVision — Этап 5: Сравнение «До / После Fine-Tuning» и верификация ONNX

## 1. Сводная таблица «До (Frozen Baseline) vs После (Multi-Task Fine-Tuning)»

| Задача | Метрика | Frozen OpenCLIP (512-D) | Frozen Fusion (1280-D) | Fine-Tuned AutoVision (512-D) | Прирост vs OpenCLIP | Примечание |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Make | Top-1 (Head/LP) | 91.91% | 93.81% | **98.27%** | +6.36% |  |
| Make | Macro-F1 | 88.35% | 90.87% | **94.59%** | +6.24% |  |
| Model | Top-1 (Head/LP) | 90.52% | 93.35% | **96.38%** | +5.86% | CosFace Head |
| Model | Top-5 (Head/LP) | 98.65% | 99.02% | **98.92%** | +0.27% |  |
| Model | Macro-F1 | 80.58% | 84.84% | **89.39%** | +8.81% |  |
| Model | Nearest Prototype Top-1 | 64.09% | — | **96.16%** | +32.07% |  |
| Model | Cosine kNN (k=5) Top-1 | 90.78% | — | **97.15%** | +6.37% |  |
| Few-Shot Model | 1-Shot Prototype Top-1 | 35.07% | — | **39.06%** | +3.99% | queries=576 |
| Few-Shot Model | 1-Shot Prototype Top-5 | 50.69% | — | **71.35%** | +20.66% | queries=576 |
| Body Type | Top-1 (Head/LP) | 92.35% | 93.80% | **97.69%** | +5.34% |  |
| Body Type | Macro-F1 | 87.60% | 89.89% | **95.77%** | +8.17% | Cosine kNN baseline=0.9466 |
| Year | MAE (years, valid only) | 2.03 yr | 1.79 yr | **1.05 yr** | -0.9737 | Lower is better |
| Year | Within ±2 yr (%) | 58.72% | 64.85% | **87.56%** | +28.84% |  |
| Color | 15-Class Top-1 | 28.08% | — | **86.76%** | +58.68% | VCoR test |
| Color | 15-Class Macro-F1 | 20.95% | — | **84.13%** | +63.18% | VCoR test |
| Color | 12-Group Automotive Top-1 | 29.63% | — | **90.81%** | +61.18% | VCoR test |
| Representation | 2D t-SNE Silhouette | 0.0341 | — | **0.5321** | +0.4980 | Top-10 test classes |

## 2. Детализация по типам кузова (`Body Type`, Test)

| Тип кузова | Support | Recall | F1 |
| :--- | ---: | ---: | ---: |
| `Sedan` | 4547 | 98.00% | 98.32% |
| `SUV` | 1563 | 99.42% | 98.92% |
| `Pickup` | 369 | 98.64% | 98.11% |
| `Minivan` | 453 | 99.34% | 99.12% |
| `Coupe` | 755 | 93.77% | 95.42% |
| `Hatchback` | 760 | 98.95% | 97.60% |
| `Wagon` | 282 | 90.07% | 88.50% |
| `Convertible` | 156 | 91.03% | 90.16% |

## 3. Лучшие и наиболее сложные модели по F1

**Топ-10 лучших моделей:**

| Модель | Support | F1 |
| :--- | ---: | ---: |
| `toyota__sienna` | 182 | 100.00% |
| `volkswagen__beetle` | 149 | 100.00% |
| `toyota__prius` | 121 | 100.00% |
| `nissan__versa` | 95 | 100.00% |
| `pontiac__g6` | 92 | 100.00% |
| `mitsubishi__lancer` | 76 | 100.00% |
| `toyota__sequoia` | 70 | 100.00% |
| `saturn__vue` | 68 | 100.00% |
| `mercedes-benz__m-class` | 61 | 100.00% |
| `mazda__tribute` | 46 | 100.00% |

**10 наиболее сложных моделей:**

| Модель | Support | F1 |
| :--- | ---: | ---: |
| `mercedes-benz__slk-class` | 19 | 73.33% |
| `subaru__legacy` | 44 | 75.61% |
| `subaru__wrx` | 23 | 76.92% |
| `pontiac__firebird` | 46 | 79.01% |
| `toyota__pickup` | 18 | 80.95% |
| `suzuki__sx4` | 11 | 81.48% |
| `pontiac__transam` | 34 | 82.05% |
| `volvo__v70` | 27 | 83.33% |
| `volvo__xc70` | 18 | 85.71% |
| `ram__1500` | 11 | 85.71% |

## 4. Детализация по цветам (`VCoR Test`, 15 классов)

| Цвет | Группа | Support | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: |
| `Beige` | `Beige/Gold` | 90 | 46.67% | 54.90% |
| `Black` | `Black` | 87 | 90.80% | 92.94% |
| `Blue` | `Blue` | 159 | 97.48% | 97.79% |
| `Brown` | `Brown` | 121 | 81.82% | 83.54% |
| `Gold` | `Beige/Gold` | 45 | 60.00% | 61.36% |
| `Gray` | `Gray` | 92 | 86.96% | 82.05% |
| `Green` | `Green` | 121 | 98.35% | 97.14% |
| `Orange` | `Orange` | 114 | 92.98% | 92.58% |
| `Pink` | `Purple/Pink` | 103 | 94.17% | 96.04% |
| `Purple` | `Purple/Pink` | 115 | 96.52% | 96.94% |
| `Red` | `Red` | 136 | 95.59% | 93.19% |
| `Silver` | `Silver` | 77 | 63.64% | 69.50% |
| `Tan` | `Beige/Gold` | 86 | 76.74% | 62.86% |
| `White` | `White` | 86 | 88.37% | 86.86% |
| `Yellow` | `Yellow` | 124 | 91.94% | 94.21% |

## 5. Проверка инференса CPU и паритета ONNX

- Чекпоинт `.pt` найден: `True` (CPU медиана latency batch=1: `331.28 мс`)
- Модель `.onnx` найдена: `False` (CPU медиана latency batch=1: `None мс`)
- Максимальное расхождение выходов PyTorch vs ONNX (`max_abs_diff`): `None`

## 6. Сформированные графики

- `../figures/v1/stage5_training_curves.png`
- `../figures/v1/stage5_before_after_bars.png`
- `../figures/v1/stage5_tsne_before_after.png`
- `../figures/v1/stage5_body_type_confusion_matrix.png`
- `../figures/v1/stage5_year_dldl_evaluation.png`
- `../figures/v1/stage5_color_confusion_matrix.png`
- `../figures/v1/stage5_retrieval_montage_finetuned.png`

# AutoVision v2 — Итоговый отчёт по качеству моделей и графики для ВКР

## 1. Эволюция архитектуры и масштаба (AutoVision v1 -> v2)

| Параметр | AutoVision v1 | AutoVision v2 | Рост |
| :--- | ---: | ---: | ---: |
| Чистых изображений | 89,535 | **274,319** | **3.06x** |
| Закрытых марок | 22 | **50** | **2.27x** |
| Закрытых моделей | 225 | **617** | **2.74x** |
| Классов в прототипах (вкл. few-shot) | 342 | **738** | **2.16x** |
| Диапазон годов выпуска | 1950–2016 (67 лет) | **1950–2026 (77 лет)** | **1.15x** |
| Цветовой модуль | Голова поверх ViT-B/16 (15 кл.) | **ConvNeXt-Tiny + 16-D Masked Lab (12 кл.) + YOLOv8n-seg** | — |

## 2. Метрики Phase 1 по доменам (`autovision-v2-vitb16-multitask`)

| Домен оценки | N | Make Top-1 | Model Top-1 | Model Top-5 | Model Macro-F1 | Body Top-1 | Year MAE | Year +-2 yr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall Holdout** | 21,884 | 96.28% | **94.08%** | 97.52% | 92.21% | 94.59% | 1.25 yr | 90.76% |
| **Modern 2017–2026** | 6,385 | 96.88% | **95.00%** | — | — | 95.25% | **0.92 yr** | **97.82%** |
| **VMMRdb Street** | 7,605 | **98.54%** | **95.96%** | — | — | **97.80%** | 1.25 yr | 81.87% |
| *Для сравнения: AutoVision v1 (VMMRdb)* | 8,885 | 98.27% | 96.36% | 98.92% | 89.35% | 97.69% | 1.05 yr | 87.56% |

- **Nearest Prototype Top-1 по всем 738 классам**: **92.47%**.
- **Косинусный Silhouette (t-SNE, 12 разнообразных моделей из галереи)**: **0.691**.

## 3. Эксперт по цвету Phase 2 (`ConvNeXt-Tiny + Masked CIE-Lab`)

| Тестовый набор | N | Top-1 | Top-3 | Macro-F1 | Ключевые ошибки на VCoR |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **VCoR Test (уличные фото)** | 1,556 | **92.80%** | **98.39%** | **91.87%** | Black<->Brown: **0/0**, Silver<->Gray: 8/5, White->Silver: 3 |
| **DVM-CAR Test (студийные)** | 1,638 | **87.48%** | **96.09%** | **84.41%** | Баланс по 12 цветам с маской фона |
| *Для сравнения: v1 Color Head (VCoR)* | 1,556 | 86.76% | 98.33% | 84.13% | Без сегментации и CIE-Lab ветви |

## 4. Детализация по типам кузова (`Body Type`, Holdout N=21,884)

| Тип кузова | Support | Recall | F1 |
| :--- | ---: | ---: | ---: |
| `Sedan` | 8,574 | 95.03% | 95.31% |
| `SUV` | 5,510 | 98.35% | 98.06% |
| `Pickup` | 1,070 | 97.76% | 96.85% |
| `Minivan` | 826 | 98.43% | 97.83% |
| `Coupe` | 2,748 | 91.45% | 91.78% |
| `Hatchback` | 1,877 | 92.70% | 92.60% |
| `Wagon` | 567 | 76.19% | 77.07% |
| `Convertible` | 712 | 82.58% | 81.55% |

## 5. Точность определения года по эпохам (1950–2026)

| Эпоха | N (valid) | Within +-2 yr | Within +-3 yr | MAE (midpoint) |
| :--- | ---: | ---: | ---: | ---: |
| `1950-1989` | 1,860 | 87.96% | 91.45% | 2.03 yr |
| `1990-1999` | 2,563 | 84.82% | 93.80% | 1.55 yr |
| `2000-2009` | 8,174 | 89.00% | 96.55% | 1.18 yr |
| `2010-2016` | 2,869 | 87.14% | 94.32% | 1.38 yr |
| `2017-2026` | 6,385 | 97.82% | 98.89% | 0.92 yr |

## 6. Лучшие и наиболее сложные модели (из 617 закрытых моделей, support >= 15)

**Топ-10 лучших по F1:**

| Модель | Support | F1 |
| :--- | ---: | ---: |
| Toyota Sienna (`toyota__sienna`) | 188 | 100.00% |
| Volkswagen Beetle (`volkswagen__beetle`) | 159 | 100.00% |
| Nissan Versa (`nissan__versa`) | 92 | 100.00% |
| Kia Optima (`kia__optima`) | 69 | 100.00% |
| Saturn Vue (`saturn__vue`) | 62 | 100.00% |
| Smart Fortwo (`smart__fortwo`) | 53 | 100.00% |
| Mazda Tribute (`mazda__tribute`) | 45 | 100.00% |
| Nissan Rogue (`nissan__rogue`) | 45 | 100.00% |
| Volkswagen Touareg (`volkswagen__touareg`) | 45 | 100.00% |
| Mazda RX-8 (`mazda__rx8`) | 42 | 100.00% |

**10 наиболее сложных моделей (близкие поколения / ребеджи):**

| Модель | Support | F1 |
| :--- | ---: | ---: |
| Chevrolet Nova (`chevrolet__nova`) | 29 | 65.71% |
| Chevrolet Caprice (`chevrolet__caprice`) | 38 | 76.60% |
| Ford F-250/F-350 (`ford__f250-f350`) | 54 | 76.79% |
| Mercedes-Benz A-Class (`mercedes-benz__a-class`) | 23 | 76.92% |
| Mazda 626 (`mazda__626`) | 29 | 77.33% |
| Acura Integra (`acura__integra`) | 19 | 78.26% |
| Chevrolet S-10 (`chevrolet__s10`) | 19 | 80.00% |
| Toyota Pickup (`toyota__pickup`) | 44 | 80.39% |
| GMC Sierra 1500 (`gmc__sierra_1500`) | 20 | 80.95% |
| Pontiac Trans Am (`pontiac__transam`) | 40 | 81.72% |

## 7. 10 наиболее сложных марок (из 50)

| Марка | Support | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Isuzu | 13 | 100.00% | 76.47% |
| Lucid | 10 | 80.00% | 84.21% |
| GMC | 97 | 92.78% | 91.84% |
| Acura | 146 | 96.58% | 91.86% |
| Plymouth | 56 | 91.07% | 91.89% |
| Honda | 653 | 90.66% | 92.00% |
| Lincoln | 184 | 96.20% | 92.19% |
| Dodge | 287 | 93.73% | 92.60% |
| Chevrolet | 1,125 | 95.64% | 92.92% |
| Mazda | 491 | 96.74% | 93.14% |

## 8. Подготовленные графики (`reports/figures/v2/`)

- `../figures/v2/v2_evolution_comparison.png` — рост масштаба датасета и сравнение Baseline -> v1 -> v2.
- `../figures/v2/v2_domain_breakdown.png` — сравнение доменов Overall / Modern 2017–2026 / Street и Top-K метрик.
- `../figures/v2/v2_body_type_confusion_matrix.png` — матрица ошибок по 8 типам кузова (N=21,884).
- `../figures/v2/v2_makes_f1_ranking.png` — рейтинг F1-score по всем 50 маркам.
- `../figures/v2/v2_year_interval_evaluation.png` — точность предсказания года 1950–2026 по эпохам и scatter midpoint vs prediction.
- `../figures/v2/v2_color_training_curves.png` — кривые обучения Phase 2 Color Expert v2 (VCoR + DVM-CAR).
- `../figures/v2/v2_color_confusion_matrix.png` — матрица ошибок цветового эксперта на 12 автомобильных цветах.
- `../figures/v2/v2_tsne_embeddings.png` — 2D t-SNE эмбеддингов 512-D на 12 моделях из галереи v2.
- `../figures/v2/v2_retrieval_montage.png` — визуальный монтаж поиска топ-5 похожих автомобилей из `gallery_v2_thumbs.zip`.

# Аудит локального датасета (VMMRdb subset)

- **Источник:** `D:\Projects\bitirme\data` (режим строго read-only)
- **Время прогона:** 233.4 сек
- **Папок `<make>_<model>_<year>`:** 3,356
- **Всего изображений:** 91,678 (пропущено не-изображений: 0)

## 1. Нормализация таксономии

| Показатель | До очистки (raw) | После нормализации |
| :--- | ---: | ---: |
| Уникальных марок | 35 | 34 |
| Уникальных моделей (`make+model`) | 478 | 342 |
| Классов для основного обучения (≥15 чистых фото) | — | 225 |
| Few-shot классов (1..14 чистых фото) | — | 117 |
| Классов с 0 чистых фото | — | 0 |

## 2. Фильтрация шума и дедупликация (64-bit dHash)

| Категория | Количество фото | Доля |
| :--- | ---: | ---: |
| Всего файлов | 91,678 | 100.0% |
| Исключено: не легковые (`non_car`) | 50 | 0.05% |
| Исключено: размытые/сводные метки (`ambiguous_label`) | 135 | 0.15% |
| Битые файлы (`is_corrupt`) | 0 | 0.00% |
| Слишком мелкие (<80 px, `is_tiny`) | 6 | 0.01% |
| Дубликаты внутри класса (`is_duplicate`) | 1,806 | 1.97% |
| Коллизии dHash между разными классами (`is_cross_class_dup`) | 146 | 0.16% |
| **Итого чистых уникальных фото (`is_clean`)** | **89,535** | **97.66%** |

## 3. Проверка корректности годов (`year_valid`)

- Чистых фото с подозрительным/ошибочным годом (`year_valid=False`): **88** (остаются для обучения/теста марки и модели, но исключаются из регрессии года).
- Примеры папок с невалидным годом:
  - `porsche_911_1900` → `porsche__911` (год `1900`)
  - `toyota_mr2_1900` → `toyota__mr2` (год `1900`)
  - `toyota_supra_1900` → `toyota__supra` (год `1900`)
  - `volkswagen_golf_1900` → `volkswagen__golf` (год `1900`)
  - `volkswagen_beetle_1908` → `volkswagen__beetle` (год `1908`)
  - `plymouth_deluxe_1935` → `plymouth__deluxe` (год `1935`)
  - `packard_super8_1938` → `packard__super8` (год `1938`)
  - `toyota_camry_1938` → `toyota__camry` (год `1938`)
  - `studebaker_commander_1939` → `studebaker__commander` (год `1939`)
  - `oldsmobile_ninetyeight_1940` → `oldsmobile__ninetyeight` (год `1940`)
  - `willys_cj2a_1946` → `willys__cj2a` (год `1946`)
  - `mercedes benz_slk280_1974` → `mercedes-benz__slk-class` (год `1974`)
  - `mercedes benz_slk300_1984` → `mercedes-benz__slk-class` (год `1984`)
  - `mercedes benz_c300_1988` → `mercedes-benz__c-class` (год `1988`)
  - `mercedes benz_slk300_1988` → `mercedes-benz__slk-class` (год `1988`)

## 4. Разбивка на сплиты (без утечки по dHash, порог `min_class_images=15`)

| Сплит | Фото | Доля от всех |
| :--- | ---: | ---: |
| `train` | 71,072 | 77.52% |
| `val` | 8,885 | 9.69% |
| `test` | 8,885 | 9.69% |
| `few_shot` | 693 | 0.76% |
| `excluded` | 2,143 | 2.34% |

## 5. Топ-20 канонических классов по числу чистых фото

| Класс | Всего | Чистых | Train | Val | Test | Годы (валидные) | Объединённые raw-модели |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `toyota__camry` | 5,814 | 5,663 | 4,531 | 566 | 566 | 1984–2016 | `camry;camry_le;camry_se;camry_xle` |
| `nissan__altima` | 5,055 | 4,963 | 3,971 | 496 | 496 | 1993–2016 | `altima` |
| `toyota__corolla` | 4,058 | 3,975 | 3,179 | 398 | 398 | 1974–2015 | `corolla;corolla_ce;corolla_dx;corolla_le;corolla_s;corolla_wagon` |
| `volkswagen__jetta` | 3,571 | 3,517 | 2,813 | 352 | 352 | 1984–2016 | `jetta;jetta_wagon` |
| `nissan__maxima` | 3,177 | 3,104 | 2,484 | 310 | 310 | 1986–2016 | `maxima` |
| `nissan__sentra` | 2,358 | 2,295 | 1,835 | 230 | 230 | 1984–2016 | `sentra` |
| `volkswagen__passat` | 2,305 | 2,262 | 1,810 | 226 | 226 | 1992–2016 | `passat;passat_wagon` |
| `toyota__4runner` | 1,881 | 1,856 | 1,484 | 186 | 186 | 1984–2016 | `4runner` |
| `toyota__sienna` | 1,863 | 1,816 | 1,452 | 182 | 182 | 1997–2015 | `sienna` |
| `toyota__tacoma` | 1,581 | 1,553 | 1,243 | 155 | 155 | 1995–2016 | `tacoma` |
| `volkswagen__beetle` | 1,540 | 1,489 | 1,191 | 149 | 149 | 1957–2015 | `beetle;bug;newbeetle;superbeetle` |
| `mitsubishi__eclipse` | 1,525 | 1,480 | 1,184 | 148 | 148 | 1985–2013 | `eclipse;eclipse_convertible` |
| `mercedes-benz__c-class` | 1,374 | 1,340 | 1,072 | 134 | 134 | 1994–2015 | `c220;c230;c240;c250;c280;c300;c32;c320;c350;c55` |
| `nissan__pathfinder` | 1,370 | 1,332 | 1,066 | 133 | 133 | 1987–2015 | `pathfinder` |
| `nissan__quest` | 1,325 | 1,304 | 1,044 | 130 | 130 | 1993–2013 | `quest` |
| `subaru__outback` | 1,302 | 1,270 | 1,016 | 127 | 127 | 1990–2015 | `outback` |
| `mercedes-benz__e-class` | 1,251 | 1,229 | 983 | 123 | 123 | 1981–2016 | `cd300;ce300;e300;e320;e320_wagon;e350;e350_wagon;e420;e430;e500;e55;e550;e63;td300` |
| `toyota__prius` | 1,250 | 1,214 | 972 | 121 | 121 | 2001–2016 | `prius` |
| `pontiac__grandprix` | 1,182 | 1,166 | 932 | 117 | 117 | 1962–2009 | `grandprix` |
| `mini__cooper` | 1,149 | 1,104 | 884 | 110 | 110 | 2001–2015 | `cooper` |

## 6. Выборочная проверка экстерьера через YOLOv8n

Проверено **500** случайных чистых изображений (`classes=[car, bus, truck]`, `conf=0.25`):

| Категория YOLO | Кол-во | Доля |
| :--- | ---: | ---: |
| Крупный экстерьер (bbox > 8% площади, `clean_exterior`) | 496 | 99.2% |
| Мелкое авто на кадре (bbox ≤ 8% площади, `small_vehicle`) | 0 | 0.0% |
| Авто не найдено (салон / детали / мусор, `no_vehicle`) | 4 | 0.8% |
| Несколько крупных авто в кадре (≥2 боксов > 5% площади) | 143 | 28.6% |

Примеры `no_vehicle` кадров (для визуальной проверки):
- `nissan_armada_2005/2005 nissan armada_00S0S_9Hn7y6w8xro_600x450.jpg`
- `mercedes benz_c300_2009/2009 Mercedes Benz C300_00S0S_g1KgGTSzZz2_600x450.jpg`
- `pontiac_firebird_2002/2002 pontiac firebird_01111_g3SJqEvPh4b_600x450.jpg`
- `nissan_armada_2005/2005 nissan armada_01313_kymME2ZNuJ8_600x450.jpg`

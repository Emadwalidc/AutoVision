"""
AutoVision - SOTA Feature Extractor (YOLO Auto-Crop + OpenCLIP ViT-B/16)
Заменяет слабые базовые признаки в train_features.csv на 512-мерные семантические эмбеддинги.
"""

import os
import csv
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from ultralytics import YOLO


def get_vehicle_crop(img: Image.Image, yolo_model, conf_thresh=0.25):
    """Ищет автомобиль на фото через YOLO и возвращает кроп кузова."""
    w, h = img.size
    results = yolo_model.predict(
        img, classes=[2, 5, 7], conf=conf_thresh, verbose=False
    )

    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Выбираем самый крупный бокс (главный автомобиль в кадре)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_box = boxes[np.argmax(areas)]
        x1, y1, x2, y2 = best_box

        # Добавляем 5% отступа, чтобы не обрезать края кузова
        bw, bh = x2 - x1, y2 - y1
        if (bw * bh) > 0.08 * (w * h):  # Защита от мелких машин на заднем плане
            x1 = max(0, int(x1 - 0.05 * bw))
            y1 = max(0, int(y1 - 0.05 * bh))
            x2 = min(w, int(x2 + 0.05 * bw))
            y2 = min(h, int(y2 + 0.05 * bh))
            return img.crop((x1, y1, x2, y2))

    # Fallback: мягкий центральный кроп 85% (убирает края кадра с землёй и небом)
    margin_w, margin_h = int(w * 0.075), int(h * 0.075)
    return img.crop((margin_w, margin_h, w - margin_w, h - margin_h))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Устройство вычислений: {device.upper()}")

    # 1. Загружаем список файлов
    rows = []
    with open("train_data.csv", "r", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append((r[0].strip(), r[1].strip()))
    print(f"[1/4] Найдено записей в train_data.csv: {len(rows)}")

    # 2. Загружаем YOLOv8n для очистки фона и OpenCLIP ViT-B/16 (выход ровно 512-D)
    print("[2/4] Загрузка моделей YOLOv8n и OpenCLIP (ViT-B-16)...")
    yolo_model = YOLO("yolov8n.pt")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
    )
    clip_model.eval()

    # 3. Извлекаем признаки с Test-Time Augmentation (Оригинал + Зеркальный флип)
    print("[3/4] Извлечение 512-мерных признаков из train_images/...")
    embeddings = np.zeros((len(rows), 512), dtype=np.float32)

    with torch.no_grad():
        for idx, (label, filename) in enumerate(tqdm(rows, desc="Extracting")):
            img_path = os.path.join("train_images", label, filename)
            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                cropped = get_vehicle_crop(img, yolo_model)
                flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)

                tensor_orig = preprocess(cropped).unsqueeze(0).to(device)
                tensor_flip = preprocess(flipped).unsqueeze(0).to(device)
                batch = torch.cat([tensor_orig, tensor_flip], dim=0)

                feats = clip_model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                # Усредняем оригинал и флип для инвариантности к ракурсу
                mean_feat = feats.mean(dim=0)
                mean_feat = mean_feat / mean_feat.norm()
                embeddings[idx] = mean_feat.cpu().numpy()
            except Exception as e:
                print(f"[WARN] Ошибка на файле {img_path}: {e}")

    # 4. Бэкап старых признаков и сохранение новых
    if os.path.exists("train_features.csv") and not os.path.exists(
        "train_features_legacy.csv"
    ):
        shutil.copy("train_features.csv", "train_features_legacy.csv")
        print("[4/4] Старые признаки сохранены в 'train_features_legacy.csv'")

    np.savetxt("train_features.csv", embeddings, delimiter=",", fmt="%.6f")
    print("[SUCCESS] Новые 512-D признаки сохранены в 'train_features.csv'!")


if __name__ == "__main__":
    main()

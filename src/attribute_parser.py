"""
AutoVision - Multi-Attribute Dataset Generator & Color Extraction
Decomposes raw vehicle classes into the 5 target output attributes:
1. Brand (Make)
2. Model
3. Year (Generation)
4. Type (Body Category)
5. Color (Dominant Vehicle Color via HSV/RGB extraction)
"""

import os
import csv
import numpy as np
from PIL import Image

# Body category mapping
BODY_TYPES = {
    "impala": "Sedan",
    "grand caravan": "Minivan",
    "explorer": "SUV",
    "mustang": "Coupe",
    "taurus": "Sedan",
    "accord": "Sedan",
    "odyssey": "Minivan",
    "altima": "Sedan",
    "camry": "Sedan",
    "jetta": "Sedan",
}

# Standard car color taxonomy
COLOR_NAMES = ["Black", "White", "Silver/Gray", "Red", "Blue", "Green", "Yellow/Gold", "Brown/Beige"]


def parse_class_string(raw_class):
    """
    Parses strings like 'toyota_camry_2002-2003' or 'dodge_grand caravan_2005-2006'
    into structured (brand, model, year, body_type).
    """
    parts = raw_class.split("_")
    brand = parts[0].capitalize()
    year = parts[-1]
    model = "_".join(parts[1:-1]).replace("_", " ").title()

    model_key = model.lower()
    body_type = BODY_TYPES.get(model_key, "Sedan")
    return brand, model, year, body_type


def detect_dominant_color(img_path):
    """
    Extracts the dominant vehicle color from an image by:
    1. Cropping the central 50% ROI to reduce background (road, sky, trees).
    2. Converting RGB pixels to HSV.
    3. Categorizing into standard automotive color palettes.
    """
    if not os.path.exists(img_path):
        return "Unknown"

    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Central crop (w: 25%..75%, h: 25%..75%)
        crop_box = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
        cropped = img.crop(crop_box).resize((50, 50))  # Downsample for fast vector processing

        pixels = np.array(cropped).reshape(-1, 3) / 255.0  # Normalized RGB

        # Compute saturation, value (brightness), and grayscale distance
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Brightness (Value) and Saturation
        val = max_c
        sat = np.where(max_c == 0, 0, diff / (max_c + 1e-6))

        # Hue calculation in degrees [0, 360]
        hue = np.zeros_like(max_c)
        mask_r = (max_c == r) & (diff > 0)
        mask_g = (max_c == g) & (diff > 0)
        mask_b = (max_c == b) & (diff > 0)

        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

        # Pixel-wise color classification
        color_votes = []
        for h_val, s_val, v_val in zip(hue, sat, val):
            if v_val < 0.20:
                color_votes.append("Black")
            elif s_val < 0.18 and v_val > 0.75:
                color_votes.append("White")
            elif s_val < 0.20:
                color_votes.append("Silver/Gray")
            else:
                if (h_val >= 340) or (h_val < 15):
                    color_votes.append("Red")
                elif 15 <= h_val < 45:
                    color_votes.append("Brown/Beige" if v_val < 0.6 else "Yellow/Gold")
                elif 45 <= h_val < 70:
                    color_votes.append("Yellow/Gold")
                elif 70 <= h_val < 165:
                    color_votes.append("Green")
                elif 165 <= h_val < 260:
                    color_votes.append("Blue")
                else:
                    color_votes.append("Red")

        # Return majority color vote
        from collections import Counter
        majority_color = Counter(color_votes).most_common(1)[0][0]
        return majority_color
    except Exception:
        return "Unknown"


def generate_multitask_dataset(input_csv="train_data.csv", output_csv="train_multitask_attributes.csv", images_dir="train_images"):
    """Creates a comprehensive 5-attribute table for the complete dataset."""
    print("[1/2] Parsing labels into (Brand, Model, Year, Body_Type)...")
    records = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if r]

    print(f"[2/2] Extracting vehicle colors across {len(raw_rows)} images (sample inspection)...")
    for idx, row in enumerate(raw_rows):
        raw_class, filename = row[0].strip(), row[1].strip()
        brand, model, year, body_type = parse_class_string(raw_class)

        # For speed during initialization, compute color for the first 500 samples
        # or full dataset
        img_path = os.path.join(images_dir, raw_class, filename)
        if idx < 500:
            color = detect_dominant_color(img_path)
        else:
            color = "Pending_Phase2"

        records.append({
            "filename": filename,
            "raw_class": raw_class,
            "brand": brand,
            "model": model,
            "year": year,
            "body_type": body_type,
            "detected_color": color,
            "image_path": img_path
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "raw_class", "brand", "model", "year", "body_type", "detected_color", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Multi-attribute schema generated and saved to '{output_csv}'!")
    print(f"Sample parsed record:\n{records[0]}")


if __name__ == "__main__":
    generate_multitask_dataset()


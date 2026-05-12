import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATASET_PATH = Path("../data/labels/grading_labels.csv")
OUTPUT_PATH = Path("../data/processed/features.csv")


def whitening_ratio(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 75, 255]))
    return float(np.count_nonzero(mask)) / float(mask.size)


def extract_image_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape[:2]

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)

    border = max(4, int(min(w, h) * 0.035))

    top = whitening_ratio(image[:border, :])
    bottom = whitening_ratio(image[h - border :, :])
    left = whitening_ratio(image[:, :border])
    right = whitening_ratio(image[:, w - border :])

    avg_edge_whitening = np.mean([top, bottom, left, right])

    return {
        "width": w,
        "height": h,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "avg_edge_whitening": avg_edge_whitening,
    }


def main():
    dataset = pd.read_csv(DATASET_PATH)

    rows = []

    for _, row in dataset.iterrows():
        features = extract_image_features(row["front_scan"])

        rows.append({
            "card_id": row["card_id"],
            "psa_grade": row["psa_grade"],
            **features,
        })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

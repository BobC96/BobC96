from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pokemon Card AI Grader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class CardAnalysis:
    card_found: bool
    width: int
    height: int
    centering_score: float
    edges_score: float
    corners_score: float
    surface_score: float
    overall_score: float
    diagnostics: dict[str, Any]


def clamp(value: float, minimum: float = 1.0, maximum: float = 10.0) -> float:
    return max(minimum, min(maximum, value))


def round_score(value: float) -> float:
    return round(clamp(value), 2)


def decode_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image")
    return image


def find_card_contour(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / max(w, 1)
        if 1.25 <= aspect_ratio <= 1.60:
            candidates.append((area, contour, (x, y, w, h)))

    if not candidates:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return largest, (x, y, w, h)

    _, contour, box = max(candidates, key=lambda item: item[0])
    return contour, box


def crop_card(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return image[y : y + h, x : x + w]


def estimate_centering(card: np.ndarray) -> tuple[float, dict[str, Any]]:
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use the inner card art transition as an approximate border boundary.
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )

    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    search = threshold[margin_y : h - margin_y, margin_x : w - margin_x]

    contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 7.0, {"method": "fallback", "reason": "inner border not detected"}

    largest = max(contours, key=cv2.contourArea)
    x, y, inner_w, inner_h = cv2.boundingRect(largest)
    x += margin_x
    y += margin_y

    left = x
    right = w - (x + inner_w)
    top = y
    bottom = h - (y + inner_h)

    horizontal_ratio = min(left, right) / max(left, right, 1)
    vertical_ratio = min(top, bottom) / max(top, bottom, 1)
    ratio_score = ((horizontal_ratio + vertical_ratio) / 2) * 10

    score = clamp(5.0 + ratio_score / 2)
    return round_score(score), {
        "method": "inner-border-estimate",
        "left_border_px": int(left),
        "right_border_px": int(right),
        "top_border_px": int(top),
        "bottom_border_px": int(bottom),
        "horizontal_balance": round(horizontal_ratio, 3),
        "vertical_balance": round(vertical_ratio, 3),
    }


def whitening_ratio(region: np.ndarray) -> float:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # White/very light low-saturation pixels are treated as possible whitening.
    mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 75, 255]))
    return float(np.count_nonzero(mask)) / float(mask.size)


def estimate_edges(card: np.ndarray) -> tuple[float, dict[str, Any]]:
    h, w = card.shape[:2]
    border = max(4, int(min(w, h) * 0.035))

    regions = {
        "top": card[:border, :],
        "bottom": card[h - border :, :],
        "left": card[:, :border],
        "right": card[:, w - border :],
    }
    ratios = {name: whitening_ratio(region) for name, region in regions.items()}
    avg_whitening = float(np.mean(list(ratios.values())))
    score = 10 - (avg_whitening * 18)

    return round_score(score), {
        "border_sample_px": border,
        "average_possible_whitening": round(avg_whitening, 4),
        "side_ratios": {key: round(value, 4) for key, value in ratios.items()},
    }


def estimate_corners(card: np.ndarray) -> tuple[float, dict[str, Any]]:
    h, w = card.shape[:2]
    size = max(12, int(min(w, h) * 0.10))

    regions = {
        "top_left": card[:size, :size],
        "top_right": card[:size, w - size :],
        "bottom_left": card[h - size :, :size],
        "bottom_right": card[h - size :, w - size :],
    }
    ratios = {name: whitening_ratio(region) for name, region in regions.items()}
    avg_whitening = float(np.mean(list(ratios.values())))
    worst_corner = max(ratios.values())
    score = 10 - (avg_whitening * 10) - (worst_corner * 6)

    return round_score(score), {
        "corner_sample_px": size,
        "average_possible_whitening": round(avg_whitening, 4),
        "worst_corner_possible_whitening": round(worst_corner, 4),
        "corner_ratios": {key: round(value, 4) for key, value in ratios.items()},
    }


def estimate_surface(card: np.ndarray) -> tuple[float, dict[str, Any]]:
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    inset_x = int(w * 0.08)
    inset_y = int(h * 0.08)
    inner = gray[inset_y : h - inset_y, inset_x : w - inset_x]

    laplacian = cv2.Laplacian(inner, cv2.CV_64F)
    texture_noise = float(np.std(laplacian))
    score = 10 - max(0, (texture_noise - 18) / 12)

    return round_score(score), {
        "texture_noise": round(texture_noise, 2),
        "note": "Surface scoring is a rough placeholder until angled-light scans are supported.",
    }


def analyze_single_image(image: np.ndarray) -> CardAnalysis:
    contour, box = find_card_contour(image)
    if box is None:
        return CardAnalysis(
            card_found=False,
            width=image.shape[1],
            height=image.shape[0],
            centering_score=1,
            edges_score=1,
            corners_score=1,
            surface_score=1,
            overall_score=1,
            diagnostics={"error": "No card-like contour found"},
        )

    card = crop_card(image, box)
    centering_score, centering_details = estimate_centering(card)
    edges_score, edge_details = estimate_edges(card)
    corners_score, corner_details = estimate_corners(card)
    surface_score, surface_details = estimate_surface(card)

    overall = (
        centering_score * 0.35
        + corners_score * 0.25
        + edges_score * 0.25
        + surface_score * 0.15
    )

    return CardAnalysis(
        card_found=True,
        width=card.shape[1],
        height=card.shape[0],
        centering_score=centering_score,
        edges_score=edges_score,
        corners_score=corners_score,
        surface_score=surface_score,
        overall_score=round_score(overall),
        diagnostics={
            "card_box": {"x": int(box[0]), "y": int(box[1]), "width": int(box[2]), "height": int(box[3])},
            "centering": centering_details,
            "edges": edge_details,
            "corners": corner_details,
            "surface": surface_details,
        },
    )


def to_response(analysis: CardAnalysis) -> dict[str, Any]:
    return {
        "card_found": analysis.card_found,
        "detected_card_size": {
            "width_px": analysis.width,
            "height_px": analysis.height,
        },
        "scores": {
            "centering": analysis.centering_score,
            "corners": analysis.corners_score,
            "edges": analysis.edges_score,
            "surface": analysis.surface_score,
            "overall": analysis.overall_score,
        },
        "diagnostics": analysis.diagnostics,
    }


@app.get("/")
def root():
    return {"message": "Pokemon Card AI Grader API Running"}


@app.post("/grade")
async def grade_card(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
):
    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    front = analyze_single_image(decode_image(front_bytes))
    back = analyze_single_image(decode_image(back_bytes))

    combined_score = round_score((front.overall_score * 0.6) + (back.overall_score * 0.4))
    estimated_grade = int(round(combined_score))

    confidence = 0.55
    if front.card_found and back.card_found:
        confidence += 0.2
    if front.width >= 1500 and back.width >= 1500:
        confidence += 0.1
    if combined_score >= 9:
        confidence -= 0.05

    return {
        "estimated_grade": estimated_grade,
        "combined_score": combined_score,
        "confidence": round(min(confidence, 0.9), 2),
        "recommendation": "Consider PSA submission" if combined_score >= 8.5 else "Probably not worth PSA submission yet",
        "front": to_response(front),
        "back": to_response(back),
        "disclaimer": "This is an AI pre-grading estimate, not an official PSA grade.",
    }

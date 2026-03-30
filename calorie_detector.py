import base64
import imghdr
import json
import os
from dataclasses import dataclass

import requests


DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_FOOD_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short food name like grilled chicken, rice, fries, salad, or coffee.",
                    },
                    "calories": {
                        "type": "integer",
                        "description": "Estimated calories for the visible portion of this one item.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Estimated confidence from 0.0 to 1.0 for the food identification and calorie estimate.",
                    },
                },
                "required": ["name", "calories", "confidence"],
            },
        },
        "total_calories": {
            "type": "integer",
            "description": "Sum of all item calorie estimates.",
        },
    },
    "required": ["items", "total_calories"],
}

_PROMPT = (
    "You are estimating food calories from a single meal photo for a calorie tracking app. "
    "Return only foods that are clearly visible. Ignore plates, utensils, tables, packaging, and background objects. "
    "Estimate portion-aware calories for each visible food item. "
    "If the image is not food, return an empty items array and total_calories 0. "
    "Use short lowercase-friendly names. "
    "Confidence must be a decimal from 0.0 to 1.0 that reflects your own certainty in both item identification and calorie estimate."
)


class FoodAnalysisError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class NormalizedFoodItem:
    name: str
    calories: int
    confidence: float


def _detect_mime_type(image_bytes: bytes) -> str:
    image_type = imghdr.what(None, image_bytes)
    if image_type == "jpeg":
        return "image/jpeg"
    if image_type == "png":
        return "image/png"
    if image_type == "gif":
        return "image/gif"
    if image_type == "webp":
        return "image/webp"
    raise FoodAnalysisError("Unsupported image format. Use JPG, PNG, WEBP, or GIF.")


def _build_request_payload(image_bytes: bytes, mime_type: str) -> dict:
    return {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                    {"text": _PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": _FOOD_SCHEMA,
            "temperature": 0.2,
        },
    }


def _gemini_endpoint(model: str) -> str:
    return GEMINI_API_URL.format(model=model)


def _request_gemini(image_bytes: bytes, mime_type: str) -> dict:
    if not GEMINI_API_KEY:
        raise FoodAnalysisError("Missing GEMINI_API_KEY environment variable.", status_code=500)

    response = requests.post(
        _gemini_endpoint(DEFAULT_GEMINI_MODEL),
        params={"key": GEMINI_API_KEY},
        json=_build_request_payload(image_bytes, mime_type),
        timeout=60,
    )
    if not response.ok:
        detail = ""
        try:
            payload = response.json()
            detail = payload.get("error", {}).get("message", "")
        except ValueError:
            detail = response.text[:300]
        raise FoodAnalysisError(
            f"Gemini request failed ({response.status_code}): {detail or 'unknown error'}",
            status_code=502,
        )

    try:
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise FoodAnalysisError("Gemini returned an invalid response payload.", status_code=502) from exc


def _normalize_confidence(raw_confidence) -> float:
    try:
        value = float(raw_confidence)
    except (TypeError, ValueError):
        value = 0.7
    return round(min(1.0, max(0.0, value)), 3)


def _normalize_item(raw_item: dict) -> NormalizedFoodItem | None:
    if not isinstance(raw_item, dict):
        return None

    name = str(raw_item.get("name", "")).strip()
    if not name:
        return None

    try:
        calories = int(round(float(raw_item.get("calories", 0))))
    except (TypeError, ValueError):
        calories = 0

    return NormalizedFoodItem(
        name=name,
        calories=max(0, calories),
        confidence=_normalize_confidence(raw_item.get("confidence")),
    )


def analyze_food_image(image_bytes: bytes) -> dict:
    if not image_bytes:
        raise FoodAnalysisError("Image file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise FoodAnalysisError("Image file is too large. Keep it under 20MB.")

    mime_type = _detect_mime_type(image_bytes)
    raw_response = _request_gemini(image_bytes, mime_type)

    items = []
    for raw_item in raw_response.get("items", []):
        item = _normalize_item(raw_item)
        if item is None:
            continue
        items.append(
            {
                "name": item.name,
                "calories": item.calories,
                "confidence": item.confidence,
            }
        )

    total_calories = sum(item["calories"] for item in items)
    return {
        "items": items,
        "total_calories": total_calories,
    }

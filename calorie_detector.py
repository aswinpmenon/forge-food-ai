from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO


CONFIDENCE_THRESHOLD = 0.5

FOOD_CALORIES = {
    "apple": 95,
    "banana": 105,
    "orange": 62,
    "pizza": 285,
    "sandwich": 250,
    "hot dog": 290,
    "donut": 195,
    "cake": 260,
    "broccoli": 55,
    "carrot": 30,
}

MODEL_TO_FOOD = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "pizza": "pizza",
    "sandwich": "sandwich",
    "hot dog": "hot dog",
    "donut": "donut",
    "cake": "cake",
    "broccoli": "broccoli",
    "carrot": "carrot",
}


@lru_cache(maxsize=1)
def get_model():
    return YOLO("yolov8n.pt")


def portion_multiplier(box_area_ratio: float) -> float:
    reference_ratio = 0.12
    multiplier = box_area_ratio / reference_ratio
    return max(0.5, min(1.8, multiplier))


def analyze_food_image(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_area = image.width * image.height

    model = get_model()
    result = model.predict(source=image_np, verbose=False)[0]

    items = []
    total = 0

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        model_name = result.names[class_id]
        mapped_name = MODEL_TO_FOOD.get(model_name)
        if not mapped_name:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        multiplier = portion_multiplier(box_area / image_area)
        calories = round(FOOD_CALORIES[mapped_name] * multiplier)

        item = {
            "name": mapped_name,
            "calories": calories,
            "confidence": round(confidence, 3),
        }
        items.append(item)
        total += calories

    return {
        "items": items,
        "total_calories": total,
    }

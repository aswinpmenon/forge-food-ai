import json
import os
import requests

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

class CoachAnalysisError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code

def _build_request_payload(data: dict) -> dict:
    prompt = (
        "You are an elite AI personal trainer for a health tracking app called FORGE. "
        "You will receive a user's calorie goal and their last 7 days of calorie and workout data. "
        "Analyze the data for consistency, calorie adherence (surplus/deficit relative to goal), workout frequency, and recovery gaps. "
        "Provide exactly 1-2 concise, direct, actionable, and encouraging sentences of daily feedback. "
        "Do not include greetings or pleasantries. Be specific and data-driven where possible.\n\n"
    )
    
    prompt += f"Goal: {data.get('goal', 2400)} kcal\n\nPast 7 days data:\n"
    for day in data.get('days', []):
        workouts = ", ".join(day.get('workouts', [])) or "Rest"
        prompt += f"{day.get('date', 'Unknown')}: {day.get('calories', 0)} kcal, Workouts: {workouts}\n"

    return {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
        },
    }

def analyze_user_data(data: dict) -> dict:
    if not GEMINI_API_KEY:
        raise CoachAnalysisError("Missing GEMINI_API_KEY environment variable.", status_code=500)

    payload = _build_request_payload(data)
    url = GEMINI_API_URL.format(model=DEFAULT_GEMINI_MODEL)
    
    response = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=30)
    
    if not response.ok:
        detail = ""
        try:
            res_json = response.json()
            detail = res_json.get("error", {}).get("message", "")
        except ValueError:
            detail = response.text[:300]
        raise CoachAnalysisError(
            f"Gemini request failed ({response.status_code}): {detail or 'unknown error'}",
            status_code=502
        )

    try:
        res_json = response.json()
        feedback = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"feedback": feedback}
    except (KeyError, IndexError, TypeError) as exc:
        raise CoachAnalysisError("Gemini returned an invalid response payload.", status_code=502) from exc

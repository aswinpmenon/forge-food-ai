import os
import re
from datetime import date

import requests


SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
FORGE_SHORTCUT_SECRET = os.environ.get("FORGE_SHORTCUT_SECRET", "").strip()


class HealthSyncError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _extract_calories(value) -> int:
    if isinstance(value, (int, float)):
        return max(0, int(round(float(value))))

    source = str(value or "").replace(",", " ")
    kcal_match = re.search(r"(\d{1,5})(?:\s*(?:kcal|calories?|cals?))\b", source, re.IGNORECASE)
    if kcal_match:
        return max(0, int(kcal_match.group(1)))

    plain_match = re.search(r"\b(\d{2,5})\b", source)
    if plain_match:
        return max(0, int(plain_match.group(1)))

    return 0


def _require_env() -> None:
    if not FORGE_SHORTCUT_SECRET:
        raise HealthSyncError("Missing FORGE_SHORTCUT_SECRET environment variable.", status_code=500)
    if not SUPABASE_URL:
        raise HealthSyncError("Missing SUPABASE_URL environment variable.", status_code=500)
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise HealthSyncError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable.", status_code=500)


def _validate_secret(secret: str) -> None:
    if not secret or secret != FORGE_SHORTCUT_SECRET:
        raise HealthSyncError("Invalid shortcut secret.", status_code=401)


def _normalize_payload(payload: dict, header_secret: str = "") -> dict:
    if not isinstance(payload, dict):
        raise HealthSyncError("Invalid JSON body.", status_code=400)

    secret = str(payload.get("secret") or header_secret or "").strip()
    _validate_secret(secret)

    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        raise HealthSyncError("Missing user_id.", status_code=400)

    sync_date = str(payload.get("date") or date.today().isoformat()).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", sync_date):
        raise HealthSyncError("date must be YYYY-MM-DD.", status_code=400)

    calories = _extract_calories(
        payload.get("burned_calories")
        or payload.get("calories")
        or payload.get("text")
        or payload.get("result")
    )
    if calories <= 0:
        raise HealthSyncError("Missing or invalid burned_calories value.", status_code=400)

    source = str(payload.get("source") or "apple_shortcuts").strip() or "apple_shortcuts"
    return {
        "user_id": user_id,
        "date": sync_date,
        "burned_calories": calories,
        "source": source,
    }


def _upsert_supabase_setting(user_id: str, setting: str, value: str) -> None:
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/app_config?on_conflict=owner_id,setting",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=representation",
        },
        json=[{"owner_id": user_id, "setting": setting, "value": value}],
        timeout=30,
    )
    if response.ok:
        return

    detail = ""
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:300]
    raise HealthSyncError(f"Supabase sync failed: {detail or response.status_code}", status_code=502)


def sync_burned_calories(payload: dict, header_secret: str = "") -> dict:
    _require_env()
    normalized = _normalize_payload(payload, header_secret=header_secret)
    setting = f"Burned_Calories_{normalized['date']}"
    _upsert_supabase_setting(normalized["user_id"], setting, str(normalized["burned_calories"]))
    return {
        "status": "ok",
        "user_id": normalized["user_id"],
        "date": normalized["date"],
        "burned_calories": normalized["burned_calories"],
        "setting": setting,
        "source": normalized["source"],
    }

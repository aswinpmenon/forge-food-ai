import os

from flask import Flask, jsonify, request

from calorie_detector import FoodAnalysisError, analyze_food_image
from coach_agent import analyze_user_data, CoachAnalysisError


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/food/analyze", methods=["POST", "OPTIONS"])
def analyze_food():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Missing image file"}), 400
    if not file.filename:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        payload = analyze_food_image(file.read())
        return jsonify(payload)
    except FoodAnalysisError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:  # pragma: no cover - runtime dependency errors surface here
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health/coach", methods=["POST", "OPTIONS"])
def coach_analysis():
    if request.method == "OPTIONS":
        return ("", 204)
        
    data = request.get_json(silent=True) or {}
    try:
        payload = analyze_user_data(data)
        return jsonify(payload)
    except CoachAnalysisError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)


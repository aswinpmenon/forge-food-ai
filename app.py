import os

from flask import Flask, jsonify, request

from calorie_detector import analyze_food_image


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/api/food/analyze", methods=["POST", "OPTIONS"])
def analyze_food():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Missing image file"}), 400

    try:
        payload = analyze_food_image(file.read())
        return jsonify(payload)
    except Exception as exc:  # pragma: no cover - runtime dependency errors surface here
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)

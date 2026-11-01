"""
app.py
------
Web Application for Conformal Leaf Disease Detection & Uncertainty Quantification.
Runs a local Flask server providing a drag-and-drop interface for uploading leaf images
and viewing model scores and split-conformal candidate sets.
"""

import json
import re
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from predict import predict_with_uncertainty
from paths import RESULTS_PATH, THRESH_PATH

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['TEMPLATES_AUTO_RELOAD'] = True
THRESHOLD_PATH = THRESH_PATH


def coverage_summary() -> dict:
    """Read the calibrated target and most recently recorded test result."""
    target_coverage = None
    empirical_coverage = None
    evaluation_sample_count = None

    try:
        threshold_data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        target_coverage = float(threshold_data["target_coverage"]) * 100
    except (OSError, ValueError, KeyError, TypeError):
        pass

    try:
        results_text = RESULTS_PATH.read_text(encoding="utf-8")
    except OSError:
        results_text = ""
    match = re.search(r"Empirical Coverage\s*:\s*([\d.]+)%\s*\(\d+/(\d+)\)", results_text)
    if match:
        empirical_coverage = float(match.group(1))
        evaluation_sample_count = int(match.group(2))

    return {
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "evaluation_sample_count": evaluation_sample_count,
    }

@app.route("/")
def index():
    return render_template("index.html", **coverage_summary())


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify({"error": "Uploaded image exceeds the 16 MB limit."}), 413

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Choose an image file to upload."}), 400

    try:
        with Image.open(file.stream) as uploaded_image:
            uploaded_image.verify()
        file.stream.seek(0)
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    temp_file_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
            suffix = ".img"
        with tempfile.NamedTemporaryFile(prefix="tomato-leaf-", suffix=suffix, delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            file.save(temp_file)

        # Run conformal inference
        results = predict_with_uncertainty(temp_file_path)

        leaf_screen_passed = results.get(
            "leaf_screen_passed", results.get("is_tomato_leaf", True)
        )
        
        # Add formatted confidence percentage for frontend display
        formatted_confidences = []
        for cls_name, prob in sorted(results["confidences"].items(), key=lambda x: x[1], reverse=True):
            formatted_confidences.append({
                "class_name": cls_name.replace("Tomato_", "").replace("_", " "),
                "raw_name": cls_name,
                "percentage": round(prob * 100, 2),
                "in_set": cls_name in results["prediction_set"]
            })

        top_raw_class = results["top_class"]
        top_disease_clean = top_raw_class.replace("Tomato_", "").replace("_", " ")
        top_conf_pct = round(results["top_confidence"] * 100, 2)
        is_confident = results["is_confident"]

        return jsonify({
            "success": True,
            "is_tomato_leaf": leaf_screen_passed,  # Compatibility alias for the heuristic screen.
            "leaf_screen_passed": leaf_screen_passed,
            "leaf_screen_threshold": results.get("leaf_screen_threshold", 0.20),
            "leaf_screen_validated": results.get("leaf_screen_validated", False),
            "top_disease": top_disease_clean,
            "top_confidence_percentage": top_conf_pct,
            "is_confident": is_confident,
            "uncertainty_level": results.get("uncertainty_level", "HIGH"),
            "prediction_set_status": results.get(
                "prediction_set_status", "nonempty" if results["prediction_set"] else "empty"
            ),
            "prediction_set": [c.replace("Tomato_", "").replace("_", " ") for c in results["prediction_set"]],
            "raw_prediction_set": results["prediction_set"],
            "set_size": results["set_size"],
            "confidences": formatted_confidences,
            "target_coverage": results.get("target_coverage"),
        })

    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({"error": "Prediction failed. Check the server log for details."}), 500
    finally:
        if temp_file_path is not None:
            try:
                temp_file_path.unlink(missing_ok=True)
            except Exception:
                pass

if __name__ == "__main__":
    print("=" * 70)
    print("  Starting Conformal Leaf Disease Detection Web App")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)

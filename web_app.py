"""
PsyNLP Web Application
Mental Health Text Analysis with Flask
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import time

# Import from modules
from modules.file_reader import read_file, get_file_info
from modules.sentence_process import process_markdown, clean_text, split_sentences
from modules.predict_score import calculate_article_severity
from modules.visualizer import generate_html_report, open_in_browser

# Try to import LLM enhancements
ENHANCEMENTS_AVAILABLE = False
try:
    from modules.llm_enhancements import check_suicidal_risk

    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    print("Warning: LLM enhancements not available")

    def check_suicidal_risk(analysis_result, threshold=6):
        """Fallback function when LLM enhancements not available"""
        return False


# Flask app configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = "psynlp-secret-key-2025"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["ALLOWED_EXTENSIONS"] = {".txt", ".md", ".docx"}

# Create upload folder if not exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Global storage for analysis results (in production, use Redis or database)
analysis_cache = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify({"error": "Unsupported file format. Use .txt, .md, or .docx"}),
                400,
            )

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        # Get file info
        info = get_file_info(filepath)

        return jsonify(
            {
                "success": True,
                "filename": unique_filename,
                "original_name": filename,
                "size": info["size"],
                "extension": info["extension"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """Analyze uploaded text file"""
    try:
        data = request.json
        filename = data.get("filename")
        mode = data.get("mode", "user")

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Read and process file
        content = read_file(filepath)
        clean_content = clean_text(content)
        sentences = split_sentences(clean_content)

        # Create temporary file for analysis
        temp_file = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{filename}.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sentences))

        # Run analysis
        analysis_result = calculate_article_severity(temp_file)

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Store result in cache
        result_id = f"{filename}_{mode}_{int(time.time())}"
        analysis_cache[result_id] = {
            "result": analysis_result,
            "mode": mode,
            "filename": filename,
        }

        # Check for suicidal risk
        high_risk = False
        if mode == "user" and check_suicidal_risk(analysis_result):
            high_risk = True

        # Extract statistics
        stats = analysis_result["severity_stats"]

        return jsonify(
            {
                "success": True,
                "result_id": result_id,
                "high_risk": high_risk,
                "stats": {
                    "overall_severity_score": stats["overall_severity_score"],
                    "max_severity": stats["max_severity"],
                    "avg_severity": stats["avg_severity"],
                    "high_risk_sentences": stats["high_risk_sentences"],
                    "sentence_count": analysis_result["sentence_count"],
                },
                "nlp_analysis": analysis_result["nlp_analysis"],
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/<result_id>", methods=["GET"])
def generate_report(result_id):
    """Generate HTML report"""
    try:
        if result_id not in analysis_cache:
            return jsonify({"error": "Analysis result not found"}), 404

        cached = analysis_cache[result_id]
        analysis_result = cached["result"]
        mode = cached["mode"]
        filename = cached["filename"]

        # Generate report
        base_name = Path(filename).stem
        report_name = f"{base_name}_report_{mode}_{int(time.time())}.html"
        report_path = os.path.join("reports", report_name)

        generate_html_report(analysis_result, report_path, language="en", mode=mode)

        return jsonify(
            {
                "success": True,
                "report_url": url_for("view_report", filename=report_name),
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/report/<filename>")
def view_report(filename):
    """View generated report"""
    report_path = os.path.join("reports", filename)
    if os.path.exists(report_path):
        return send_file(report_path)
    return "Report not found", 404


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "enhancements_available": ENHANCEMENTS_AVAILABLE}
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7800)

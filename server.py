"""
Brass Arranger - Backend Server
New Orleans Street Brass Band style: Big 6, Rebirth, Hot 8, Young Pinstripes,
To Be Continued, Lil Rascals DNA baked in.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from analyzer import analyze_audio
from arranger import generate_arrangement
from notation import render_sheet_music

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    suffix = Path(f.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=UPLOAD_DIR
    )
    f.save(tmp.name)
    tmp.close()

    try:
        print(f"[server] Analyzing: {f.filename}")

        # Step 1: Audio analysis (key, BPM, chords, melody MIDI)
        analysis = analyze_audio(tmp.name)
        print(f"[server] Analysis complete: key={analysis['key']} bpm={analysis['bpm']}")

        # Step 2: Generate arrangement with NOLA street style rules
        arrangement = generate_arrangement(analysis)
        print(f"[server] Arrangement generated")

        # Step 3: Render sheet music PDFs per instrument
        pdf_paths = render_sheet_music(arrangement, OUTPUT_DIR)
        print(f"[server] Sheet music rendered: {list(pdf_paths.keys())}")

        return jsonify({
            "analysis": analysis,
            "arrangement": arrangement,
            "pdfs": {k: f"/download/{Path(v).name}" for k, v in pdf_paths.items()}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  Brass Arranger — New Orleans Street Style")
    print("  Big 6 · Rebirth · Hot 8 · Young Pinstripes")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)

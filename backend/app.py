"""Flask API for the recovered local GRU-small deployment model."""

from __future__ import annotations

import os
import random
import tempfile
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from src.data_prep.midi_representation import decode_tokens
from src.generation.run_local_model import generate, load_local_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT_ID = "gru_small_250"
ARTIFACT_DIR = PROJECT_ROOT / "local_artifacts" / MODEL_ARTIFACT_ID
SEED_FILES_DIR = PROJECT_ROOT / "data" / "seed"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PRODUCTION_FRONTEND_URL = "https://audiogroove.vercel.app"


def _load_artifacts() -> tuple[object | None, dict[str, int] | None, dict | None, str | None]:
    try:
        model, vocabulary, config = load_local_model(ARTIFACT_DIR)
        return model, vocabulary, config, None
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        return None, None, None, f"{type(error).__name__}: {error}"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    frontend_url = os.environ.get("FRONTEND_URL", PRODUCTION_FRONTEND_URL)
    allowed_origins = {
        PRODUCTION_FRONTEND_URL,
        frontend_url,
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://localhost:8000",
    }
    CORS(app, origins=sorted(allowed_origins))

    model, vocabulary, config, load_error = _load_artifacts()
    app.extensions["audiogroove"] = {
        "model": model,
        "vocabulary": vocabulary,
        "config": config,
        "load_error": load_error,
    }

    @app.get("/")
    def health_check():
        state = app.extensions["audiogroove"]
        config = state["config"]
        response = {
            "status": "ok" if state["model"] is not None else "degraded",
            "message": "AudioGroove backend is running.",
            "model_loaded": state["model"] is not None,
            "model_family": config.get("model_family") if config else None,
            "model_profile": config.get("model_profile") if config else None,
            "model_artifact": config.get("artifact_id") if config else None,
            "dataset_size": config.get("dataset_size") if config else None,
            "vocabulary_size": len(state["vocabulary"]) if state["vocabulary"] else None,
        }
        if state["load_error"]:
            response["load_error"] = state["load_error"]
        return jsonify(response), 200 if state["model"] is not None else 503

    @app.post("/generate")
    def generate_music_endpoint():
        state = app.extensions["audiogroove"]
        if state["model"] is None:
            return jsonify({"error": "Recovered model is not loaded.", "details": state["load_error"]}), 503

        temporary_seed: Path | None = None
        try:
            uploaded = request.files.get("seed_midi")
            if uploaded and uploaded.filename:
                suffix = Path(uploaded.filename).suffix.lower()
                if suffix not in {".mid", ".midi"}:
                    return jsonify({"error": "seed_midi must be a .mid or .midi file."}), 400
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as seed_handle:
                    uploaded.save(seed_handle)
                    seed_path = Path(seed_handle.name)
                temporary_seed = seed_path
            else:
                seeds = sorted(SEED_FILES_DIR.glob("*.mid")) + sorted(SEED_FILES_DIR.glob("*.midi"))
                if not seeds:
                    return jsonify({"error": "The server seed directory is empty."}), 500
                seed_path = random.choice(seeds)

            tokens, ticks_per_beat = generate(
                state["model"], state["vocabulary"], state["config"], seed_path
            )
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as output_handle:
                output_path = Path(output_handle.name)
            decode_tokens(tokens, output_path, ticks_per_beat=ticks_per_beat)
            midi_bytes = output_path.read_bytes()
            return send_file(
                BytesIO(midi_bytes),
                mimetype="audio/midi",
                as_attachment=True,
                download_name="generated_music.mid",
            )
        except (OSError, ValueError, EOFError) as error:
            return jsonify({"error": "Could not generate MIDI from the supplied seed.", "details": str(error)}), 400
        finally:
            if temporary_seed is not None:
                temporary_seed.unlink(missing_ok=True)
            if "output_path" in locals():
                output_path.unlink(missing_ok=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))

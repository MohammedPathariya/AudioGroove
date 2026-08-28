from __future__ import annotations

from io import BytesIO
from pathlib import Path

import mido
import pytest

from backend.app import app


ARTIFACT_DIR = Path(__file__).parents[1] / "local_artifacts" / "gru_small_250"
ARTIFACT_CHECKPOINT = ARTIFACT_DIR / "checkpoints" / "deploy.pt"
ARTIFACTS_AVAILABLE = ARTIFACT_CHECKPOINT.exists()


pytestmark = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason="local recovered model package is intentionally ignored by Git",
)


def test_health_reports_recovered_model():
    response = app.test_client().get("/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model_loaded"] is True
    assert payload["model_family"] == "gru"
    assert payload["model_profile"] == "small"
    assert payload["model_artifact"] == "gru_small_250"
    assert payload["dataset_size"] == 250
    assert payload["vocabulary_size"] == 18849


def test_cors_allows_production_frontend():
    response = app.test_client().get(
        "/",
        headers={"Origin": "https://audiogroove.vercel.app"},
    )

    assert response.headers["Access-Control-Allow-Origin"] == "https://audiogroove.vercel.app"


def test_generate_without_seed_returns_parseable_midi():
    response = app.test_client().post("/generate")

    assert response.status_code == 200
    midi = mido.MidiFile(file=BytesIO(response.data))
    assert len(midi.tracks) >= 1
    assert any(message.type in {"note_on", "note_off"} for track in midi.tracks for message in track)


def test_generate_with_uploaded_seed_returns_parseable_midi():
    seed = Path(__file__).parents[1] / "data" / "seed" / "Boom_Boom_Boom.mid"
    response = app.test_client().post(
        "/generate",
        data={"seed_midi": (seed.open("rb"), "seed.mid")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    midi = mido.MidiFile(file=BytesIO(response.data))
    assert len(midi.tracks) >= 1


def test_generate_rejects_non_midi_upload():
    response = app.test_client().post(
        "/generate",
        data={"seed_midi": (BytesIO(b"not midi"), "seed.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "mid" in response.get_json()["error"]

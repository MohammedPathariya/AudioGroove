import json
from pathlib import Path

import pytest
import torch


ARTIFACT_DIR = Path(__file__).parents[1] / "local_artifacts" / "gru_small_250"
DEPLOY_CHECKPOINT = ARTIFACT_DIR / "checkpoints" / "deploy.pt"
MANIFEST = ARTIFACT_DIR / "deployment_manifest.json"


pytestmark = pytest.mark.skipif(
    not DEPLOY_CHECKPOINT.exists() or not MANIFEST.exists(),
    reason="local deployment artifact package is intentionally ignored by Git",
)


def test_deployment_checkpoint_contains_inference_state_only() -> None:
    checkpoint = torch.load(DEPLOY_CHECKPOINT, map_location="cpu")

    assert set(checkpoint) == {"model", "dataset_revision", "vocabulary_hash"}


def test_deployment_manifest_matches_gru_small_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "gru_small_250"
    assert manifest["dataset"]["song_count"] == 250
    assert manifest["model"]["family"] == "gru"
    assert manifest["model"]["profile"] == "small"
    assert manifest["model"]["parameter_count"] == 6_236_001
    assert manifest["vocabulary"]["size"] == 18_849

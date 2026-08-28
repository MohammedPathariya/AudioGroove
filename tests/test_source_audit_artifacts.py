from pathlib import Path


AUDIT_DIR = Path(__file__).parents[1] / "data" / "audit" / "source_audit"


def test_source_audit_artifacts_use_the_current_directory_name() -> None:
    for artifact in AUDIT_DIR.glob("*.json*"):
        contents = artifact.read_text(encoding="utf-8")

        assert "data/audit/day2" not in contents


def test_chunk_manifests_reference_the_source_audit_directory() -> None:
    for filename in ("smoke_chunks.json", "development_chunks.json"):
        contents = (AUDIT_DIR / filename).read_text(encoding="utf-8")

        assert "data/audit/source_audit/chunks/" in contents

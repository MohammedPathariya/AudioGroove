"""Load a recovered local GRU checkpoint and generate MIDI locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data_prep.midi_representation import decode_tokens, encode_midi
from src.models.compact_midi_models import build_compact_model, count_parameters


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "local_artifacts" / "gru_small_250"
DEFAULT_SEED = ROOT / "data" / "seed" / "Boom_Boom_Boom.mid"


def load_local_model(artifact_dir: Path) -> tuple[torch.nn.Module, dict[str, int], dict]:
    config = json.loads((artifact_dir / "config" / "experiment_config.json").read_text())
    manifest = json.loads((artifact_dir / "deployment_manifest.json").read_text())
    vocabulary_path = artifact_dir / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())
    vocabulary_hash = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    if manifest["artifact_schema_version"] != 1:
        raise ValueError("unsupported deployment manifest schema")
    if vocabulary_hash != manifest["vocabulary"]["sha256"]:
        raise ValueError("local vocabulary hash does not match the deployment manifest")
    if len(vocabulary) != manifest["vocabulary"]["size"]:
        raise ValueError("local vocabulary size does not match the deployment manifest")
    if config["dataset_revision"] != manifest["dataset"]["revision"]:
        raise ValueError("experiment configuration dataset revision does not match the deployment manifest")
    if config["model_family"] != manifest["model"]["family"]:
        raise ValueError("experiment configuration model family does not match the deployment manifest")
    if config["model_profile"] != manifest["model"]["profile"]:
        raise ValueError("experiment configuration model profile does not match the deployment manifest")

    model = build_compact_model(config["model_family"], len(vocabulary), config["model_parameters"])
    if count_parameters(model) != manifest["model"]["parameter_count"]:
        raise ValueError("model parameter count does not match the deployment manifest")

    checkpoint_path = artifact_dir / manifest["checkpoint"]["path"]
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != manifest["checkpoint"]["sha256"]:
        raise ValueError("deployment checkpoint hash does not match the deployment manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("checkpoint dataset revision does not match the package")
    if checkpoint["vocabulary_hash"] != vocabulary_hash:
        raise ValueError("checkpoint vocabulary hash does not match the package")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    config["dataset_size"] = manifest["dataset"]["song_count"]
    config["artifact_id"] = manifest["artifact_id"]
    return model, vocabulary, config


def generate(
    model: torch.nn.Module, vocabulary: dict[str, int], config: dict, seed: Path
) -> tuple[list[str], int]:
    encoded = encode_midi(seed)
    sequence_length = config["training"]["sequence_length"]
    generated = [vocabulary.get(token, vocabulary["<UNK>"]) for token in encoded.tokens]
    if len(generated) < sequence_length:
        raise ValueError(f"seed contains fewer than {sequence_length} encoded tokens")
    generated = generated[:sequence_length]
    inverse = {index: token for token, index in vocabulary.items()}
    generation = config["generation"]
    sampler = torch.Generator(device="cpu").manual_seed(generation["seed"])
    with torch.no_grad():
        for _ in range(generation["token_count"]):
            context = torch.tensor([generated[-sequence_length:]], dtype=torch.long)
            logits = model(context)[0].float() / generation["temperature"]
            top_k = min(generation["top_k"], logits.shape[0])
            values, indices = torch.topk(logits, top_k)
            sampled = torch.multinomial(
                torch.softmax(values, dim=-1), 1, generator=sampler
            )
            generated.append(int(indices[sampled].item()))
    return [inverse[index] for index in generated], encoded.ticks_per_beat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "generation" / "generated.mid",
    )
    args = parser.parse_args()

    model, vocabulary, config = load_local_model(args.artifact_dir)
    tokens, ticks_per_beat = generate(model, vocabulary, config, args.seed)
    decode_tokens(tokens, args.output, ticks_per_beat=ticks_per_beat)
    print(
        f"model={config['model_family']}-{config['model_profile']} "
        f"dataset={config['dataset_size']} vocabulary={len(vocabulary)}"
    )
    print(f"seed={args.seed}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

"""Load the recovered 2,500-song GRU-large checkpoint and generate MIDI locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data_prep.midi_representation import decode_tokens, encode_midi
from src.models.compact_midi_models import build_compact_model, count_parameters


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "local_artifacts" / "gru_large_2500"
DEFAULT_SEED = ROOT / "data" / "seed" / "Boom_Boom_Boom.mid"


def load_local_model(artifact_dir: Path) -> tuple[torch.nn.Module, dict[str, int], dict]:
    config = json.loads((artifact_dir / "config" / "experiment_config.json").read_text())
    vocabulary_path = artifact_dir / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())
    vocabulary_hash = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    expected_hash = "adca960cd5bbd89500e8977a485f2e5ae5e07f80560fd648044f7bf309a1f267"
    if vocabulary_hash != expected_hash:
        raise ValueError("local vocabulary hash does not match the recovered checkpoint")
    if config["model_family"] != "gru" or config["model_profile"] != "large":
        raise ValueError("artifact package is not the recovered GRU-large model")

    model = build_compact_model("gru", len(vocabulary), config["model_parameters"])
    if count_parameters(model) != 22_155_835:
        raise ValueError("unexpected recovered GRU-large parameter count")

    checkpoint = torch.load(
        artifact_dir / "checkpoints" / "best.pt", map_location="cpu"
    )
    if checkpoint["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("checkpoint dataset revision does not match the package")
    if checkpoint["vocabulary_hash"] != vocabulary_hash:
        raise ValueError("checkpoint vocabulary hash does not match the package")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocabulary, config


def generate(model: torch.nn.Module, vocabulary: dict[str, int], config: dict, seed: Path) -> list[str]:
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
    print(f"model=gru-large dataset=2500 vocabulary={len(vocabulary)}")
    print(f"seed={args.seed}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

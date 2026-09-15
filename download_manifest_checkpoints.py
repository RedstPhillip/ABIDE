"""Download and verify the exact W&B checkpoints named by an ensemble manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abide_gnn.reproducibility import file_sha256 as file_sha256


def verify_checkpoint(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint artifact did not create {path}.")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path}: {actual} != {expected_sha256}"
        )


def download_manifest_checkpoints(manifest_path):
    import wandb

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    api = wandb.Api()
    paths = []
    for item in manifest["checkpoints"]:
        destination = (manifest_path.parent / item["path"]).resolve()
        expected_sha256 = item["sha256"]
        if destination.exists():
            verify_checkpoint(destination, expected_sha256)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            api.artifact(item["artifact"], type="model").download(
                root=str(destination.parent)
            )
            verify_checkpoint(destination, expected_sha256)
        paths.append(destination)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Hydrate and SHA-256 verify an ensemble manifest from W&B."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    for path in download_manifest_checkpoints(args.manifest):
        print(path)


if __name__ == "__main__":
    main()

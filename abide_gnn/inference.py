import argparse
import json

import numpy as np
import torch

from .data import fetch_cc200_atlas, load_cc200_coordinates
from .ensemble import (
    calibrate_probability,
    checkpoint_paths_from_manifest,
    validate_checkpoint_ensemble,
)
from .graph_builder import create_graph
from .models import BrainGraphClassifier


def load_trained_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = BrainGraphClassifier(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_roi_file(checkpoint_path, roi_file, data_dir, device="cpu"):
    return predict_roi_file_ensemble(
        [checkpoint_path],
        roi_file=roi_file,
        data_dir=data_dir,
        device=device,
    )


def predict_roi_file_ensemble(
    checkpoint_paths,
    roi_file,
    data_dir,
    device="cpu",
    probability_calibration=None,
    threshold=0.5,
):
    """Average checkpoint probabilities for one ROI time-series file."""
    config, label_mapping = validate_checkpoint_ensemble(
        checkpoint_paths,
        device="cpu",
    )
    atlas_path = fetch_cc200_atlas(data_dir)
    coordinates = load_cc200_coordinates(atlas_path)
    roi_time_series = np.loadtxt(roi_file)
    graph = create_graph(
        roi_time_series=roi_time_series,
        roi_coordinates=coordinates,
        **config["graph"],
    ).to(device)

    member_probabilities = []
    for checkpoint_path in checkpoint_paths:
        model, _ = load_trained_model(checkpoint_path, device=device)
        with torch.no_grad():
            probabilities = model(graph).softmax(dim=1).squeeze(0)
        member_probabilities.append(probabilities.detach().cpu())
        del model

    probabilities = torch.stack(member_probabilities).mean(dim=0)
    raw_probability_asd = float(probabilities[1])
    if probability_calibration is not None:
        probability_asd = calibrate_probability(
            raw_probability_asd,
            probability_calibration,
        )
        probabilities = torch.tensor(
            [1.0 - probability_asd, probability_asd],
            dtype=probabilities.dtype,
        )

    predicted_label = int(float(probabilities[1]) >= threshold)
    result = {
        "predicted_class": label_mapping[predicted_label],
        "probabilities": {
            label_mapping[index]: probability.item()
            for index, probability in enumerate(probabilities)
        },
        "ensemble_size": len(checkpoint_paths),
        "member_asd_probabilities": [
            float(probabilities[1]) for probabilities in member_probabilities
        ],
        "notice": "Research-model output only; not a clinical diagnosis.",
    }
    if probability_calibration is not None:
        result["raw_probability_asd"] = raw_probability_asd
        result["probability_calibration"] = probability_calibration
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run an ABIDE CNN-GNN research-model prediction."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--ensemble", help="Path to an ensemble.json manifest.")
    parser.add_argument("--roi-file", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    if arguments.ensemble:
        checkpoint_paths, manifest = checkpoint_paths_from_manifest(arguments.ensemble)
        probability_calibration = manifest.get("probability_calibration")
        threshold = manifest.get("threshold", 0.5)
    else:
        checkpoint_paths = [arguments.checkpoint]
        probability_calibration = None
        threshold = 0.5

    prediction = predict_roi_file_ensemble(
        checkpoint_paths=checkpoint_paths,
        roi_file=arguments.roi_file,
        data_dir=arguments.data_dir,
        device=arguments.device,
        probability_calibration=probability_calibration,
        threshold=threshold,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()

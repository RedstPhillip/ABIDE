import argparse
import json

import numpy as np
import torch

from .data import fetch_cc200_atlas, load_cc200_coordinates
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
    model, checkpoint = load_trained_model(checkpoint_path, device=device)
    atlas_path = fetch_cc200_atlas(data_dir)
    coordinates = load_cc200_coordinates(atlas_path)
    roi_time_series = np.loadtxt(roi_file)
    graph = create_graph(
        roi_time_series=roi_time_series,
        roi_coordinates=coordinates,
        **checkpoint["config"]["graph"],
    ).to(device)

    with torch.no_grad():
        probabilities = model(graph).softmax(dim=1).squeeze(0)

    predicted_label = int(probabilities.argmax().item())
    label_mapping = checkpoint.get(
        "label_mapping",
        {0: "Control", 1: "ASD"},
    )
    return {
        "predicted_class": label_mapping[predicted_label],
        "probabilities": {
            label_mapping[index]: probability.item()
            for index, probability in enumerate(probabilities)
        },
        "notice": "Research-model output only; not a clinical diagnosis.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run an ABIDE CNN-GNN research-model prediction."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi-file", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    prediction = predict_roi_file(
        checkpoint_path=arguments.checkpoint,
        roi_file=arguments.roi_file,
        data_dir=arguments.data_dir,
        device=arguments.device,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()

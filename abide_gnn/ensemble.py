"""Reusable checkpoint ensembles and locked-test evaluation."""

import argparse
import copy
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from .data import fetch_cc200_atlas, load_abide_subjects
from .dataset import (
    build_graphs,
    create_graph_loader,
    create_subject_split,
    split_locked_test_sites,
    summarize_graphs,
)
from .models import BrainGraphClassifier
from .reproducibility import participant_hash as hash_subject_ids
from .training import classification_metrics, predict

MANIFEST_SCHEMA_VERSION = 1


def resolve_device(requested="auto"):
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def participant_hash(graphs):
    return hash_subject_ids(graph.subject_id for graph in graphs)


def load_checkpoint_payload(checkpoint_path, device="cpu"):
    """Load a training checkpoint without constructing its model."""
    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


def _inference_config(checkpoint):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dictionary.")
    required = ("data", "graph", "split", "loader", "model")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(
            "Checkpoint config is missing required sections: " + ", ".join(missing)
        )
    inference_config = {name: copy.deepcopy(config[name]) for name in required}

    if "in_dim" in inference_config["model"]:
        inference_config["graph"].setdefault(
            "connectivity_timepoints",
            "target",
        )
    else:
        inference_config["graph"].setdefault(
            "connectivity_timepoints",
            "full",
        )
    return inference_config


def validate_checkpoint_ensemble(checkpoint_paths, device="cpu"):
    """Require every checkpoint to use the same inference-relevant config."""
    checkpoint_paths = [Path(path) for path in checkpoint_paths]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")

    reference_config = None
    label_mapping = None
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
        config = _inference_config(checkpoint)
        if reference_config is None:
            reference_config = config
            label_mapping = checkpoint.get("label_mapping", {0: "Control", 1: "ASD"})
        elif config != reference_config:
            raise ValueError(
                "All ensemble checkpoints must use identical data, graph, split, "
                f"loader, and model configs. Mismatch: {checkpoint_path}"
            )
        if checkpoint.get("label_mapping", {0: "Control", 1: "ASD"}) != label_mapping:
            raise ValueError(f"Checkpoint label mapping differs: {checkpoint_path}")
    return reference_config, label_mapping


def average_prediction_sets(prediction_sets, threshold=0.5):
    """Align participant predictions and average ASD probabilities."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if not prediction_sets:
        raise ValueError("At least one prediction set is required.")

    aligned = []
    reference_ids = None
    reference = None
    for member_index, rows in enumerate(prediction_sets, start=1):
        by_id = {str(row["participant_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(
                f"Ensemble member {member_index} has duplicate participant IDs."
            )
        participant_ids = sorted(by_id)
        if reference_ids is None:
            reference_ids = participant_ids
            reference = by_id
        elif participant_ids != reference_ids:
            raise ValueError(
                f"Ensemble member {member_index} has a different participant set."
            )
        aligned.append(by_id)

    ensemble_rows = []
    for participant_id in reference_ids:
        first = reference[participant_id]
        member_probabilities = []
        for member_index, by_id in enumerate(aligned, start=1):
            row = by_id[participant_id]
            if int(row["true_label"]) != int(first["true_label"]):
                raise ValueError(
                    f"Label mismatch for participant {participant_id} in member {member_index}."
                )
            if str(row["site_id"]) != str(first["site_id"]):
                raise ValueError(
                    f"Site mismatch for participant {participant_id} in member {member_index}."
                )
            member_probabilities.append(float(row["probability_asd"]))

        probability_asd = float(np.mean(member_probabilities))
        predicted_label = int(probability_asd >= threshold)
        ensemble_rows.append(
            {
                "participant_id": participant_id,
                "site_id": str(first["site_id"]),
                "true_label": int(first["true_label"]),
                "probability_control": 1.0 - probability_asd,
                "probability_asd": probability_asd,
                "predicted_label": predicted_label,
                "member_probabilities_asd": member_probabilities,
            }
        )
    return ensemble_rows


def calibrate_probability(probability, calibration):
    """Apply a manifest-defined monotonic probability calibration transform."""
    if calibration is None:
        return float(probability)
    if calibration.get("type") != "platt_logit":
        raise ValueError(f"Unsupported probability calibration: {calibration!r}")
    coefficient = float(calibration["coefficient"])
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Platt-logit calibration must be monotonic increasing.")
    intercept = float(calibration["intercept"])
    if not math.isfinite(intercept):
        raise ValueError("Calibration intercept must be finite.")
    epsilon = float(calibration.get("clip_epsilon", 1e-5))
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("Calibration clip_epsilon must be between 0 and 0.5.")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be finite and between 0 and 1.")
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    logit = math.log(clipped / (1.0 - clipped))
    calibrated_logit = intercept + coefficient * logit
    if calibrated_logit >= 0:
        return 1.0 / (1.0 + math.exp(-calibrated_logit))
    exponential = math.exp(calibrated_logit)
    return exponential / (1.0 + exponential)


def calibrate_prediction_rows(rows, calibration, threshold=0.5):
    """Calibrate ensemble means while retaining raw and member probabilities."""
    calibrated_rows = []
    for row in rows:
        updated = dict(row)
        raw_probability = float(row["probability_asd"])
        probability_asd = calibrate_probability(raw_probability, calibration)
        updated["raw_probability_asd"] = raw_probability
        updated["probability_asd"] = probability_asd
        updated["probability_control"] = 1.0 - probability_asd
        updated["predicted_label"] = int(probability_asd >= threshold)
        calibrated_rows.append(updated)
    return calibrated_rows


def predict_checkpoint_ensemble(
    checkpoint_paths,
    loader,
    device,
    threshold=0.5,
    probability_calibration=None,
):
    """Run checkpoints sequentially and average their participant probabilities."""
    validate_checkpoint_ensemble(checkpoint_paths, device="cpu")
    prediction_sets = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
        model = BrainGraphClassifier(**checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        prediction_sets.append(predict(model, loader, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = average_prediction_sets(prediction_sets, threshold=threshold)
    if probability_calibration is not None:
        rows = calibrate_prediction_rows(
            rows,
            probability_calibration,
            threshold=threshold,
        )
    return rows


def ensemble_metrics(predictions):
    """Calculate the same metrics as training plus probability log loss."""
    labels = [row["true_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    probabilities = [row["probability_asd"] for row in predictions]
    metrics = classification_metrics(labels, predicted, probabilities)
    probability_matrix = [[1.0 - value, value] for value in probabilities]
    metrics["loss"] = float(log_loss(labels, probability_matrix, labels=[0, 1]))
    return metrics


def ensemble_member_metrics(predictions, checkpoints, threshold=0.5):
    """Calculate comparable test metrics for every ensemble member."""
    if not predictions:
        raise ValueError("At least one prediction is required.")
    member_count = len(predictions[0]["member_probabilities_asd"])
    if member_count != len(checkpoints):
        raise ValueError("Checkpoint and member-probability counts differ.")

    labels = [row["true_label"] for row in predictions]
    results = []
    for index, checkpoint in enumerate(checkpoints):
        probabilities = [row["member_probabilities_asd"][index] for row in predictions]
        predicted = [int(value >= threshold) for value in probabilities]
        metrics = classification_metrics(labels, predicted, probabilities)
        probability_matrix = [[1.0 - value, value] for value in probabilities]
        metrics["loss"] = float(log_loss(labels, probability_matrix, labels=[0, 1]))
        results.append(
            {
                "seed": checkpoint.get("seed"),
                "run_id": checkpoint.get("run_id"),
                "run_name": checkpoint.get("run_name"),
                "metrics": metrics,
            }
        )
    return results


def configured_test_graphs(config, data_dir):
    """Rebuild the exact test split described by a checkpoint config."""
    data_dir = Path(data_dir)
    atlas_path = fetch_cc200_atlas(data_dir)
    subjects = load_abide_subjects(
        data_dir / "abide_pcp",
        n_subjects=config["data"].get("n_subjects"),
    )
    graphs = build_graphs(
        subjects,
        atlas_path=atlas_path,
        cache_dir=data_dir / "processed" / "graphs",
        **config["graph"],
    )
    split = config["split"]
    strategy = split["strategy"]
    if strategy == "subject":
        _, _, test_graphs = create_subject_split(
            graphs,
            train_fraction=split["train_fraction"],
            validation_fraction=split["validation_fraction"],
            test_fraction=split["test_fraction"],
            seed=split["seed"],
        )
    elif strategy == "site":
        _, test_graphs = split_locked_test_sites(
            graphs,
            test_sites=split["locked_test_sites"],
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")
    return test_graphs


def checkpoint_paths_from_manifest(manifest_path):
    """Resolve checkpoint paths stored relative to an ensemble manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ensemble manifest schema: {manifest.get('schema_version')!r}"
        )
    checkpoint_paths = [
        (manifest_path.parent / item["path"]).resolve()
        for item in manifest["checkpoints"]
    ]
    return checkpoint_paths, manifest


def write_test_outputs(
    output_dir,
    predictions,
    metrics,
    member_metrics,
    manifest,
    test_hash,
    test_summary,
):
    output_dir = Path(output_dir)
    metrics_payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ensemble_size": len(manifest["checkpoints"]),
        "threshold": manifest["threshold"],
        "test_participant_hash": test_hash,
        "test_summary": test_summary,
        "metrics": metrics,
        "member_metrics": member_metrics,
        "members": manifest["checkpoints"],
        "probability_calibration": manifest.get("probability_calibration"),
    }
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    prediction_path = output_dir / "test_predictions.csv"
    member_columns = [
        f"probability_asd_seed_{item.get('seed', index)}"
        for index, item in enumerate(manifest["checkpoints"], start=1)
    ]
    columns = [
        "participant_id",
        "site_id",
        "true_label",
        "probability_control",
        "probability_asd",
        "predicted_label",
        *member_columns,
    ]
    if manifest.get("probability_calibration") is not None:
        columns.insert(5, "raw_probability_asd")
    with prediction_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            output_row = {
                name: row[name] for name in columns if name not in member_columns
            }
            output_row.update(zip(member_columns, row["member_probabilities_asd"]))
            writer.writerow(output_row)
    return metrics_path, prediction_path


def _print_metrics(metrics):
    print("\nLocked test metrics")
    for name in (
        "loss",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "brier_score",
        "f1_asd",
        "sensitivity_asd",
        "specificity_control",
    ):
        value = metrics.get(name)
        print(f"  {name}: {'N/A' if value is None else f'{value:.6f}'}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint ensemble."
    )
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help="Required before evaluating held-out participants.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.confirm_test_evaluation:
        raise SystemExit("Test evaluation requires --confirm-test-evaluation.")
    manifest_path = args.ensemble.resolve()
    checkpoint_paths, manifest = checkpoint_paths_from_manifest(manifest_path)
    expected_hashes = {
        item["expected_test_participant_hash"]
        for item in manifest["checkpoints"]
        if item.get("expected_test_participant_hash")
    }
    if len(expected_hashes) > 1:
        raise ValueError("Ensemble manifest contains conflicting test hashes.")
    expected_test_hash = next(iter(expected_hashes), None)
    config, _ = validate_checkpoint_ensemble(checkpoint_paths, device="cpu")
    print(f"Ensemble manifest: {manifest_path}")
    print(f"Members: {len(checkpoint_paths)}")

    metrics_path = args.output_dir / "test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            f"This output directory already contains a test evaluation: {metrics_path}"
        )

    test_graphs = configured_test_graphs(config, args.data_dir)
    actual_test_hash = participant_hash(test_graphs)
    if expected_test_hash and actual_test_hash != expected_test_hash:
        raise ValueError(
            "Rebuilt test participant hash does not match the W&B runs: "
            f"{actual_test_hash} != {expected_test_hash}"
        )

    device = resolve_device(args.device)
    loader = create_graph_loader(
        test_graphs,
        batch_size=config["loader"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions = predict_checkpoint_ensemble(
        checkpoint_paths,
        loader,
        device,
        threshold=manifest["threshold"],
        probability_calibration=manifest.get("probability_calibration"),
    )
    metrics = ensemble_metrics(predictions)
    member_metrics = ensemble_member_metrics(
        predictions,
        manifest["checkpoints"],
        threshold=manifest["threshold"],
    )
    metrics_path, prediction_path = write_test_outputs(
        args.output_dir,
        predictions,
        metrics,
        member_metrics,
        manifest,
        actual_test_hash,
        summarize_graphs(test_graphs),
    )
    _print_metrics(metrics)
    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()

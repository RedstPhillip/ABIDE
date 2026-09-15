import os
from numbers import Number
from pathlib import Path

import wandb

TRAIN_METRICS = ("loss", "accuracy", "roc_auc")
VALIDATION_METRICS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "brier_score",
    "f1_asd",
    "sensitivity_asd",
    "specificity_control",
)


def login_wandb():
    """Log in interactively, except during offline or disabled tests."""
    if os.environ.get("WANDB_MODE", "online").lower() in {"offline", "disabled"}:
        return False
    return wandb.login()


def initialize_wandb_run(config, artifacts_dir):
    """Start a W&B run and let W&B generate its name."""
    wandb_config = config["wandb"]
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    init_options = {
        "project": wandb_config["project"],
        "config": config,
        "tags": wandb_config.get("tags", []),
        "dir": str(artifacts_dir),
    }
    if wandb_config.get("name"):
        init_options["name"] = wandb_config["name"]
    if wandb_config.get("group"):
        init_options["group"] = wandb_config["group"]
    if mode := os.environ.get("WANDB_MODE"):
        init_options["mode"] = mode
    return wandb.init(**init_options)


def log_wandb_epoch(run, epoch, train_metrics, validation_metrics, learning_rate):
    """Log the metrics used to judge training and validation performance."""
    metrics = {"epoch": epoch}

    for name in TRAIN_METRICS:
        metrics[f"train/{name}"] = train_metrics[name]
    for name in VALIDATION_METRICS:
        metrics[f"validation/{name}"] = validation_metrics[name]

    metrics["training/learning_rate"] = learning_rate
    run.log(metrics, step=epoch)


def log_wandb_evaluation(
    run,
    predictions,
    best_epoch,
    validation_metrics,
    checkpoint_path,
    training_seed,
    split_seed,
    split_strategy,
    candidate="best",
):
    """Log validation predictions, plots, metrics, and a candidate model."""
    if candidate == "best":
        log_namespace = "validation"
        summary_namespace = "best_validation"
        epoch_key = "best_epoch"
        epoch_column = "best_epoch"
    else:
        log_namespace = f"{candidate}_validation"
        summary_namespace = log_namespace
        epoch_key = f"{candidate}_epoch"
        epoch_column = epoch_key

    true_labels = [row["true_label"] for row in predictions]
    predicted_labels = [row["predicted_label"] for row in predictions]
    probabilities = [
        [row["probability_control"], row["probability_asd"]] for row in predictions
    ]
    prediction_columns = [
        "participant_id",
        "site_id",
        "true_label",
        "probability_control",
        "probability_asd",
        "predicted_label",
        epoch_column,
        "training_seed",
        "split_seed",
        "split_strategy",
    ]
    prediction_data = [
        [
            row["participant_id"],
            row["site_id"],
            row["true_label"],
            row["probability_control"],
            row["probability_asd"],
            row["predicted_label"],
            best_epoch,
            training_seed,
            split_seed,
            split_strategy,
        ]
        for row in predictions
    ]

    run.log(
        {
            f"{log_namespace}/predictions": wandb.Table(
                columns=prediction_columns,
                data=prediction_data,
            ),
            f"{log_namespace}/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=true_labels,
                preds=predicted_labels,
                class_names=["Control", "ASD"],
            ),
            f"{log_namespace}/roc_curve": wandb.plot.roc_curve(
                true_labels,
                probabilities,
                labels=["Control", "ASD"],
            ),
        }
    )

    run.summary[epoch_key] = best_epoch
    for name, value in validation_metrics.items():
        if isinstance(value, Number):
            run.summary[f"{summary_namespace}/{name}"] = value
        elif name == "confusion_matrix":
            run.summary[f"{summary_namespace}/{name}"] = value

    run.log_model(
        path=str(checkpoint_path),
        name=(
            f"{run.name}-model"
            if candidate == "best"
            else f"{run.name}-{candidate}-model"
        ),
        aliases=[candidate],
    )

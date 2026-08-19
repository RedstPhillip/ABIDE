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
    if mode := os.environ.get("WANDB_MODE"):
        init_options["mode"] = mode
    return wandb.init(**init_options)


def log_wandb_epoch(run, epoch, train_metrics, validation_metrics):
    """Log the metrics used to judge training and validation performance."""
    metrics = {"epoch": epoch}

    # Training needs a compact optimization view; validation gets class-aware metrics.
    for name in TRAIN_METRICS:
        metrics[f"train/{name}"] = train_metrics[name]
    for name in VALIDATION_METRICS:
        metrics[f"validation/{name}"] = validation_metrics[name]
    run.log(metrics, step=epoch)


def log_wandb_evaluation(
    run,
    predictions,
    best_epoch,
    validation_metrics,
    checkpoint_path,
):
    """Log final validation plots, summary metrics, and the best model."""
    true_labels = [row["true_label"] for row in predictions]
    predicted_labels = [row["predicted_label"] for row in predictions]
    probabilities = [
        [row["probability_control"], row["probability_asd"]]
        for row in predictions
    ]

    # These plots show ranking quality and the actual classification errors.
    run.log(
        {
            "validation/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=true_labels,
                preds=predicted_labels,
                class_names=["Control", "ASD"],
            ),
            "validation/roc_curve": wandb.plot.roc_curve(
                true_labels,
                probabilities,
                labels=["Control", "ASD"],
            ),
        }
    )

    run.summary["best_epoch"] = best_epoch
    for name, value in validation_metrics.items():
        if isinstance(value, Number):
            run.summary[f"best_validation/{name}"] = value

    run.log_model(
        path=str(checkpoint_path),
        name=f"{run.name}-model",
        aliases=["best"],
    )

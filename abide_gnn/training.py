import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


class ExponentialMovingAverage:
    """Maintain a detached exponential moving average of model state."""

    def __init__(self, decay):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in the interval [0, 1).")
        self.decay = decay
        self.state_dict = None

    def update(self, model):
        """Initialize from the first model or update the existing average."""
        model_state = model.state_dict()
        if self.state_dict is None:
            self.state_dict = {
                name: value.detach().clone() for name, value in model_state.items()
            }
            return

        with torch.no_grad():
            for name, value in model_state.items():
                source = value.detach()
                averaged = self.state_dict[name]
                if averaged.is_floating_point() or averaged.is_complex():
                    averaged.mul_(self.decay).add_(source, alpha=1 - self.decay)
                else:
                    averaged.copy_(source)

    @property
    def initialized(self):
        return self.state_dict is not None

    def copy_to(self, model):
        """Load the averaged state into ``model``."""
        if not self.initialized:
            raise RuntimeError("EMA has not received any model updates.")
        model.load_state_dict(self.state_dict)


def seed_everything(seed):
    """Seed training RNGs and require deterministic PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def classification_metrics(labels, predictions, asd_probabilities):
    """Calculate binary classification metrics with ASD as the positive class."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    asd_probabilities = np.asarray(asd_probabilities)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    roc_auc = None
    average_precision = None
    if np.unique(labels).size == 2:
        roc_auc = float(roc_auc_score(labels, asd_probabilities))
        average_precision = float(average_precision_score(labels, asd_probabilities))

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(labels, asd_probabilities)),
        "f1_asd": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity_asd": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_control": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _run_loader(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_graphs = 0
    labels = []
    predictions = []
    asd_probabilities = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)

            if is_training:
                loss.backward()
                optimizer.step()

            graph_count = batch.num_graphs
            total_loss += loss.item() * graph_count
            total_graphs += graph_count
            labels.extend(batch.y.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            asd_probabilities.extend(
                logits.softmax(dim=1)[:, 1].detach().cpu().tolist()
            )

    metrics = classification_metrics(labels, predictions, asd_probabilities)
    metrics["loss"] = total_loss / total_graphs
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    return _run_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )


def evaluate(model, loader, criterion, device):
    return _run_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
    )


def predict(model, loader, device):
    """Return participant-level predictions and probabilities."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            probabilities = model(batch).softmax(dim=1).detach().cpu()
            labels = batch.y.detach().cpu().tolist()
            participant_ids = _metadata_values(
                batch.subject_id,
                len(labels),
                "subject_id",
            )
            site_ids = _metadata_values(
                batch.site_id,
                len(labels),
                "site_id",
            )

            for index, label in enumerate(labels):
                predicted_label = int(probabilities[index].argmax().item())
                rows.append(
                    {
                        "participant_id": str(participant_ids[index]),
                        "site_id": str(site_ids[index]),
                        "true_label": int(label),
                        "predicted_label": predicted_label,
                        "probability_control": float(probabilities[index, 0]),
                        "probability_asd": float(probabilities[index, 1]),
                    }
                )
    return rows


def export_predictions_csv(predictions, path):
    """Write participant-aligned probabilities for an inspectable evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "participant_id",
        "site_id",
        "true_label",
        "predicted_label",
        "probability_control",
        "probability_asd",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(predictions)


def _metadata_values(values, expected_count, field_name):
    """Return one graph-level metadata value for each item in a PyG batch."""
    if isinstance(values, (list, tuple)):
        items = list(values)
    elif expected_count == 1:
        items = [values]
    else:
        raise ValueError(f"Batched {field_name} must contain {expected_count} values.")

    if len(items) != expected_count:
        raise ValueError(
            f"Batched {field_name} contains {len(items)} values; "
            f"expected {expected_count}."
        )
    return items


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    config,
    validation_metrics,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "validation_metrics": validation_metrics,
            "label_mapping": {0: "Control", 1: "ASD"},
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


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
    if np.unique(labels).size == 2:
        roc_auc = float(roc_auc_score(labels, asd_probabilities))

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": roc_auc,
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

    # The same loop is used for training and evaluation to keep metrics identical.
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)

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
            batch = batch.to(device)
            probabilities = model(batch).softmax(dim=1).detach().cpu()
            labels = batch.y.detach().cpu().tolist()

            # Keep only the values needed for final W&B plots.
            for index, label in enumerate(labels):
                predicted_label = int(probabilities[index].argmax().item())
                rows.append(
                    {
                        "true_label": int(label),
                        "predicted_label": predicted_label,
                        "probability_control": float(probabilities[index, 0]),
                        "probability_asd": float(probabilities[index, 1]),
                    }
                )
    return rows


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

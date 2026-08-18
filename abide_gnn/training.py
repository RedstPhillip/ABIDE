from pathlib import Path

import torch


def _run_loader(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

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
            total_correct += (logits.argmax(dim=1) == batch.y).sum().item()
            total_graphs += graph_count

    return {
        "loss": total_loss / total_graphs,
        "accuracy": total_correct / total_graphs,
    }


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

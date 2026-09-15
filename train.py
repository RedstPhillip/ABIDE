import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
import struct
import time
import warnings
from pathlib import Path

import torch

from abide_gnn import (
    BrainGraphClassifier,
    ExponentialMovingAverage,
    build_graphs,
    create_graph_loader,
    create_site_validation_folds,
    create_subject_development_split,
    create_subject_split,
    evaluate,
    export_predictions_csv,
    fetch_cc200_atlas,
    index_abide_subjects,
    initialize_wandb_run,
    load_abide_subjects,
    load_checkpoint,
    log_wandb_epoch,
    log_wandb_evaluation,
    login_wandb,
    materialize_abide_subjects,
    predict,
    save_checkpoint,
    seed_everything,
    select_development_sites,
    split_locked_test_sites,
    summarize_graphs,
    train_one_epoch,
)
from abide_gnn.reproducibility import git_commit as _git_commit
from abide_gnn.reproducibility import participant_hash

DEFAULT_LOCKED_TEST_SITES = ["CALTECH", "CMU", "UM_2"]


METRIC_DIRECTION = {
    "roc_auc": "maximize",
    "balanced_accuracy": "maximize",
    "f1_asd": "maximize",
    "accuracy": "maximize",
    "loss": "minimize",
}


def deep_merge(base, override):
    """Merge nested sweep overrides while preserving unspecified defaults."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN-GNN classifier on ABIDE resting-state fMRI data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing abide_pcp/ and atlases/.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for W&B logs and experiment checkpoints.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Validation fold index for the site strategy (0-based).",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Maximum training epochs."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early stopping patience. Stop after this many epochs without improvement. Set to 0 to disable.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=50,
        help="Warm-up epochs before checkpoint selection, LR scheduling, and early stopping begin.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=15,
        help="Validation-loss plateau epochs before reducing the learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Multiplicative learning-rate reduction after a validation-loss plateau.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Minimum learning rate used by the validation-loss scheduler.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Adam weight decay."
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument(
        "--node-feature-mode",
        choices=("temporal", "connectivity"),
        default="temporal",
        help="Use learned BOLD embeddings or full ROI connectivity profiles.",
    )
    parser.add_argument(
        "--coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append normalized CC200 coordinates to each ROI feature vector.",
    )
    parser.add_argument(
        "--temporal-pooling",
        choices=("max", "mean"),
        default="max",
        help="Pooling used by the temporal ROI encoder.",
    )
    parser.add_argument(
        "--graph-pooling",
        choices=("mean", "max", "mean_max", "roi_concat"),
        default="mean",
        help="Readout used to convert ROI embeddings into a graph embedding.",
    )
    parser.add_argument(
        "--concat-node-dim",
        type=int,
        default=8,
        help=(
            "Per-ROI dimension before fixed-order ROI concatenation; used "
            "only by --graph-pooling roi_concat."
        ),
    )
    parser.add_argument(
        "--gnn-type",
        choices=("graphconv", "gatv2", "appnp", "bnt"),
        default="graphconv",
        help="Graph model; BNT uses dense connectivity-profile self-attention.",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="Number of attention heads used by GATv2.",
    )
    parser.add_argument(
        "--appnp-steps",
        type=int,
        default=10,
        help="Number of PageRank propagation steps used by APPNP.",
    )
    parser.add_argument(
        "--appnp-alpha",
        type=float,
        default=0.1,
        help="APPNP teleport probability.",
    )
    parser.add_argument(
        "--appnp-dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied to APPNP propagation edges.",
    )
    parser.add_argument(
        "--bnt-clusters",
        type=int,
        default=100,
        help="OCRead cluster count; 100 is the official BNT configuration.",
    )
    parser.add_argument(
        "--bnt-heads",
        type=int,
        default=4,
        help="Self-attention heads per BNT layer (official BNT: 4).",
    )
    parser.add_argument(
        "--bnt-feedforward-dim",
        type=int,
        default=1024,
        help="BNT Transformer feed-forward width (official BNT: 1024).",
    )
    parser.add_argument(
        "--bnt-dropout",
        type=float,
        default=0.1,
        help="Dropout in BNT attention and feed-forward blocks.",
    )
    parser.add_argument(
        "--bnt-cluster-hidden-dim",
        type=int,
        default=32,
        help="Flattened assignment encoder width in BNT OCRead.",
    )
    parser.add_argument(
        "--bnt-orthogonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gram--Schmidt/QR orthonormal initialization for OCRead centers.",
    )
    parser.add_argument(
        "--bnt-freeze-centers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep BNT OCRead centers frozen as in the official configuration.",
    )
    parser.add_argument(
        "--bnt-project-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BNT's squared-projection soft assignment rather than Student-t.",
    )
    parser.add_argument(
        "--roi-embedding-dim",
        type=int,
        default=0,
        help="Learned ROI identity dimensions; set to 0 to disable.",
    )
    parser.add_argument(
        "--num-rois",
        type=int,
        default=200,
        help="Number of atlas ROIs (CC200 -> 200).",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.5,
        help="Absolute Pearson-correlation cutoff used to retain graph edges.",
    )
    parser.add_argument(
        "--bold-normalization",
        choices=("none", "per_roi_zscore"),
        default="per_roi_zscore",
        help="Normalization applied independently to each ROI time series.",
    )
    parser.add_argument(
        "--target-timepoints",
        type=int,
        default=196,
        help="Crop or pad every ROI time series to this length.",
    )
    parser.add_argument(
        "--connectivity-timepoints",
        choices=("full", "target"),
        default="full",
        help="Compute connectivity from the full or resized time series.",
    )
    parser.add_argument(
        "--negative-edge-policy",
        choices=("signed", "absolute", "positive_only"),
        default="signed",
        help="How negative functional-connectivity edges are represented.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("subject", "site"),
        default="subject",
        help="Use a random subject split or locked-site evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset split construction seed.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
        help="Model, optimizer, dropout, and training DataLoader RNG seed.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Limit the number of subjects (for quick tests).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Training fraction for subject splitting.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Validation fraction for subject splitting.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Test fraction for subject splitting.",
    )
    parser.add_argument(
        "--locked-test-sites",
        nargs="*",
        default=DEFAULT_LOCKED_TEST_SITES,
        help="Held-out test sites used only by the site strategy.",
    )
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=5,
        help="Number of validation folds used only by the site strategy.",
    )
    parser.add_argument("--tags", nargs="*", default=None, help="W&B run tags.")
    parser.add_argument(
        "--wandb-project", default="abide-gnn", help="W&B project name."
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Optional W&B group used to compare a related experiment family.",
    )
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--selection-metric",
        choices=tuple(METRIC_DIRECTION),
        default="loss",
        help="Metric used for best-checkpoint selection.",
    )
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate an exponential moving average of late-training weights.",
    )
    parser.add_argument(
        "--ema-start-epoch",
        type=int,
        default=200,
        help="First epoch included in the exponential moving average.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="Decay applied to the previous EMA weights.",
    )
    return parser.parse_args()


def build_config(args):
    """Build nested config dict from CLI arguments."""
    return {
        "data": {
            "n_subjects": args.n_subjects,
        },
        "graph": {
            "correlation_threshold": args.correlation_threshold,
            "bold_normalization": args.bold_normalization,
            "target_timepoints": args.target_timepoints,
            "connectivity_timepoints": args.connectivity_timepoints,
            "negative_edge_policy": args.negative_edge_policy,
        },
        "split": {
            "strategy": args.split_strategy,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "locked_test_sites": args.locked_test_sites,
            "validation_folds": args.validation_folds,
            "validation_fold": args.fold,
            "seed": args.seed,
        },
        "loader": {
            "batch_size": args.batch_size,
        },
        "model": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_classes": args.num_classes,
            "gnn_type": args.gnn_type,
            "attention_heads": args.attention_heads,
            "appnp_steps": args.appnp_steps,
            "appnp_alpha": args.appnp_alpha,
            "appnp_dropout": args.appnp_dropout,
            "bnt_clusters": args.bnt_clusters,
            "bnt_heads": args.bnt_heads,
            "bnt_feedforward_dim": args.bnt_feedforward_dim,
            "bnt_dropout": args.bnt_dropout,
            "bnt_cluster_hidden_dim": args.bnt_cluster_hidden_dim,
            "bnt_orthogonal": args.bnt_orthogonal,
            "bnt_freeze_centers": args.bnt_freeze_centers,
            "bnt_project_assignment": args.bnt_project_assignment,
            "num_rois": args.num_rois,
            "roi_embedding_dim": args.roi_embedding_dim,
            "node_feature_mode": args.node_feature_mode,
            "use_coordinates": args.coordinates,
            "temporal_pooling": args.temporal_pooling,
            "graph_pooling": args.graph_pooling,
            "concat_node_dim": args.concat_node_dim,
        },
        "training": {
            "seed": args.training_seed,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "selection_metric": args.selection_metric,
            "patience": args.patience,
            "min_epochs": args.min_epochs,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "min_learning_rate": args.min_learning_rate,
            "ema_enabled": args.ema,
            "ema_start_epoch": args.ema_start_epoch,
            "ema_decay": args.ema_decay,
        },
        "wandb": {
            "project": args.wandb_project,
            "tags": (
                list(dict.fromkeys([*(args.tags or []), "brain-transformer"]))
                if args.gnn_type == "bnt"
                else (args.tags or [])
            ),
            "name": args.wandb_run_name,
            "group": args.wandb_group,
        },
    }


def resolve_config(wandb_run, original_config):
    """Resolve W&B overrides without modifying locked sweep configuration."""
    wandb_overrides = {}
    for section in ("data", "graph", "split", "loader", "model", "training", "wandb"):
        section_value = wandb_run.config.get(section)
        if section_value is not None:
            wandb_overrides[section] = dict(section_value)
    return deep_merge(original_config, wandb_overrides)


def format_epoch_line(epoch, train_metrics, val_metrics, is_best):
    return (
        f"Epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f}, "
        f"accuracy {train_metrics['accuracy']:.3f}, "
        f"ROC-AUC {_fmt_auc(train_metrics['roc_auc'])} | "
        f"validation loss {val_metrics['loss']:.4f}, "
        f"accuracy {val_metrics['accuracy']:.3f}, "
        f"balanced accuracy {val_metrics['balanced_accuracy']:.3f}, "
        f"ROC-AUC {_fmt_auc(val_metrics['roc_auc'])}, "
        f"ASD F1 {val_metrics['f1_asd']:.3f}, "
        f"sensitivity {_fmt_optional(val_metrics['sensitivity_asd'])}, "
        f"specificity {_fmt_optional(val_metrics['specificity_control'])}"
        f"{' | saved best' if is_best else ''}"
    )


def _fmt_auc(value):
    """Format ROC-AUC which can be None for single-class splits."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _fmt_optional(value):
    """Format a metric that might be None."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _check_selection_metric(selection_metric, validation_metrics, validation_context):
    """Fail with a clear error if the selection metric is unavailable."""
    if selection_metric not in METRIC_DIRECTION:
        supported = ", ".join(sorted(METRIC_DIRECTION))
        raise ValueError(
            f"Unknown selection metric '{selection_metric}'. "
            f"Supported metrics: {supported}"
        )
    value = validation_metrics.get(selection_metric)
    if value is None:
        cm = validation_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        n_asd = fn + tp
        n_control = tn + fp
        n_total = n_asd + n_control
        raise ValueError(
            f"Selection metric '{selection_metric}' is unavailable (returned None) "
            f"on {validation_context}. The validation split may contain only one class: "
            f"{n_asd} ASD, {n_control} Control out of {n_total} total."
        )
    return value


def _source_include(path, root=None):
    """Filter for ``run.log_code``: include project Python files and config files.

    ``path`` is the absolute file path and ``root`` is the root directory
    that ``log_code`` is walking.  We compute the path relative to ``root``
    so that the matching is robust regardless of the absolute layout.
    """
    rel = os.path.relpath(path, root) if root else str(path)
    p = Path(rel)

    parts = p.parts
    for excluded in (
        ".git",
        ".venv",
        "__pycache__",
        "wandb",
        "artifacts",
        "data",
        "notebooks",
        "abide_gnn.egg-info",
    ):
        if excluded in parts:
            return False

    if p.name in ("requirements.txt", "pyproject.toml"):
        return True

    if p.suffix == ".py":
        if p.name == "train.py":
            return True
        if len(parts) >= 2 and parts[0] == "abide_gnn":
            return True
    return False


def _validate_selection_metric(selection_metric):
    """Check the metric name is supported and return its direction."""
    if selection_metric not in METRIC_DIRECTION:
        supported = ", ".join(sorted(METRIC_DIRECTION))
        raise ValueError(
            f"Unknown selection metric '{selection_metric}'. Supported: {supported}"
        )
    return METRIC_DIRECTION[selection_metric]


def _is_better(score, best, direction):
    """Compare *score* against *best* using *direction*."""
    if score is None:
        return False
    if direction == "maximize":
        return score > best
    return score < best


def _validate_training_schedule(training_config):
    """Validate warm-up, early-stopping, and LR-scheduler settings."""
    epochs = training_config["epochs"]
    min_epochs = training_config["min_epochs"]
    scheduler_patience = training_config["lr_scheduler_patience"]
    scheduler_factor = training_config["lr_scheduler_factor"]
    min_learning_rate = training_config["min_learning_rate"]
    ema_start_epoch = training_config["ema_start_epoch"]
    ema_decay = training_config["ema_decay"]

    if not 1 <= min_epochs <= epochs:
        raise ValueError("training.min_epochs must be between 1 and training.epochs.")
    if scheduler_patience < 0:
        raise ValueError("training.lr_scheduler_patience must be non-negative.")
    if not 0 < scheduler_factor < 1:
        raise ValueError("training.lr_scheduler_factor must be between 0 and 1.")
    if not 0 <= min_learning_rate <= training_config["learning_rate"]:
        raise ValueError(
            "training.min_learning_rate must be non-negative and no greater "
            "than training.learning_rate."
        )
    if ema_start_epoch < 1:
        raise ValueError("training.ema_start_epoch must be at least 1.")
    if not 0 <= ema_decay < 1:
        raise ValueError("training.ema_decay must be in the interval [0, 1).")


def _validate_model_graph_compatibility(model_config, graph_config):
    """Reject model/graph combinations whose propagation is not well-defined."""
    if (
        model_config["gnn_type"] == "appnp"
        and graph_config["negative_edge_policy"] == "signed"
    ):
        raise ValueError(
            "APPNP requires nonnegative edge weights; use "
            "graph.negative_edge_policy='absolute' or 'positive_only'."
        )


def _create_lr_scheduler(optimizer, training_config):
    """Create the validation-loss learning-rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config["lr_scheduler_factor"],
        patience=training_config["lr_scheduler_patience"],
        min_lr=training_config["min_learning_rate"],
    )


def main():
    warnings.filterwarnings("ignore", message="IProgress not found.*")

    args = parse_args()
    original_config = build_config(args)

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    login_wandb()
    wandb_run = initialize_wandb_run(original_config, args.artifacts_dir)

    try:
        _run(wandb_run, args, original_config)
    except Exception:
        wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish()


def _run(wandb_run, args, original_config):
    """Core training logic separated so ``main`` can wrap it exception-safe."""

    cfg = resolve_config(wandb_run, original_config)

    selection_metric = cfg["training"]["selection_metric"]
    metric_direction = _validate_selection_metric(selection_metric)
    _validate_training_schedule(cfg["training"])

    wandb_run.summary["effective_config"] = cfg

    data_cfg = cfg["data"]
    graph_cfg = cfg["graph"]
    split_cfg = cfg["split"]
    loader_cfg = cfg["loader"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    _validate_model_graph_compatibility(model_cfg, graph_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(training_cfg["seed"])

    wandb_run.log_code(root=".", include_fn=_source_include)

    split_strategy = split_cfg["strategy"]

    development_only = True
    if development_only:
        subject_index = index_abide_subjects(
            args.data_dir / "abide_pcp",
            n_subjects=data_cfg["n_subjects"],
        )
        if split_strategy == "subject":
            development_train, development_validation = (
                create_subject_development_split(
                    subject_index,
                    train_fraction=split_cfg["train_fraction"],
                    validation_fraction=split_cfg["validation_fraction"],
                    test_fraction=split_cfg["test_fraction"],
                    seed=split_cfg["seed"],
                )
            )
            fold_index = None
            validation_context = "subject validation split"
        elif split_strategy == "site":
            development_subjects = select_development_sites(
                subject_index,
                test_sites=split_cfg["locked_test_sites"],
            )
            validation_folds = create_site_validation_folds(
                development_subjects,
                n_splits=split_cfg["validation_folds"],
                seed=split_cfg["seed"],
            )
            fold_index = split_cfg["validation_fold"]
            development_train, development_validation = validation_folds[fold_index]
            validation_context = f"site fold {fold_index}"
        else:
            raise ValueError(
                f"Unknown split strategy '{split_strategy}'. Use 'subject' or 'site'."
            )
        train_ids = {subject.subject_id for subject in development_train}
        validation_ids = {subject.subject_id for subject in development_validation}
        development_ids = train_ids | validation_ids

        subjects = materialize_abide_subjects(
            [
                subject
                for subject in subject_index
                if subject.subject_id in development_ids
            ]
        )
    else:
        subjects = load_abide_subjects(
            args.data_dir / "abide_pcp",
            n_subjects=data_cfg["n_subjects"],
        )

    data_dir = args.data_dir
    atlas_path = fetch_cc200_atlas(data_dir)
    graphs = build_graphs(
        subjects,
        atlas_path=atlas_path,
        cache_dir=data_dir / "processed" / "graphs",
        **graph_cfg,
    )

    if development_only:
        train_graphs = [graph for graph in graphs if graph.subject_id in train_ids]
        validation_graphs = [
            graph for graph in graphs if graph.subject_id in validation_ids
        ]
        test_graphs = None
    elif split_strategy == "subject":
        train_graphs, validation_graphs, test_graphs = create_subject_split(
            graphs,
            train_fraction=split_cfg["train_fraction"],
            validation_fraction=split_cfg["validation_fraction"],
            test_fraction=split_cfg["test_fraction"],
            seed=split_cfg["seed"],
        )
        fold_index = None
        validation_context = "subject validation split"
    elif split_strategy == "site":
        development_graphs, test_graphs = split_locked_test_sites(
            graphs, test_sites=split_cfg["locked_test_sites"]
        )
        validation_folds = create_site_validation_folds(
            development_graphs,
            n_splits=split_cfg["validation_folds"],
            seed=split_cfg["seed"],
        )
        fold_index = split_cfg["validation_fold"]
        train_graphs, validation_graphs = validation_folds[fold_index]
        validation_context = f"site fold {fold_index}"
    else:
        raise ValueError(
            f"Unknown split strategy '{split_strategy}'. Use 'subject' or 'site'."
        )

    train_participant_hash = _participant_hash(train_graphs)
    validation_participant_hash = _participant_hash(validation_graphs)
    split_metadata = {
        "train_sites": sorted({g.site_id for g in train_graphs}),
        "validation_sites": sorted({g.site_id for g in validation_graphs}),
        "split_summary": {
            "train": summarize_graphs(train_graphs),
            "validation": summarize_graphs(validation_graphs),
        },
        "train_participant_hash": train_participant_hash,
        "validation_participant_hash": validation_participant_hash,
    }
    if test_graphs is not None:
        split_metadata["test_sites"] = sorted({g.site_id for g in test_graphs})
        split_metadata["split_summary"]["test"] = summarize_graphs(test_graphs)
        split_metadata["test_participant_hash"] = _participant_hash(test_graphs)
    wandb_run.config.update(split_metadata, allow_val_change=True)

    train_loader = create_graph_loader(
        train_graphs,
        batch_size=loader_cfg["batch_size"],
        shuffle=True,
        seed=training_cfg["seed"],
    )
    validation_loader = create_graph_loader(
        validation_graphs,
        batch_size=loader_cfg["batch_size"],
    )

    model = BrainGraphClassifier(**model_cfg).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    initial_parameter_hash = _model_parameter_hash(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    lr_scheduler = _create_lr_scheduler(optimizer, training_cfg)

    print(f"Device: {device}")
    print(f"Split strategy: {split_strategy}")
    if fold_index is not None:
        print(f"Fold: {fold_index}")
    print(f"Train: {len(train_graphs)}  |  Validation: {len(validation_graphs)}")
    if development_only:
        print("Held-out test partition: reserved and not inspected.")
    else:
        print(f"Test: {len(test_graphs)}")
    print(f"Selection metric: {selection_metric} ({metric_direction})")
    print(f"GNN type: {model_cfg['gnn_type']}")
    print(f"Node features: {model_cfg['node_feature_mode']}")
    print(f"Graph pooling: {model_cfg['graph_pooling']}")
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Train participant hash: {train_participant_hash}")
    print(f"Validation participant hash: {validation_participant_hash}")
    print(f"Initial parameter hash: {initial_parameter_hash}")

    wandb_run.summary["reproducibility/train_participant_hash"] = train_participant_hash
    wandb_run.summary["reproducibility/validation_participant_hash"] = (
        validation_participant_hash
    )
    wandb_run.summary["reproducibility/initial_parameter_hash"] = initial_parameter_hash
    wandb_run.summary["environment/git_commit"] = _git_commit()
    wandb_run.summary["environment/device"] = str(device)
    if device.type == "cuda":
        wandb_run.summary["environment/gpu"] = torch.cuda.get_device_name(device)
    wandb_run.summary["environment/cuda"] = torch.version.cuda
    wandb_run.summary["environment/pytorch"] = torch.__version__
    wandb_run.summary["environment/pyg"] = _pyg_version()
    wandb_run.summary["model/trainable_parameters"] = trainable_parameters

    run_name = wandb_run.name or wandb_run.id or "unnamed"
    run_dir = args.artifacts_dir / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    final_checkpoint_path = run_dir / "final_model.pt"
    ema_checkpoint_path = run_dir / "ema_model.pt"

    patience = training_cfg.get("patience")
    min_epochs = training_cfg["min_epochs"]
    use_early_stopping = patience is not None and patience > 0
    initial_best = float("inf") if metric_direction == "minimize" else float("-inf")
    best_validation_score = initial_best
    epochs_without_improvement = 0
    ema = None
    if training_cfg["ema_enabled"]:
        ema = ExponentialMovingAverage(training_cfg["ema_decay"])

    training_started = time.perf_counter()
    epoch_durations = []
    for epoch in range(1, training_cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        if ema is not None and epoch >= training_cfg["ema_start_epoch"]:
            ema.update(model)
        validation_metrics = evaluate(model, validation_loader, criterion, device)

        is_best = False
        if epoch >= min_epochs:
            current_score = _check_selection_metric(
                selection_metric, validation_metrics, validation_context
            )
            is_best = _is_better(
                current_score,
                best_validation_score,
                metric_direction,
            )

            if is_best:
                best_validation_score = current_score
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=cfg,
                    validation_metrics=validation_metrics,
                )
            else:
                epochs_without_improvement += 1

            lr_scheduler.step(validation_metrics["loss"])

        current_learning_rate = optimizer.param_groups[0]["lr"]
        log_wandb_epoch(
            wandb_run,
            epoch,
            train_metrics,
            validation_metrics,
            current_learning_rate,
        )
        epoch_durations.append(time.perf_counter() - epoch_started)
        print(format_epoch_line(epoch, train_metrics, validation_metrics, is_best))

        if use_early_stopping and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
            )
            break

    final_epoch = epoch
    final_validation_metrics = evaluate(model, validation_loader, criterion, device)
    final_validation_predictions = predict(model, validation_loader, device)
    final_model_hash = _model_parameter_hash(model)
    final_prediction_hash = _ordered_probability_hash(final_validation_predictions)
    export_predictions_csv(
        final_validation_predictions,
        run_dir / "final_validation_predictions.csv",
    )
    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=final_epoch,
        config=cfg,
        validation_metrics=final_validation_metrics,
    )

    best_epoch = epoch
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        best_epoch = checkpoint["epoch"]
    best_model_hash = _model_parameter_hash(model)

    best_validation_metrics = evaluate(model, validation_loader, criterion, device)
    validation_predictions = predict(model, validation_loader, device)
    export_predictions_csv(
        validation_predictions,
        run_dir / "best_validation_predictions.csv",
    )
    prediction_hash = _ordered_probability_hash(validation_predictions)
    training_seconds = time.perf_counter() - training_started

    wandb_run.summary["reproducibility/ordered_validation_probability_hash"] = (
        prediction_hash
    )
    wandb_run.summary["reproducibility/best_model_hash"] = best_model_hash
    wandb_run.summary["reproducibility/final_model_hash"] = final_model_hash
    wandb_run.summary["reproducibility/final_ordered_validation_probability_hash"] = (
        final_prediction_hash
    )
    wandb_run.summary["performance/training_seconds"] = training_seconds
    wandb_run.summary["performance/mean_epoch_seconds"] = sum(epoch_durations) / len(
        epoch_durations
    )

    log_wandb_evaluation(
        wandb_run,
        predictions=validation_predictions,
        best_epoch=best_epoch,
        validation_metrics=best_validation_metrics,
        checkpoint_path=checkpoint_path,
        training_seed=training_cfg["seed"],
        split_seed=split_cfg["seed"],
        split_strategy=split_strategy,
    )
    log_wandb_evaluation(
        wandb_run,
        predictions=final_validation_predictions,
        best_epoch=final_epoch,
        validation_metrics=final_validation_metrics,
        checkpoint_path=final_checkpoint_path,
        training_seed=training_cfg["seed"],
        split_seed=split_cfg["seed"],
        split_strategy=split_strategy,
        candidate="final",
    )

    if ema is not None and ema.initialized:
        ema.copy_to(model)
        ema_validation_metrics = evaluate(model, validation_loader, criterion, device)
        ema_validation_predictions = predict(model, validation_loader, device)
        ema_model_hash = _model_parameter_hash(model)
        ema_prediction_hash = _ordered_probability_hash(ema_validation_predictions)
        save_checkpoint(
            checkpoint_path=ema_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=final_epoch,
            config=cfg,
            validation_metrics=ema_validation_metrics,
        )
        wandb_run.summary["reproducibility/ema_model_hash"] = ema_model_hash
        wandb_run.summary["reproducibility/ema_ordered_validation_probability_hash"] = (
            ema_prediction_hash
        )
        log_wandb_evaluation(
            wandb_run,
            predictions=ema_validation_predictions,
            best_epoch=final_epoch,
            validation_metrics=ema_validation_metrics,
            checkpoint_path=ema_checkpoint_path,
            training_seed=training_cfg["seed"],
            split_seed=split_cfg["seed"],
            split_strategy=split_strategy,
            candidate="ema",
        )
        ema_value = ema_validation_metrics.get(selection_metric)
        ema_value_str = f"{ema_value:.4f}" if ema_value is not None else "N/A"
        print(f"EMA validation {selection_metric}: {ema_value_str}")
        print(f"EMA checkpoint: {ema_checkpoint_path}")
    elif ema is not None:
        wandb_run.summary["ema_status"] = (
            "unavailable: training ended before ema_start_epoch"
        )

    best_val = best_validation_metrics.get(selection_metric)
    best_str = f"{best_val:.4f}" if best_val is not None else "N/A"
    print(f"\nBest epoch: {best_epoch}")
    print(f"Validation {selection_metric}: {best_str}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best model hash: {best_model_hash}")
    print(f"Ordered validation probability hash: {prediction_hash}")
    print(
        f"Training time: {training_seconds:.2f}s ({sum(epoch_durations) / len(epoch_durations):.2f}s/epoch mean)"
    )


def _model_parameter_hash(model):
    """Hash model state in a stable name/dtype/shape/value order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _participant_hash(graphs):
    return participant_hash(graph.subject_id for graph in graphs)


def _ordered_probability_hash(predictions):
    """Hash ordered validation ASD probabilities as IEEE-754 doubles."""
    digest = hashlib.sha256()
    for row in predictions:
        digest.update(struct.pack("!d", row["probability_asd"]))
    return digest.hexdigest()


def _pyg_version():
    import torch_geometric

    return torch_geometric.__version__


if __name__ == "__main__":
    main()

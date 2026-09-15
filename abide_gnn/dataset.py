import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)
from torch_geometric.loader import DataLoader

from .data import load_cc200_coordinates
from .graph_builder import create_graph, normalize_coordinates
from .reproducibility import file_sha256 as _file_sha256

GRAPH_SCHEMA_VERSION = 3


def seed_worker(worker_id):
    """Seed Python and NumPy from each DataLoader worker's PyTorch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _graph_configuration(
    atlas_path,
    correlation_threshold,
    bold_normalization,
    target_timepoints,
    connectivity_timepoints,
    negative_edge_policy,
):
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "atlas_sha256": _file_sha256(atlas_path),
        "correlation_threshold": correlation_threshold,
        "bold_normalization": bold_normalization,
        "target_timepoints": target_timepoints,
        "connectivity_timepoints": connectivity_timepoints,
        "negative_edge_policy": negative_edge_policy,
    }


def _configuration_id(configuration):
    serialized = json.dumps(configuration, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def build_graphs(
    subjects,
    atlas_path,
    cache_dir=None,
    correlation_threshold=0.5,
    bold_normalization="per_roi_zscore",
    target_timepoints=196,
    connectivity_timepoints="full",
    negative_edge_policy="signed",
):
    """Build or load one cached PyG graph per participant."""
    coordinates = normalize_coordinates(load_cc200_coordinates(atlas_path))
    configuration = _graph_configuration(
        atlas_path=atlas_path,
        correlation_threshold=correlation_threshold,
        bold_normalization=bold_normalization,
        target_timepoints=target_timepoints,
        connectivity_timepoints=connectivity_timepoints,
        negative_edge_policy=negative_edge_policy,
    )

    configuration_dir = None
    if cache_dir is not None:
        configuration_dir = Path(cache_dir) / _configuration_id(configuration)
        configuration_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "configuration": configuration,
            "atlas_path": str(Path(atlas_path).resolve()),
        }
        (configuration_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    graphs = []
    for subject in subjects:
        graph_path = None
        if configuration_dir is not None:
            graph_path = configuration_dir / f"{subject.subject_id}.pt"

        if graph_path is not None and graph_path.exists():
            graph = torch.load(graph_path, weights_only=False)
        else:
            graph = create_graph(
                roi_time_series=subject.roi_time_series,
                roi_coordinates=coordinates,
                dx_group=subject.dx_group,
                correlation_threshold=correlation_threshold,
                bold_normalization=bold_normalization,
                target_timepoints=target_timepoints,
                connectivity_timepoints=connectivity_timepoints,
                negative_edge_policy=negative_edge_policy,
                subject_id=subject.subject_id,
                site_id=subject.site_id,
                coordinates_are_normalized=True,
            )
            if graph_path is not None:
                temporary_path = graph_path.with_suffix(".tmp")
                torch.save(graph, temporary_path)
                temporary_path.replace(graph_path)

        graphs.append(graph)

    return graphs


def split_graphs(
    graphs,
    train_fraction=0.7,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=42,
    split_by="site",
):
    """Create deterministic participant- or site-grouped graph splits."""
    if not math.isclose(
        train_fraction + validation_fraction + test_fraction,
        1.0,
    ):
        raise ValueError("Train, validation, and test fractions must sum to 1.")

    if split_by == "site":
        groups = [graph.site_id for graph in graphs]
    elif split_by == "subject":
        groups = [graph.subject_id for graph in graphs]
    else:
        raise ValueError("split_by must be 'site' or 'subject'.")

    if len(set(groups)) < 3:
        raise ValueError(f"At least three distinct {split_by} groups are required.")

    indices = list(range(len(graphs)))
    holdout_fraction = validation_fraction + test_fraction
    train_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        test_size=holdout_fraction,
        random_state=seed,
    )
    train_indices, holdout_indices = next(train_splitter.split(indices, groups=groups))

    holdout_groups = [groups[index] for index in holdout_indices]
    validation_share = validation_fraction / holdout_fraction
    holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_share,
        random_state=seed,
    )
    validation_local, test_local = next(
        holdout_splitter.split(holdout_indices, groups=holdout_groups)
    )

    validation_indices = holdout_indices[validation_local]
    test_indices = holdout_indices[test_local]

    return (
        [graphs[index] for index in train_indices],
        [graphs[index] for index in validation_indices],
        [graphs[index] for index in test_indices],
    )


def create_graph_loaders(
    graphs,
    batch_size=16,
    train_fraction=0.7,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=42,
    split_by="site",
    num_workers=2,
    pin_memory=True,
):
    """Create PyG loaders for reproducible train, validation, and test splits."""
    train_graphs, validation_graphs, test_graphs = split_graphs(
        graphs=graphs,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        split_by=split_by,
    )

    return tuple(
        create_graph_loader(
            split,
            batch_size=batch_size,
            shuffle=index == 0,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for index, split in enumerate((train_graphs, validation_graphs, test_graphs))
    )


def split_locked_test_sites(graphs, test_sites):
    """Separate immutable held-out sites from the development participants."""
    test_sites = {str(site) for site in test_sites}
    available_sites = {str(graph.site_id) for graph in graphs}
    missing_sites = test_sites - available_sites
    if missing_sites:
        missing = ", ".join(sorted(missing_sites))
        raise ValueError(f"Locked test sites are missing from the graphs: {missing}")

    development_graphs = [
        graph for graph in graphs if str(graph.site_id) not in test_sites
    ]
    locked_test_graphs = [graph for graph in graphs if str(graph.site_id) in test_sites]
    return development_graphs, locked_test_graphs


def select_development_sites(graphs, test_sites):
    """Return non-test-site graphs without materializing locked test graphs."""
    test_sites = {str(site) for site in test_sites}
    available_sites = {str(graph.site_id) for graph in graphs}
    missing_sites = test_sites - available_sites
    if missing_sites:
        missing = ", ".join(sorted(missing_sites))
        raise ValueError(f"Locked test sites are missing from the graphs: {missing}")
    return [graph for graph in graphs if str(graph.site_id) not in test_sites]


def _subject_label(subject):
    value = getattr(subject, "y", None)
    if value is None:
        value = getattr(subject, "label", None)
    if value is None:
        raise ValueError("Each participant must provide a y or label value.")
    return int(value.item()) if hasattr(value, "item") else int(value)


def _subject_split_ids(
    graphs, train_fraction, validation_fraction, test_fraction, seed
):
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Train, validation, and test fractions must be positive.")

    subject_labels = {}
    for graph in graphs:
        subject_id = str(graph.subject_id)
        label = _subject_label(graph)
        if subject_labels.setdefault(subject_id, label) != label:
            raise ValueError(
                f"Participant {subject_id} has inconsistent diagnosis labels."
            )

    subject_ids = np.asarray(sorted(subject_labels))
    labels = np.asarray([subject_labels[subject_id] for subject_id in subject_ids])
    holdout_fraction = validation_fraction + test_fraction
    train_indices, holdout_indices = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=holdout_fraction,
            random_state=seed,
        ).split(subject_ids, labels)
    )
    holdout_ids = subject_ids[holdout_indices]
    validation_indices, test_indices = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction / holdout_fraction,
            random_state=seed,
        ).split(holdout_ids, labels[holdout_indices])
    )
    return (
        set(subject_ids[train_indices]),
        set(holdout_ids[validation_indices]),
        set(holdout_ids[test_indices]),
    )


def create_subject_split(
    graphs,
    train_fraction=0.7,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=42,
):
    """Split participants by diagnosis while keeping repeated IDs together."""
    split_ids = _subject_split_ids(
        graphs,
        train_fraction,
        validation_fraction,
        test_fraction,
        seed,
    )
    return tuple(
        [graph for graph in graphs if str(graph.subject_id) in ids] for ids in split_ids
    )


def create_subject_development_split(
    graphs,
    train_fraction=0.7,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=42,
):
    """Return only development participants from the shared split allocation."""
    split_ids = _subject_split_ids(
        graphs,
        train_fraction,
        validation_fraction,
        test_fraction,
        seed,
    )
    return tuple(
        [graph for graph in graphs if str(graph.subject_id) in ids]
        for ids in split_ids[:2]
    )


def create_site_validation_folds(graphs, n_splits=5, seed=42):
    """Return reproducible, diagnosis-balanced folds with sites kept intact."""
    labels = []
    for graph in graphs:
        labels.append(_subject_label(graph))
    groups = [str(graph.site_id) for graph in graphs]
    if len(set(groups)) < n_splits:
        raise ValueError(
            f"At least {n_splits} distinct sites are required for {n_splits} folds."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = []
    for train_indices, validation_indices in splitter.split(
        graphs,
        y=labels,
        groups=groups,
    ):
        folds.append(
            (
                [graphs[index] for index in train_indices],
                [graphs[index] for index in validation_indices],
            )
        )
    return folds


def create_graph_loader(
    graphs,
    batch_size=16,
    shuffle=False,
    seed=42,
    num_workers=2,
    pin_memory=True,
):
    """Create one deterministic loader for an already-defined graph split."""
    persistent = num_workers > 0
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )


def summarize_graphs(graphs):
    """Return a compact class and site summary for a graph collection."""
    labels = [int(graph.y.item()) for graph in graphs]
    return {
        "graphs": len(graphs),
        "controls": labels.count(0),
        "asd": labels.count(1),
        "sites": len({graph.site_id for graph in graphs}),
    }

import hashlib
import json
import math
from pathlib import Path

import torch
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.loader import DataLoader

from .data import load_cc200_coordinates
from .graph_builder import create_graph, normalize_coordinates


GRAPH_SCHEMA_VERSION = 1


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as atlas_file:
        for chunk in iter(lambda: atlas_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_configuration(
    atlas_path,
    correlation_threshold,
    bold_normalization,
    target_timepoints,
    negative_edge_policy,
):
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "atlas_sha256": _file_sha256(atlas_path),
        "correlation_threshold": correlation_threshold,
        "bold_normalization": bold_normalization,
        "target_timepoints": target_timepoints,
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
    negative_edge_policy="signed",
):
    """Build or load one cached PyG graph per participant."""
    coordinates = normalize_coordinates(load_cc200_coordinates(atlas_path))
    configuration = _graph_configuration(
        atlas_path=atlas_path,
        correlation_threshold=correlation_threshold,
        bold_normalization=bold_normalization,
        target_timepoints=target_timepoints,
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
        raise ValueError(
            f"At least three distinct {split_by} groups are required."
        )

    indices = list(range(len(graphs)))
    holdout_fraction = validation_fraction + test_fraction
    train_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        test_size=holdout_fraction,
        random_state=seed,
    )
    train_indices, holdout_indices = next(
        train_splitter.split(indices, groups=groups)
    )

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

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_graphs,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def summarize_graphs(graphs):
    """Return a compact class and site summary for a graph collection."""
    labels = [int(graph.y.item()) for graph in graphs]
    return {
        "graphs": len(graphs),
        "controls": labels.count(0),
        "asd": labels.count(1),
        "sites": len({graph.site_id for graph in graphs}),
    }

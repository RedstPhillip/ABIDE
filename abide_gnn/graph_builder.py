import numpy as np
import torch
from torch_geometric.data import Data

DX_GROUP_TO_LABEL = {1: 1, 2: 0}


def normalize_bold(roi_time_series, method="none"):
    time_series = np.asarray(roi_time_series, dtype=np.float32)

    if method == "none":
        return time_series
    if method == "per_roi_zscore":
        mean = time_series.mean(axis=0, keepdims=True)
        std = time_series.std(axis=0, keepdims=True)
        std[std == 0] = 1
        return (time_series - mean) / std

    raise ValueError(f"Unknown BOLD normalization: {method}")


def resize_time_series(roi_time_series, target_timepoints=None):
    if target_timepoints is None:
        return roi_time_series

    timepoints = roi_time_series.shape[0]
    if timepoints >= target_timepoints:
        start = (timepoints - target_timepoints) // 2
        return roi_time_series[start : start + target_timepoints]

    padding = target_timepoints - timepoints
    before = padding // 2
    after = padding - before
    return np.pad(roi_time_series, ((before, after), (0, 0)))


def prepare_bold(
    roi_time_series,
    normalization="none",
    target_timepoints=None,
):
    time_series = normalize_bold(roi_time_series, method=normalization)
    time_series = resize_time_series(time_series, target_timepoints)
    return torch.as_tensor(time_series.T.copy(), dtype=torch.float)


def compute_connectivity(roi_time_series):
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation_matrix = np.corrcoef(roi_time_series, rowvar=False)
    return np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def threshold_connectivity(
    correlation_matrix,
    threshold=0.5,
    negative_edge_policy="signed",
):
    connectivity_matrix = correlation_matrix.copy()
    connectivity_matrix[np.abs(connectivity_matrix) < threshold] = 0
    np.fill_diagonal(connectivity_matrix, 0)

    if negative_edge_policy == "positive_only":
        connectivity_matrix[connectivity_matrix < 0] = 0
    elif negative_edge_policy == "absolute":
        connectivity_matrix = np.abs(connectivity_matrix)
    elif negative_edge_policy != "signed":
        raise ValueError(f"Unknown negative-edge policy: {negative_edge_policy}")

    return connectivity_matrix


def normalize_coordinates(roi_coordinates):
    coordinates = torch.as_tensor(roi_coordinates, dtype=torch.float)
    coordinate_mean = coordinates.mean(dim=0, keepdim=True)
    coordinate_std = coordinates.std(dim=0, unbiased=False, keepdim=True)
    return (coordinates - coordinate_mean) / coordinate_std


def build_edges(connectivity_matrix):
    source_nodes, target_nodes = np.nonzero(connectivity_matrix)
    edge_index = torch.as_tensor(
        np.vstack((source_nodes, target_nodes)),
        dtype=torch.long,
    )
    edge_weight = torch.as_tensor(
        connectivity_matrix[source_nodes, target_nodes],
        dtype=torch.float,
    )
    return edge_index, edge_weight


def create_graph(
    roi_time_series,
    roi_coordinates,
    dx_group=None,
    correlation_threshold=0.5,
    bold_normalization="none",
    target_timepoints=None,
    connectivity_timepoints="full",
    negative_edge_policy="signed",
    subject_id=None,
    site_id=None,
    coordinates_are_normalized=False,
):
    """Create one CC200 participant graph with BOLD stored on its nodes."""
    if connectivity_timepoints not in {"full", "target"}:
        raise ValueError("connectivity_timepoints must be 'full' or 'target'.")
    normalized_time_series = normalize_bold(
        roi_time_series,
        method=bold_normalization,
    )
    bold = prepare_bold(
        normalized_time_series,
        target_timepoints=target_timepoints,
    )
    if coordinates_are_normalized:
        coordinates = torch.as_tensor(roi_coordinates, dtype=torch.float)
    else:
        coordinates = normalize_coordinates(roi_coordinates)

    connectivity_time_series = normalized_time_series
    if connectivity_timepoints == "target":
        connectivity_time_series = resize_time_series(
            normalized_time_series,
            target_timepoints,
        )
    correlation_matrix = compute_connectivity(connectivity_time_series)
    connectivity_matrix = threshold_connectivity(
        correlation_matrix,
        threshold=correlation_threshold,
        negative_edge_policy=negative_edge_policy,
    )
    edge_index, edge_weight = build_edges(connectivity_matrix)

    graph_attributes = {
        "bold": bold,
        "connectivity": torch.as_tensor(
            correlation_matrix.copy(),
            dtype=torch.float,
        ),
        "roi_id": torch.arange(bold.shape[0], dtype=torch.long),
        "pos": coordinates,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": bold.shape[0],
        "original_timepoints": torch.tensor(
            [np.asarray(roi_time_series).shape[0]],
            dtype=torch.long,
        ),
    }

    if dx_group is not None:
        dx_group = int(dx_group)
        if dx_group not in DX_GROUP_TO_LABEL:
            raise ValueError(f"Unexpected DX_GROUP value: {dx_group}")
        graph_attributes["y"] = torch.tensor(
            [DX_GROUP_TO_LABEL[dx_group]],
            dtype=torch.long,
        )
    if subject_id is not None:
        graph_attributes["subject_id"] = str(subject_id)
    if site_id is not None:
        graph_attributes["site_id"] = str(site_id)

    return Data(**graph_attributes)

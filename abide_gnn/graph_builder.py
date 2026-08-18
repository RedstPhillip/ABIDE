import numpy as np
import torch
from torch_geometric.data import Data


DX_GROUP_TO_LABEL = {1: 1, 2: 0}


def prepare_bold(roi_time_series):
    """Convert a [time points, ROIs] array to a [ROIs, time points] tensor."""
    return torch.as_tensor(
        np.asarray(roi_time_series).T.copy(),
        dtype=torch.float,
    )


def compute_connectivity(roi_time_series):
    """Calculate the ROI-by-ROI Pearson correlation matrix."""
    return np.corrcoef(roi_time_series, rowvar=False)


def threshold_connectivity(correlation_matrix, threshold=0.5):
    """Keep signed correlations whose absolute value reaches the threshold."""
    connectivity_matrix = correlation_matrix.copy()
    connectivity_matrix[np.abs(connectivity_matrix) < threshold] = 0
    np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix


def normalize_coordinates(roi_coordinates):
    """Standardize each atlas coordinate axis across ROIs."""
    coordinates = torch.as_tensor(roi_coordinates, dtype=torch.float)
    coordinate_mean = coordinates.mean(dim=0, keepdim=True)
    coordinate_std = coordinates.std(dim=0, unbiased=False, keepdim=True)
    return (coordinates - coordinate_mean) / coordinate_std


def build_edges(connectivity_matrix):
    """Convert a thresholded connectivity matrix to weighted PyG edges."""
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
    dx_group,
    correlation_threshold=0.5,
):
    """Create one training-ready CC200 participant graph."""
    bold = prepare_bold(roi_time_series)
    coordinates = normalize_coordinates(roi_coordinates)

    correlation_matrix = compute_connectivity(roi_time_series)
    connectivity_matrix = threshold_connectivity(
        correlation_matrix,
        threshold=correlation_threshold,
    )
    edge_index, edge_weight = build_edges(connectivity_matrix)

    dx_group = int(dx_group)
    if dx_group not in DX_GROUP_TO_LABEL:
        raise ValueError(f"Unexpected DX_GROUP value: {dx_group}")

    graph_label = torch.tensor(
        [DX_GROUP_TO_LABEL[dx_group]],
        dtype=torch.long,
    )

    return Data(
        bold=bold,
        pos=coordinates,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=graph_label,
        num_nodes=bold.shape[0],
    )

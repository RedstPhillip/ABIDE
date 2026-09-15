import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    APPNP,
    GATv2Conv,
    GraphConv,
    global_max_pool,
    global_mean_pool,
)


class OrthonormalClusterReadout(nn.Module):
    """Official BNT-style soft cluster assignment and readout.

    The BNT paper initializes frozen cluster centers with Gram--Schmidt
    orthonormalization, projects each encoded ROI onto them, squares the
    projections, and applies a softmax before pooling ``P.T @ Z``.  This module
    keeps that deliberately unusual formulation instead of replacing it with a
    generic attention-pooling layer.
    """

    def __init__(
        self,
        feature_dim,
        num_clusters,
        orthogonal=True,
        freeze_centers=True,
        project_assignment=True,
    ):
        super().__init__()
        if num_clusters < 1 or num_clusters > feature_dim:
            raise ValueError(
                "BNT num_clusters must be in [1, feature_dim] so its "
                "orthonormal centers can be constructed."
            )

        centers = torch.empty(num_clusters, feature_dim)
        nn.init.xavier_uniform_(centers)
        if orthogonal:
            centers = torch.linalg.qr(centers.T, mode="reduced").Q.T

        self.project_assignment = project_assignment
        self.cluster_centers = nn.Parameter(
            centers,
            requires_grad=not freeze_centers,
        )

    def forward(self, node_embeddings):
        if self.project_assignment:
            assignment = node_embeddings @ self.cluster_centers.T
            assignment = assignment.pow(2)
            center_norms = self.cluster_centers.norm(dim=-1).clamp_min(1e-12)
            assignment = assignment / center_norms
            return assignment.softmax(dim=-1)

        distances = (
            (node_embeddings.unsqueeze(-2) - self.cluster_centers).pow(2).sum(dim=-1)
        )
        weights = 1.0 / (1.0 + distances)
        return weights / weights.sum(dim=-1, keepdim=True)


class BNTTransformerPoolingEncoder(nn.Module):
    """One official BNT MHSA block, optionally followed by OCRead pooling."""

    def __init__(
        self,
        feature_dim,
        input_nodes,
        output_nodes,
        nhead=4,
        feedforward_dim=1024,
        dropout=0.1,
        cluster_hidden_dim=32,
        pooling=True,
        orthogonal=True,
        freeze_centers=True,
        project_assignment=True,
    ):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.pooling = pooling
        if pooling:
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim * input_nodes, cluster_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(cluster_hidden_dim, cluster_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(cluster_hidden_dim, feature_dim * input_nodes),
            )
            self.readout = OrthonormalClusterReadout(
                feature_dim=feature_dim,
                num_clusters=output_nodes,
                orthogonal=orthogonal,
                freeze_centers=freeze_centers,
                project_assignment=project_assignment,
            )

    def forward(self, node_features):
        node_features = self.transformer(node_features)
        if not self.pooling:
            return node_features, None

        batch_size, node_count, feature_dim = node_features.shape
        encoded = self.encoder(node_features.reshape(batch_size, -1)).reshape(
            batch_size,
            node_count,
            feature_dim,
        )
        assignments = self.readout(encoded)
        return assignments.transpose(1, 2) @ encoded, assignments


class BrainNetworkTransformer(nn.Module):
    """Faithful CC200 adaptation of the NeurIPS 2022 Brain Network Transformer.

    BNT is intentionally a dense ROI-sequence model: the rows of the complete
    connectivity matrix are tokens in fixed CC200 order.  It never consumes the
    thresholded ``edge_index``/``edge_weight`` graph used by the GNN variants.
    """

    def __init__(
        self,
        num_rois=200,
        num_clusters=100,
        nhead=4,
        feedforward_dim=1024,
        dropout=0.1,
        cluster_hidden_dim=32,
        orthogonal=True,
        freeze_centers=True,
        project_assignment=True,
        num_classes=2,
    ):
        super().__init__()
        if num_rois < 1:
            raise ValueError("BNT num_rois must be positive.")
        if nhead < 1 or num_rois % nhead:
            raise ValueError("BNT num_rois must be divisible by bnt_heads.")
        if feedforward_dim < 1 or cluster_hidden_dim < 1:
            raise ValueError("BNT hidden dimensions must be positive.")

        self.num_rois = num_rois
        self.num_clusters = num_clusters

        self.attention_layers = nn.ModuleList(
            (
                BNTTransformerPoolingEncoder(
                    feature_dim=num_rois,
                    input_nodes=num_rois,
                    output_nodes=num_rois,
                    nhead=nhead,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    cluster_hidden_dim=cluster_hidden_dim,
                    pooling=False,
                    orthogonal=orthogonal,
                    freeze_centers=freeze_centers,
                    project_assignment=project_assignment,
                ),
                BNTTransformerPoolingEncoder(
                    feature_dim=num_rois,
                    input_nodes=num_rois,
                    output_nodes=num_clusters,
                    nhead=nhead,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    cluster_hidden_dim=cluster_hidden_dim,
                    pooling=True,
                    orthogonal=orthogonal,
                    freeze_centers=freeze_centers,
                    project_assignment=project_assignment,
                ),
            )
        )
        self.dimension_reduction = nn.Sequential(
            nn.Linear(num_rois, 8),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * num_clusters, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes),
        )

    def _connectivity_tokens(self, graph):
        if not hasattr(graph, "connectivity"):
            raise ValueError(
                "BNT requires graph.connectivity; rebuild graphs with the "
                "current graph schema."
            )
        connectivity = graph.connectivity
        if connectivity.ndim != 2 or connectivity.shape[1] != self.num_rois:
            raise ValueError(
                "BNT requires full square connectivity profiles with "
                f"{self.num_rois} columns in fixed ROI order."
            )

        batch = getattr(graph, "batch", None)
        if batch is None:
            if connectivity.shape[0] != self.num_rois:
                raise ValueError(
                    "An unbatched BNT graph must contain exactly num_rois nodes."
                )
            return connectivity.unsqueeze(0)

        if batch.numel() != connectivity.shape[0] or graph.num_graphs < 1:
            raise ValueError("BNT received inconsistent PyG batch metadata.")
        expected_batch = torch.arange(
            graph.num_graphs,
            device=batch.device,
        ).repeat_interleave(self.num_rois)
        if batch.shape != expected_batch.shape or not torch.equal(
            batch, expected_batch
        ):
            raise ValueError(
                "BNT requires each batch item to contain exactly num_rois "
                "contiguous, fixed-order ROI tokens."
            )
        return connectivity.reshape(graph.num_graphs, self.num_rois, self.num_rois)

    def forward(self, graph):
        node_features = self._connectivity_tokens(graph)
        for layer in self.attention_layers:
            node_features, _ = layer(node_features)
        node_features = self.dimension_reduction(node_features)
        return self.classifier(node_features.reshape(node_features.shape[0], -1))


class TemporalMaxPool1d(nn.Module):
    """Take the maximum across the complete temporal dimension."""

    def forward(self, x):
        return torch.max(x, dim=-1, keepdim=True).values


class ROIEncoder(nn.Module):
    def __init__(self, embedding_dim=32, temporal_pooling="max"):
        super().__init__()
        if temporal_pooling not in {"max", "mean"}:
            raise ValueError("temporal_pooling must be 'max' or 'mean'.")
        temporal_pool = (
            TemporalMaxPool1d()
            if temporal_pooling == "max"
            else nn.AdaptiveAvgPool1d(1)
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            temporal_pool,
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, bold):
        return self.encoder(bold)


class BrainGraphClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        hidden_dim=64,
        coordinate_dim=3,
        dropout=0.0,
        num_classes=2,
        gnn_type="graphconv",
        attention_heads=4,
        appnp_steps=10,
        appnp_alpha=0.1,
        appnp_dropout=0.0,
        bnt_clusters=100,
        bnt_heads=4,
        bnt_feedforward_dim=1024,
        bnt_dropout=0.1,
        bnt_cluster_hidden_dim=32,
        bnt_orthogonal=True,
        bnt_freeze_centers=True,
        bnt_project_assignment=True,
        num_rois=200,
        roi_embedding_dim=0,
        node_feature_mode="temporal",
        use_coordinates=True,
        temporal_pooling="max",
        graph_pooling="mean",
        concat_node_dim=8,
        in_dim=None,
    ):
        super().__init__()

        if in_dim is not None:
            if in_dim < 1:
                raise ValueError("in_dim must be at least 1.")
            num_rois = in_dim
            node_feature_mode = "connectivity"
            use_coordinates = False
        if gnn_type not in {"graphconv", "gatv2", "appnp", "bnt"}:
            raise ValueError(
                "gnn_type must be 'graphconv', 'gatv2', 'appnp', or 'bnt'."
            )
        if attention_heads < 1:
            raise ValueError("attention_heads must be at least 1.")
        if gnn_type == "gatv2" and hidden_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads for GATv2."
            )
        if gnn_type == "appnp" and appnp_steps < 1:
            raise ValueError("appnp_steps must be at least 1.")
        if gnn_type == "appnp" and not 0 < appnp_alpha <= 1:
            raise ValueError("appnp_alpha must be in the interval (0, 1].")
        if gnn_type == "appnp" and not 0 <= appnp_dropout <= 1:
            raise ValueError("appnp_dropout must be in the interval [0, 1].")
        if num_rois < 1:
            raise ValueError("num_rois must be at least 1.")
        if roi_embedding_dim < 0:
            raise ValueError("roi_embedding_dim must be non-negative.")
        if node_feature_mode not in {"temporal", "connectivity"}:
            raise ValueError("node_feature_mode must be 'temporal' or 'connectivity'.")
        if graph_pooling not in {"mean", "max", "mean_max", "roi_concat"}:
            raise ValueError(
                "graph_pooling must be 'mean', 'max', 'mean_max', or 'roi_concat'."
            )
        if concat_node_dim < 1:
            raise ValueError("concat_node_dim must be at least 1.")

        if gnn_type == "bnt":
            if node_feature_mode != "connectivity" or use_coordinates:
                raise ValueError(
                    "BNT uses only full connectivity-profile ROI features; set "
                    "node_feature_mode='connectivity' and use_coordinates=False."
                )
            self.gnn_type = gnn_type
            self.node_feature_mode = node_feature_mode
            self.use_coordinates = use_coordinates
            self.num_rois = num_rois
            self.bnt = BrainNetworkTransformer(
                num_rois=num_rois,
                num_clusters=bnt_clusters,
                nhead=bnt_heads,
                feedforward_dim=bnt_feedforward_dim,
                dropout=bnt_dropout,
                cluster_hidden_dim=bnt_cluster_hidden_dim,
                orthogonal=bnt_orthogonal,
                freeze_centers=bnt_freeze_centers,
                project_assignment=bnt_project_assignment,
                num_classes=num_classes,
            )
            return

        self.gnn_type = gnn_type
        self.attention_heads = attention_heads
        self.appnp_steps = appnp_steps
        self.appnp_alpha = appnp_alpha
        self.appnp_dropout = appnp_dropout
        self.node_feature_mode = node_feature_mode
        self.use_coordinates = use_coordinates
        self.temporal_pooling = temporal_pooling
        self.graph_pooling = graph_pooling
        self.num_rois = num_rois
        self.concat_node_dim = concat_node_dim
        self.roi_encoder = (
            ROIEncoder(
                embedding_dim=embedding_dim,
                temporal_pooling=temporal_pooling,
            )
            if node_feature_mode == "temporal"
            else None
        )
        self.roi_id_embedding = (
            nn.Embedding(num_rois, roi_embedding_dim) if roi_embedding_dim > 0 else None
        )
        node_input_dim = embedding_dim if node_feature_mode == "temporal" else num_rois
        if use_coordinates:
            node_input_dim += coordinate_dim
        node_input_dim += roi_embedding_dim
        if gnn_type == "graphconv":
            self.graph_conv1 = GraphConv(
                node_input_dim,
                hidden_dim,
                aggr="mean",
            )
            self.graph_conv2 = GraphConv(
                hidden_dim,
                hidden_dim,
                aggr="mean",
            )
        elif gnn_type == "gatv2":
            per_head_dim = hidden_dim // attention_heads
            self.graph_conv1 = GATv2Conv(
                node_input_dim,
                per_head_dim,
                heads=attention_heads,
                concat=True,
                edge_dim=1,
                add_self_loops=True,
                fill_value="mean",
            )
            self.graph_conv2 = GATv2Conv(
                hidden_dim,
                per_head_dim,
                heads=attention_heads,
                concat=True,
                edge_dim=1,
                add_self_loops=True,
                fill_value="mean",
            )
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.appnp = APPNP(
                K=appnp_steps,
                alpha=appnp_alpha,
                dropout=appnp_dropout,
                cached=False,
            )
        self.dropout = nn.Dropout(dropout)
        if graph_pooling == "roi_concat":
            self.concat_node_projection = nn.Sequential(
                nn.Linear(hidden_dim, concat_node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            pooled_dim = num_rois * concat_node_dim
            self.concat_readout_norm = nn.LayerNorm(pooled_dim)
        else:
            self.concat_node_projection = None
            self.concat_readout_norm = None
            pooled_dim = hidden_dim * 2 if graph_pooling == "mean_max" else hidden_dim
        self.classifier = nn.Linear(pooled_dim, num_classes)

    def _apply_graph_layer(self, layer, node_features, graph):
        if self.gnn_type == "graphconv":
            return layer(
                node_features,
                graph.edge_index,
                edge_weight=graph.edge_weight,
            )

        edge_attr = graph.edge_weight
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        return layer(
            node_features,
            graph.edge_index,
            edge_attr=edge_attr,
        )

    def forward(self, graph):
        if self.gnn_type == "bnt":
            return self.bnt(graph)

        if self.node_feature_mode == "temporal":
            cnn_input = graph.bold.unsqueeze(1)
            node_features = self.roi_encoder(cnn_input)
        else:
            if not hasattr(graph, "connectivity"):
                raise ValueError(
                    "Connectivity node features are unavailable; rebuild the "
                    "graph cache with the current graph schema."
                )
            node_features = graph.connectivity

        node_feature_parts = [node_features]
        if self.use_coordinates:
            node_feature_parts.append(graph.pos)
        if self.roi_id_embedding is not None:
            num_rois = self.roi_id_embedding.num_embeddings
            roi_indices = (
                torch.arange(graph.num_nodes, device=node_features.device) % num_rois
            )
            node_feature_parts.append(self.roi_id_embedding(roi_indices))
        node_features = torch.cat(node_feature_parts, dim=1)

        if self.gnn_type == "appnp":
            node_features = self.node_mlp(node_features)
            node_features = self.appnp(
                node_features,
                graph.edge_index,
                edge_weight=graph.edge_weight,
            )
        else:
            node_features = self._apply_graph_layer(
                self.graph_conv1,
                node_features,
                graph,
            )
            node_features = F.relu(node_features)
            node_features = self.dropout(node_features)

            node_features = self._apply_graph_layer(
                self.graph_conv2,
                node_features,
                graph,
            )
            node_features = F.relu(node_features)
            node_features = self.dropout(node_features)

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(
                graph.num_nodes,
                dtype=torch.long,
                device=node_features.device,
            )

        if self.graph_pooling == "mean":
            graph_embedding = global_mean_pool(node_features, batch)
        elif self.graph_pooling == "max":
            graph_embedding = global_max_pool(node_features, batch)
        elif self.graph_pooling == "mean_max":
            graph_embedding = torch.cat(
                (
                    global_mean_pool(node_features, batch),
                    global_max_pool(node_features, batch),
                ),
                dim=1,
            )
        else:
            graph_embedding = self._roi_concat_readout(node_features, graph, batch)
        return self.classifier(graph_embedding)

    def _roi_concat_readout(self, node_features, graph, batch):
        """Flatten consistently ordered ROI vectors after a compact projection."""
        if not hasattr(graph, "roi_id"):
            raise ValueError(
                "ROI concatenation requires roi_id metadata; rebuild graph cache."
            )
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        expected_nodes = graph_count * self.num_rois
        if node_features.shape[0] != expected_nodes:
            raise ValueError(
                "ROI concatenation requires exactly num_rois nodes per graph "
                f"({self.num_rois}); received {node_features.shape[0]} nodes "
                f"across {graph_count} graphs."
            )

        expected_ids = torch.arange(self.num_rois, device=node_features.device)
        roi_ids = graph.roi_id.reshape(graph_count, self.num_rois)
        if not torch.equal(roi_ids, expected_ids.expand_as(roi_ids)):
            raise ValueError(
                "ROI concatenation requires every graph to use canonical "
                "0..num_rois-1 ROI order."
            )

        projected = self.concat_node_projection(node_features)
        flattened = projected.reshape(graph_count, self.num_rois * self.concat_node_dim)
        return self.concat_readout_norm(flattened)

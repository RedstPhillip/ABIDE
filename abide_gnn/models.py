import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class ROIEncoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
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
    ):
        super().__init__()
        self.roi_encoder = ROIEncoder(embedding_dim=embedding_dim)
        self.graph_conv1 = GraphConv(
            embedding_dim + coordinate_dim,
            hidden_dim,
            aggr="mean",
        )
        self.graph_conv2 = GraphConv(
            hidden_dim,
            hidden_dim,
            aggr="mean",
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, graph):
        cnn_input = graph.bold.unsqueeze(1)
        roi_embeddings = self.roi_encoder(cnn_input)
        node_features = torch.cat((roi_embeddings, graph.pos), dim=1)

        node_features = self.graph_conv1(
            node_features,
            graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)

        node_features = self.graph_conv2(
            node_features,
            graph.edge_index,
            edge_weight=graph.edge_weight,
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

        graph_embedding = global_mean_pool(node_features, batch)
        return self.classifier(graph_embedding)

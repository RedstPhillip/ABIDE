import unittest

import torch
from torch_geometric.data import Batch, Data

from abide_gnn.models import BrainGraphClassifier, BrainNetworkTransformer


def _connectivity_graph(label, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(200, 12, generator=generator)
    connectivity = torch.corrcoef(features)
    return Data(
        connectivity=connectivity,
        num_nodes=200,
        y=torch.tensor([label], dtype=torch.long),
    )


class BrainNetworkTransformerTests(unittest.TestCase):
    def test_official_cc200_shape_and_orthonormal_centers(self):
        model = BrainNetworkTransformer()

        self.assertEqual(len(model.attention_layers), 2)
        self.assertFalse(model.attention_layers[0].pooling)
        self.assertTrue(model.attention_layers[1].pooling)
        self.assertEqual(model.num_rois, 200)
        self.assertEqual(model.num_clusters, 100)
        centers = model.attention_layers[1].readout.cluster_centers.detach()
        self.assertTrue(torch.allclose(centers @ centers.T, torch.eye(100), atol=1e-5))
        self.assertFalse(
            model.attention_layers[1].readout.cluster_centers.requires_grad
        )

    def test_bnt_forward_and_gradients_use_batched_connectivity_profiles(self):
        batch = Batch.from_data_list(
            [_connectivity_graph(0, 1), _connectivity_graph(1, 2)]
        )
        model = BrainGraphClassifier(
            gnn_type="bnt",
            node_feature_mode="connectivity",
            use_coordinates=False,
            bnt_clusters=4,
            bnt_feedforward_dim=32,
            bnt_cluster_hidden_dim=8,
        )

        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 2))
        logits.sum().backward()
        self.assertIsNotNone(
            model.bnt.attention_layers[0].transformer.self_attn.in_proj_weight.grad
        )
        self.assertIsNotNone(model.bnt.dimension_reduction[0].weight.grad)
        self.assertIsNotNone(model.bnt.classifier[0].weight.grad)

    def test_bnt_rejects_non_connectivity_feature_configuration(self):
        with self.assertRaisesRegex(ValueError, "connectivity-profile"):
            BrainGraphClassifier(gnn_type="bnt")

    def test_bnt_rejects_batches_without_fixed_contiguous_roi_order(self):
        batch = Batch.from_data_list(
            [_connectivity_graph(0, 3), _connectivity_graph(1, 4)]
        )
        batch.batch = batch.batch.roll(1)
        model = BrainNetworkTransformer(
            num_clusters=4,
            feedforward_dim=32,
            cluster_hidden_dim=8,
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            model(batch)

    def test_five_epoch_synthetic_trainability_smoke(self):
        """Exercise optimization for five epochs without touching ABIDE data."""
        torch.manual_seed(17)
        batch = Batch.from_data_list(
            [_connectivity_graph(0, 5), _connectivity_graph(1, 6)]
        )
        model = BrainGraphClassifier(
            gnn_type="bnt",
            node_feature_mode="connectivity",
            use_coordinates=False,
            bnt_clusters=4,
            bnt_feedforward_dim=32,
            bnt_cluster_hidden_dim=8,
            bnt_dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        initial_weight = model.bnt.classifier[0].weight.detach().clone()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertFalse(torch.equal(initial_weight, model.bnt.classifier[0].weight))


if __name__ == "__main__":
    unittest.main()

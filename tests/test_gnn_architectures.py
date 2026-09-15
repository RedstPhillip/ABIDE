import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import APPNP, GATv2Conv, GraphConv

from abide_gnn.graph_builder import threshold_connectivity
from abide_gnn.models import BrainGraphClassifier


def graph(edge_weight):
    return Data(
        bold=torch.randn(4, 12),
        connectivity=torch.randn(4, 4),
        roi_id=torch.arange(4, dtype=torch.long),
        pos=torch.randn(4, 3),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 0],
                [1, 0, 2, 1, 3, 2, 0, 3],
            ],
            dtype=torch.long,
        ),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float),
        num_nodes=4,
    )


class GnnArchitectureTests(unittest.TestCase):
    def test_signed_and_absolute_policies_keep_identical_topology(self):
        correlations = np.array(
            [
                [1.0, 0.8, -0.7],
                [0.8, 1.0, -0.2],
                [-0.7, -0.2, 1.0],
            ]
        )

        signed = threshold_connectivity(
            correlations,
            threshold=0.5,
            negative_edge_policy="signed",
        )
        absolute = threshold_connectivity(
            correlations,
            threshold=0.5,
            negative_edge_policy="absolute",
        )

        np.testing.assert_array_equal(signed != 0, absolute != 0)
        np.testing.assert_allclose(np.abs(signed), absolute)

    def test_graphconv_remains_the_backward_compatible_default(self):
        model = BrainGraphClassifier(embedding_dim=8, hidden_dim=16)

        self.assertIsInstance(model.graph_conv1, GraphConv)
        self.assertEqual(model.gnn_type, "graphconv")
        self.assertIsNone(model.roi_id_embedding)
        self.assertEqual(model.graph_pooling, "mean")

    def test_all_architectures_produce_graph_logits_and_gradients(self):
        batch = Batch.from_data_list(
            [
                graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9]),
                graph([0.7, 0.7, 0.9, 0.9, 0.6, 0.6, 0.8, 0.8]),
            ]
        )

        for gnn_type in ("graphconv", "gatv2", "appnp"):
            with self.subTest(gnn_type=gnn_type):
                model = BrainGraphClassifier(
                    embedding_dim=8,
                    hidden_dim=16,
                    dropout=0.1,
                    gnn_type=gnn_type,
                    attention_heads=4,
                )
                logits = model(batch)
                logits.sum().backward()

                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertIsNotNone(model.roi_encoder.encoder[0].weight.grad)
                trainable_graph_module = (
                    model.node_mlp if gnn_type == "appnp" else model.graph_conv1
                )
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in trainable_graph_module.parameters()
                    )
                )

    def test_optional_roi_identity_embedding_works_for_every_architecture(self):
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])

        for gnn_type in ("graphconv", "gatv2", "appnp"):
            with self.subTest(gnn_type=gnn_type):
                model = BrainGraphClassifier(
                    embedding_dim=8,
                    hidden_dim=16,
                    gnn_type=gnn_type,
                    attention_heads=4,
                    num_rois=4,
                    roi_embedding_dim=3,
                )
                model(sample).sum().backward()

                self.assertIsNotNone(model.roi_id_embedding.weight.grad)

    def test_historical_node_feature_variants_produce_logits(self):
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])
        settings = (
            {"node_feature_mode": "temporal", "use_coordinates": False},
            {
                "node_feature_mode": "temporal",
                "temporal_pooling": "mean",
            },
            {
                "node_feature_mode": "connectivity",
                "use_coordinates": False,
                "num_rois": 4,
            },
        )

        for model_settings in settings:
            with self.subTest(model_settings=model_settings):
                model = BrainGraphClassifier(
                    embedding_dim=8,
                    hidden_dim=16,
                    **model_settings,
                )
                self.assertEqual(tuple(model(sample).shape), (1, 2))

    def test_all_graph_pooling_variants_produce_logits(self):
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])

        for pooling, classifier_features, settings in (
            ("mean", 16, {}),
            ("max", 16, {}),
            ("mean_max", 32, {}),
            ("roi_concat", 32, {"num_rois": 4}),
        ):
            with self.subTest(pooling=pooling):
                model = BrainGraphClassifier(
                    embedding_dim=8,
                    hidden_dim=16,
                    graph_pooling=pooling,
                    **settings,
                )
                self.assertEqual(model.classifier.in_features, classifier_features)
                self.assertEqual(tuple(model(sample).shape), (1, 2))

    def test_roi_concat_rejects_noncanonical_roi_order(self):
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])
        sample.roi_id = torch.tensor([1, 0, 2, 3])
        model = BrainGraphClassifier(
            embedding_dim=8,
            hidden_dim=16,
            num_rois=4,
            graph_pooling="roi_concat",
        )

        with self.assertRaisesRegex(ValueError, "canonical"):
            model(sample)

    def test_legacy_in_dim_selects_connectivity_without_coordinates(self):
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])
        model = BrainGraphClassifier(
            in_dim=4,
            hidden_dim=8,
            dropout=0.0,
        )

        logits = model(sample)

        self.assertEqual(model.node_feature_mode, "connectivity")
        self.assertFalse(model.use_coordinates)
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_gatv2_is_edge_aware_and_preserves_hidden_dimension(self):
        model = BrainGraphClassifier(
            embedding_dim=8,
            hidden_dim=16,
            gnn_type="gatv2",
            attention_heads=4,
        )

        self.assertIsInstance(model.graph_conv1, GATv2Conv)
        self.assertEqual(model.graph_conv1.edge_dim, 1)
        self.assertEqual(model.graph_conv1.heads, 4)
        self.assertEqual(model.graph_conv1.out_channels, 4)
        self.assertEqual(model.classifier.in_features, 16)

        logits = model(graph([0.8, 0.8, -0.7, -0.7, 0.6, 0.6, -0.9, -0.9]))
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_appnp_uses_weighted_uncached_propagation(self):
        model = BrainGraphClassifier(
            embedding_dim=8,
            hidden_dim=16,
            gnn_type="appnp",
            appnp_steps=7,
            appnp_alpha=0.2,
            appnp_dropout=0.1,
        )
        sample = graph([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.9, 0.9])

        self.assertIsInstance(model.appnp, APPNP)
        self.assertEqual(model.appnp.K, 7)
        self.assertEqual(model.appnp.alpha, 0.2)
        self.assertFalse(model.appnp.cached)

        with patch.object(
            model.appnp,
            "forward",
            wraps=model.appnp.forward,
        ) as appnp_forward:
            logits = model(sample)

        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertIs(
            appnp_forward.call_args.kwargs["edge_weight"],
            sample.edge_weight,
        )

    def test_invalid_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gnn_type"):
            BrainGraphClassifier(gnn_type="unknown")

    def test_gatv2_requires_hidden_dimension_divisible_by_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            BrainGraphClassifier(
                hidden_dim=10,
                gnn_type="gatv2",
                attention_heads=4,
            )

    def test_invalid_appnp_hyperparameters_are_rejected(self):
        invalid_settings = (
            {"appnp_steps": 0},
            {"appnp_alpha": 0.0},
            {"appnp_alpha": 1.1},
            {"appnp_dropout": -0.1},
            {"appnp_dropout": 1.1},
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                BrainGraphClassifier(gnn_type="appnp", **settings)

    def test_invalid_feature_and_pooling_settings_are_rejected(self):
        invalid_settings = (
            {"node_feature_mode": "unknown"},
            {"temporal_pooling": "unknown"},
            {"graph_pooling": "unknown"},
            {"num_rois": 0},
            {"roi_embedding_dim": -1},
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                BrainGraphClassifier(**settings)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from unittest.mock import patch

import torch

from train import (
    _create_lr_scheduler,
    _validate_model_graph_compatibility,
    _validate_training_schedule,
    build_config,
    parse_args,
)


class TrainCliTests(unittest.TestCase):
    def test_gnn_configuration_is_added_to_model_config(self):
        with patch.object(
            sys,
            "argv",
            ["train.py", "--gnn-type", "gatv2", "--attention-heads", "8"],
        ):
            model = build_config(parse_args())["model"]

        self.assertEqual(model["gnn_type"], "gatv2")
        self.assertEqual(model["attention_heads"], 8)

    def test_appnp_configuration_is_added_to_model_config(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--gnn-type",
                "appnp",
                "--appnp-steps",
                "12",
                "--appnp-alpha",
                "0.15",
                "--appnp-dropout",
                "0.2",
            ],
        ):
            model = build_config(parse_args())["model"]

        self.assertEqual(model["gnn_type"], "appnp")
        self.assertEqual(model["appnp_steps"], 12)
        self.assertEqual(model["appnp_alpha"], 0.15)
        self.assertEqual(model["appnp_dropout"], 0.2)

    def test_bnt_configuration_uses_connectivity_profiles_and_required_tag(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--gnn-type",
                "bnt",
                "--node-feature-mode",
                "connectivity",
                "--no-coordinates",
                "--bnt-clusters",
                "10",
                "--bnt-feedforward-dim",
                "512",
                "--tags",
                "cc200",
            ],
        ):
            config = build_config(parse_args())

        self.assertEqual(config["model"]["bnt_clusters"], 10)
        self.assertEqual(config["model"]["bnt_heads"], 4)
        self.assertEqual(config["model"]["bnt_feedforward_dim"], 512)
        self.assertTrue(config["model"]["bnt_orthogonal"])
        self.assertTrue(config["model"]["bnt_freeze_centers"])
        self.assertEqual(config["wandb"]["tags"], ["cc200", "brain-transformer"])

    def test_historical_architecture_flags_are_added_to_config(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--node-feature-mode",
                "connectivity",
                "--no-coordinates",
                "--temporal-pooling",
                "mean",
                "--graph-pooling",
                "mean_max",
                "--roi-embedding-dim",
                "8",
                "--num-rois",
                "200",
            ],
        ):
            model = build_config(parse_args())["model"]

        self.assertEqual(model["node_feature_mode"], "connectivity")
        self.assertFalse(model["use_coordinates"])
        self.assertEqual(model["temporal_pooling"], "mean")
        self.assertEqual(model["graph_pooling"], "mean_max")
        self.assertEqual(model["roi_embedding_dim"], 8)
        self.assertEqual(model["num_rois"], 200)

    def test_graph_construction_flags_are_added_to_config(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--correlation-threshold",
                "0.6",
                "--bold-normalization",
                "none",
                "--target-timepoints",
                "180",
                "--connectivity-timepoints",
                "target",
            ],
        ):
            graph = build_config(parse_args())["graph"]

        self.assertEqual(graph["correlation_threshold"], 0.6)
        self.assertEqual(graph["bold_normalization"], "none")
        self.assertEqual(graph["target_timepoints"], 180)
        self.assertEqual(graph["connectivity_timepoints"], "target")

    def test_appnp_rejects_signed_edges(self):
        model = {"gnn_type": "appnp"}

        with self.assertRaisesRegex(ValueError, "nonnegative edge weights"):
            _validate_model_graph_compatibility(
                model,
                {"negative_edge_policy": "signed"},
            )

        for policy in ("absolute", "positive_only"):
            with self.subTest(policy=policy):
                _validate_model_graph_compatibility(
                    model,
                    {"negative_edge_policy": policy},
                )

    def test_negative_edge_policy_is_added_to_graph_config(self):
        with patch.object(
            sys,
            "argv",
            ["train.py", "--negative-edge-policy", "absolute"],
        ):
            config = build_config(parse_args())

        self.assertEqual(config["graph"]["negative_edge_policy"], "absolute")

    def test_stabilized_training_protocol_is_the_default(self):
        with patch.object(sys, "argv", ["train.py"]):
            training = build_config(parse_args())["training"]

        self.assertEqual(training["selection_metric"], "loss")
        self.assertEqual(training["patience"], 40)
        self.assertEqual(training["min_epochs"], 50)
        self.assertEqual(training["lr_scheduler_patience"], 15)
        self.assertEqual(training["lr_scheduler_factor"], 0.5)
        self.assertEqual(training["min_learning_rate"], 1e-5)
        self.assertFalse(training["ema_enabled"])
        self.assertEqual(training["ema_start_epoch"], 200)
        self.assertEqual(training["ema_decay"], 0.99)

    def test_ema_can_be_enabled_from_cli(self):
        with patch.object(sys, "argv", ["train.py", "--ema"]):
            training = build_config(parse_args())["training"]

        self.assertTrue(training["ema_enabled"])

    def test_ema_schedule_is_validated(self):
        with patch.object(sys, "argv", ["train.py"]):
            training = build_config(parse_args())["training"]
        training["ema_start_epoch"] = 0

        with self.assertRaisesRegex(ValueError, "ema_start_epoch"):
            _validate_training_schedule(training)

    def test_scheduler_reduces_lr_once_after_a_plateau(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.001)
        scheduler = _create_lr_scheduler(
            optimizer,
            {
                "lr_scheduler_factor": 0.5,
                "lr_scheduler_patience": 1,
                "min_learning_rate": 1e-5,
            },
        )

        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0)

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0005)


if __name__ == "__main__":
    unittest.main()

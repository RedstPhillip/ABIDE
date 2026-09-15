import json
import tempfile
import unittest
from pathlib import Path

import torch

from abide_gnn.ensemble import (
    _inference_config,
    average_prediction_sets,
    calibrate_prediction_rows,
    calibrate_probability,
    checkpoint_paths_from_manifest,
    ensemble_member_metrics,
    ensemble_metrics,
    validate_checkpoint_ensemble,
)
from abide_gnn.models import BrainGraphClassifier


def prediction(participant_id, label, probability_asd, site="SITE_A"):
    return {
        "participant_id": participant_id,
        "site_id": site,
        "true_label": label,
        "probability_control": 1 - probability_asd,
        "probability_asd": probability_asd,
        "predicted_label": int(probability_asd >= 0.5),
    }


class EnsembleTests(unittest.TestCase):
    def test_legacy_connectivity_config_uses_target_timepoints(self):
        checkpoint = {
            "config": {
                "data": {"n_subjects": None},
                "graph": {"correlation_threshold": 0.5},
                "split": {"strategy": "subject"},
                "loader": {"batch_size": 16},
                "model": {"in_dim": 200, "hidden_dim": 64},
            }
        }

        config = _inference_config(checkpoint)

        self.assertEqual(config["graph"]["connectivity_timepoints"], "target")

    def test_averages_probabilities_after_participant_alignment(self):
        first = [prediction("b", 1, 0.8), prediction("a", 0, 0.4)]
        second = [prediction("a", 0, 0.2), prediction("b", 1, 0.6)]

        rows = average_prediction_sets([first, second])

        self.assertEqual([row["participant_id"] for row in rows], ["a", "b"])
        self.assertAlmostEqual(rows[0]["probability_asd"], 0.3)
        self.assertAlmostEqual(rows[1]["probability_asd"], 0.7)
        self.assertEqual([row["predicted_label"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["member_probabilities_asd"], [0.8, 0.6])

    def test_monotonic_platt_logit_calibration_preserves_raw_probability(self):
        calibration = {
            "type": "platt_logit",
            "intercept": 0.25,
            "coefficient": 0.34,
            "clip_epsilon": 1e-5,
        }
        low = calibrate_probability(0.2, calibration)
        high = calibrate_probability(0.8, calibration)
        self.assertLess(low, high)

        rows = calibrate_prediction_rows(
            [prediction("a", 0, 0.2), prediction("b", 1, 0.8)],
            calibration,
        )
        self.assertEqual(rows[0]["raw_probability_asd"], 0.2)
        self.assertAlmostEqual(rows[0]["probability_control"], 1 - low)

    def test_platt_logit_calibration_rejects_nonmonotonic_coefficient(self):
        with self.assertRaisesRegex(ValueError, "monotonic increasing"):
            calibrate_probability(
                0.5,
                {"type": "platt_logit", "intercept": 0.0, "coefficient": -1.0},
            )

    def test_platt_logit_calibration_is_stable_for_extreme_logits(self):
        self.assertEqual(
            calibrate_probability(
                0.5,
                {"type": "platt_logit", "intercept": -1000, "coefficient": 1},
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "Probability must be finite"):
            calibrate_probability(
                float("nan"),
                {"type": "platt_logit", "intercept": 0, "coefficient": 1},
            )

    def test_rejects_different_participant_sets(self):
        with self.assertRaisesRegex(ValueError, "different participant set"):
            average_prediction_sets(
                [[prediction("a", 0, 0.2)], [prediction("b", 1, 0.8)]]
            )

    def test_ensemble_metrics_include_probability_loss(self):
        rows = average_prediction_sets(
            [[prediction("a", 0, 0.2), prediction("b", 1, 0.8)]]
        )

        metrics = ensemble_metrics(rows)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertGreater(metrics["loss"], 0)

    def test_member_metrics_preserve_checkpoint_identity(self):
        rows = average_prediction_sets(
            [
                [prediction("a", 0, 0.1), prediction("b", 1, 0.9)],
                [prediction("a", 0, 0.8), prediction("b", 1, 0.2)],
            ]
        )

        metrics = ensemble_member_metrics(
            rows,
            [{"seed": 42, "run_id": "good"}, {"seed": 52, "run_id": "bad"}],
        )

        self.assertEqual(metrics[0]["seed"], 42)
        self.assertEqual(metrics[0]["metrics"]["roc_auc"], 1.0)
        self.assertEqual(metrics[1]["metrics"]["roc_auc"], 0.0)

    def test_manifest_resolves_relative_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            manifest = root / "ensemble.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoints": [{"path": "checkpoints/model.pt"}],
                    }
                ),
                encoding="utf-8",
            )

            paths, _ = checkpoint_paths_from_manifest(manifest)

            self.assertEqual(paths, [checkpoint.resolve()])

    def test_checkpoint_validation_rejects_model_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            common = {
                "data": {"n_subjects": None},
                "graph": {"correlation_threshold": 0.5},
                "split": {"strategy": "subject"},
                "loader": {"batch_size": 16},
            }
            paths = []
            for index, hidden_dim in enumerate((8, 16)):
                model_config = {
                    "embedding_dim": 4,
                    "hidden_dim": hidden_dim,
                    "dropout": 0.0,
                    "num_classes": 2,
                }
                model = BrainGraphClassifier(**model_config)
                path = root / f"model-{index}.pt"
                torch.save(
                    {
                        "config": {**common, "model": model_config},
                        "model_state_dict": model.state_dict(),
                    },
                    path,
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "identical"):
                validate_checkpoint_ensemble(paths)


if __name__ == "__main__":
    unittest.main()

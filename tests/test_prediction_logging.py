import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from abide_gnn.tracking import log_wandb_epoch, log_wandb_evaluation
from abide_gnn.training import classification_metrics, export_predictions_csv, predict


class FakeModel(torch.nn.Module):
    def forward(self, batch):
        return batch.logits


class FakeBatch(SimpleNamespace):
    def to(self, device, non_blocking=False):
        return self


class PredictionLoggingTests(unittest.TestCase):
    def test_exports_participant_aligned_probabilities_to_csv(self):
        rows = [
            {
                "participant_id": "subject-1",
                "site_id": "SITE",
                "true_label": 1,
                "predicted_label": 1,
                "probability_control": 0.2,
                "probability_asd": 0.8,
                "member_probabilities_asd": [0.8],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            export_predictions_csv(rows, output)
            with output.open(newline="", encoding="utf-8") as source:
                saved_rows = list(csv.DictReader(source))

        self.assertEqual(saved_rows[0]["participant_id"], "subject-1")
        self.assertEqual(saved_rows[0]["probability_asd"], "0.8")

    def setUp(self):
        self.predictions = [
            {
                "participant_id": "subject-001",
                "site_id": "SITE_A",
                "true_label": 0,
                "probability_control": 0.8,
                "probability_asd": 0.2,
                "predicted_label": 0,
            },
            {
                "participant_id": "subject-002",
                "site_id": "SITE_B",
                "true_label": 1,
                "probability_control": 0.3,
                "probability_asd": 0.7,
                "predicted_label": 1,
            },
        ]

    def test_predict_keeps_participant_and_site_identifiers(self):
        batch = FakeBatch(
            y=torch.tensor([0, 1]),
            subject_id=["subject-001", "subject-002"],
            site_id=["SITE_A", "SITE_B"],
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        )

        rows = predict(FakeModel(), [batch], torch.device("cpu"))

        self.assertEqual(
            [(row["participant_id"], row["site_id"]) for row in rows],
            [("subject-001", "SITE_A"), ("subject-002", "SITE_B")],
        )
        self.assertEqual([row["true_label"] for row in rows], [0, 1])

    def test_metrics_include_ranking_and_calibration_scores(self):
        metrics = classification_metrics(
            labels=[0, 1],
            predictions=[0, 1],
            asd_probabilities=[0.2, 0.7],
        )

        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.065)

    def test_epoch_logging_uses_training_learning_rate_namespace(self):
        run = SimpleNamespace(log=MagicMock())
        train_metrics = {
            "loss": 0.6,
            "accuracy": 0.7,
            "roc_auc": 0.8,
        }
        validation_metrics = {
            "loss": 0.65,
            "accuracy": 0.6,
            "balanced_accuracy": 0.61,
            "roc_auc": 0.7,
            "average_precision": 0.68,
            "brier_score": 0.22,
            "f1_asd": 0.59,
            "sensitivity_asd": 0.58,
            "specificity_control": 0.64,
        }

        log_wandb_epoch(
            run,
            epoch=12,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            learning_rate=0.0005,
        )

        logged_metrics = run.log.call_args.args[0]
        self.assertEqual(logged_metrics["training/learning_rate"], 0.0005)
        self.assertEqual(run.log.call_args.kwargs["step"], 12)

    @patch("abide_gnn.tracking.wandb.plot.roc_curve")
    @patch("abide_gnn.tracking.wandb.plot.confusion_matrix")
    @patch("abide_gnn.tracking.wandb.Table")
    def test_wandb_table_contains_reproducible_prediction_context(
        self,
        table_mock,
        confusion_matrix_mock,
        roc_curve_mock,
    ):
        table = MagicMock()
        table_mock.return_value = table
        run = SimpleNamespace(
            name="test-run",
            summary={},
            log=MagicMock(),
            log_model=MagicMock(),
        )
        metrics = {
            "roc_auc": 1.0,
            "average_precision": 1.0,
            "brier_score": 0.065,
            "confusion_matrix": [[1, 0], [0, 1]],
        }

        log_wandb_evaluation(
            run,
            predictions=self.predictions,
            best_epoch=12,
            validation_metrics=metrics,
            checkpoint_path="best_model.pt",
            training_seed=52,
            split_seed=42,
            split_strategy="subject",
        )

        columns = table_mock.call_args.kwargs["columns"]
        data = table_mock.call_args.kwargs["data"]
        self.assertEqual(columns[0:2], ["participant_id", "site_id"])
        self.assertEqual(
            columns[-4:],
            [
                "best_epoch",
                "training_seed",
                "split_seed",
                "split_strategy",
            ],
        )
        self.assertEqual(data[0][0:2], ["subject-001", "SITE_A"])
        self.assertEqual(data[0][-4:], [12, 52, 42, "subject"])
        self.assertIs(run.log.call_args.args[0]["validation/predictions"], table)
        self.assertEqual(run.summary["best_validation/average_precision"], 1.0)
        self.assertAlmostEqual(run.summary["best_validation/brier_score"], 0.065)

    @patch("abide_gnn.tracking.wandb.plot.roc_curve")
    @patch("abide_gnn.tracking.wandb.plot.confusion_matrix")
    @patch("abide_gnn.tracking.wandb.Table")
    def test_ema_evaluation_uses_a_separate_namespace(
        self,
        table_mock,
        confusion_matrix_mock,
        roc_curve_mock,
    ):
        run = SimpleNamespace(
            name="test-run",
            summary={},
            log=MagicMock(),
            log_model=MagicMock(),
        )
        metrics = {
            "roc_auc": 0.75,
            "confusion_matrix": [[1, 0], [0, 1]],
        }

        log_wandb_evaluation(
            run,
            predictions=self.predictions,
            best_epoch=300,
            validation_metrics=metrics,
            checkpoint_path="ema_model.pt",
            training_seed=42,
            split_seed=42,
            split_strategy="subject",
            candidate="ema",
        )

        logged = run.log.call_args.args[0]
        self.assertIn("ema_validation/predictions", logged)
        self.assertEqual(run.summary["ema_epoch"], 300)
        self.assertEqual(run.summary["ema_validation/roc_auc"], 0.75)
        run.log_model.assert_called_once_with(
            path="ema_model.pt",
            name="test-run-ema-model",
            aliases=["ema"],
        )


if __name__ == "__main__":
    unittest.main()

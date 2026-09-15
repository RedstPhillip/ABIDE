import unittest

import torch

from abide_gnn.training import ExponentialMovingAverage


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_first_update_initializes_from_current_model(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = ExponentialMovingAverage(decay=0.75)

        ema.update(model)
        model.weight.data.zero_()
        ema.copy_to(model)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))

    def test_later_updates_apply_decay(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = ExponentialMovingAverage(decay=0.75)
        ema.update(model)

        model.weight.data.fill_(6.0)
        ema.update(model)
        ema.copy_to(model)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3.0))

    def test_invalid_decay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "EMA decay"):
            ExponentialMovingAverage(decay=1.0)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from abide_gnn.graph_builder import create_graph


class GraphVariantTests(unittest.TestCase):
    def setUp(self):
        self.time_series = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [3.0, 1.0, 0.0],
                [4.0, 4.0, 1.0],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            dtype=np.float32,
        )

    def test_graph_stores_full_connectivity_profiles(self):
        graph = create_graph(
            self.time_series,
            self.coordinates,
            target_timepoints=4,
            connectivity_timepoints="full",
        )

        self.assertEqual(tuple(graph.bold.shape), (3, 4))
        self.assertEqual(tuple(graph.connectivity.shape), (3, 3))
        np.testing.assert_array_equal(graph.roi_id.numpy(), np.arange(3))
        np.testing.assert_allclose(
            graph.connectivity.numpy(),
            np.corrcoef(self.time_series, rowvar=False),
        )

    def test_target_connectivity_reproduces_resized_feature_experiment(self):
        graph = create_graph(
            self.time_series,
            self.coordinates,
            target_timepoints=4,
            connectivity_timepoints="target",
        )
        expected_window = self.time_series[1:5]

        np.testing.assert_allclose(
            graph.connectivity.numpy(),
            np.corrcoef(expected_window, rowvar=False),
        )

    def test_invalid_connectivity_timepoints_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "connectivity_timepoints"):
            create_graph(
                self.time_series,
                self.coordinates,
                connectivity_timepoints="unknown",
            )


if __name__ == "__main__":
    unittest.main()

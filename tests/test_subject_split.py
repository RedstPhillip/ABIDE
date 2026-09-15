import unittest
from types import SimpleNamespace

import torch

from abide_gnn.dataset import create_subject_development_split, create_subject_split


def make_graph(index, site_id, label, subject_id=None):
    return SimpleNamespace(
        subject_id=subject_id or f"subject-{index:03d}",
        site_id=site_id,
        y=torch.tensor(label),
    )


class SubjectSplitTests(unittest.TestCase):
    def setUp(self):
        self.graphs = [
            make_graph(index, f"SITE_{index % 4}", index % 2) for index in range(100)
        ]

    def test_creates_stratified_70_15_15_split(self):
        train, validation, test = create_subject_split(self.graphs, seed=42)

        self.assertEqual((len(train), len(validation), len(test)), (70, 15, 15))
        self.assertEqual(sum(int(g.y.item()) for g in train), 35)
        self.assertIn(sum(int(g.y.item()) for g in validation), (7, 8))
        self.assertIn(sum(int(g.y.item()) for g in test), (7, 8))

    def test_subjects_never_cross_split_boundaries(self):
        duplicate = make_graph(100, "SITE_0", 0, subject_id="subject-000")
        train, validation, test = create_subject_split(
            self.graphs + [duplicate], seed=42
        )

        split_ids = [
            {graph.subject_id for graph in split} for split in (train, validation, test)
        ]
        self.assertTrue(split_ids[0].isdisjoint(split_ids[1]))
        self.assertTrue(split_ids[0].isdisjoint(split_ids[2]))
        self.assertTrue(split_ids[1].isdisjoint(split_ids[2]))
        self.assertEqual(set.union(*split_ids), {g.subject_id for g in self.graphs})

    def test_development_only_split_matches_locked_train_and_validation(self):
        train, validation, _ = create_subject_split(self.graphs, seed=42)
        development_train, development_validation = create_subject_development_split(
            self.graphs,
            seed=42,
        )

        self.assertEqual(
            {graph.subject_id for graph in development_train},
            {graph.subject_id for graph in train},
        )
        self.assertEqual(
            {graph.subject_id for graph in development_validation},
            {graph.subject_id for graph in validation},
        )

    def test_same_seed_reproduces_all_three_sets(self):
        first = create_subject_split(self.graphs, seed=42)
        second = create_subject_split(self.graphs, seed=42)

        first_ids = [[graph.subject_id for graph in split] for split in first]
        second_ids = [[graph.subject_id for graph in split] for split in second]
        self.assertEqual(first_ids, second_ids)

    def test_sites_are_pooled_instead_of_locked_out(self):
        graphs = [
            make_graph(index, "LOCKED" if index < 20 else "OTHER", index % 2)
            for index in range(100)
        ]
        train, validation, test = create_subject_split(graphs, seed=42)

        locked_locations = [
            any(graph.site_id == "LOCKED" for graph in split)
            for split in (train, validation, test)
        ]
        self.assertEqual(locked_locations, [True, True, True])

    def test_rejects_inconsistent_labels_for_one_subject(self):
        graphs = [make_graph(index, "SITE", index % 2) for index in range(20)]
        graphs.append(make_graph(20, "SITE", 1, subject_id="subject-000"))

        with self.assertRaisesRegex(ValueError, "inconsistent diagnosis"):
            create_subject_split(graphs, seed=42)


if __name__ == "__main__":
    unittest.main()

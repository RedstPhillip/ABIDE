import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from abide_gnn.data import index_abide_subjects, materialize_abide_subjects

QUALITY_COLUMNS = [
    "qc_rater_1",
    "qc_anat_rater_2",
    "qc_func_rater_2",
    "qc_anat_rater_3",
    "qc_func_rater_3",
]


class DataIndexTests(unittest.TestCase):
    def write_phenotypic(self, root):
        dataset = root / "ABIDE_pcp"
        dataset.mkdir()
        path = dataset / "Phenotypic_V1_0b_preprocessed1.csv"
        columns = ["FILE_ID", "DX_GROUP", "SITE_ID", *QUALITY_COLUMNS]
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    "FILE_ID": "SITE_0001",
                    "DX_GROUP": 1,
                    "SITE_ID": "SITE",
                    **{column: "OK" for column in QUALITY_COLUMNS},
                }
            )
            writer.writerow(
                {
                    "FILE_ID": "SITE_0002",
                    "DX_GROUP": 2,
                    "SITE_ID": "SITE",
                    **{column: "OK" for column in QUALITY_COLUMNS},
                }
            )
            rejected = {column: "OK" for column in QUALITY_COLUMNS}
            rejected["qc_rater_1"] = "fail"
            writer.writerow(
                {
                    "FILE_ID": "SITE_0003",
                    "DX_GROUP": 1,
                    "SITE_ID": "SITE",
                    **rejected,
                }
            )
        return dataset

    def test_index_returns_paths_without_opening_roi_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset = self.write_phenotypic(root)

            subjects = index_abide_subjects(root)

            self.assertEqual(
                [item.subject_id for item in subjects], ["SITE_0001", "SITE_0002"]
            )
            self.assertIsInstance(subjects[0].roi_path, Path)
            self.assertFalse(subjects[0].roi_path.exists())
            self.assertEqual(
                subjects[0].roi_path.parent, dataset / "cpac" / "filt_noglobal"
            )

    def test_rejects_nonpositive_subject_limit_before_accessing_data(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                index_abide_subjects(temporary_dir, n_subjects=0)

    def test_materializes_only_supplied_index_rows(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.write_phenotypic(root)
            subjects = index_abide_subjects(root)
            subjects[0].roi_path.parent.mkdir(parents=True)
            np.savetxt(subjects[0].roi_path, np.ones((3, 2)))

            materialized = materialize_abide_subjects(subjects[:1])

            self.assertEqual(len(materialized), 1)
            self.assertEqual(materialized[0].roi_time_series.shape, (3, 2))
            self.assertFalse(subjects[1].roi_path.exists())


if __name__ == "__main__":
    unittest.main()

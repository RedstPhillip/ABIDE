from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords


CC200_ATLAS_URL = (
    "https://raw.githubusercontent.com/preprocessed-connectomes-project/"
    "abide/master/preprocessing/resources/abide_rois/CC200.nii.gz"
)


@dataclass(frozen=True)
class ABIDESubject:
    roi_time_series: np.ndarray
    dx_group: int | None
    subject_id: str
    site_id: str | None = None


def _load_roi_time_series(value):
    if isinstance(value, (str, Path)):
        return np.loadtxt(value)
    return np.asarray(value)


def load_abide_subjects(data_dir, n_subjects=None):
    """Load CC200 time series and core metadata for ABIDE participants."""
    abide = datasets.fetch_abide_pcp(
        data_dir=data_dir,
        n_subjects=n_subjects,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        quality_checked=True,
        verbose=0,
    )

    subjects = []
    for index, roi_data in enumerate(abide.rois_cc200):
        participant = abide.phenotypic.iloc[index]
        subjects.append(
            ABIDESubject(
                roi_time_series=_load_roi_time_series(roi_data),
                dx_group=int(participant["DX_GROUP"]),
                subject_id=str(participant["FILE_ID"]),
                site_id=str(participant["SITE_ID"]),
            )
        )

    return subjects


def fetch_cc200_atlas(data_dir):
    """Return the local CC200 atlas path, downloading it when necessary."""
    atlas_path = Path(data_dir) / "atlases" / "cc200_roi_atlas.nii.gz"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)

    if not atlas_path.exists():
        temporary_path = atlas_path.with_suffix(".tmp")
        urlretrieve(CC200_ATLAS_URL, temporary_path)
        temporary_path.replace(atlas_path)

    return atlas_path


def load_cc200_coordinates(atlas_path):
    """Load coordinates in verified CC200 label order."""
    coordinates, labels = find_parcellation_cut_coords(
        atlas_path,
        background_label=0,
        return_label_names=True,
    )
    labels = np.asarray(labels, dtype=int)

    if not np.array_equal(labels, np.arange(1, 201)):
        raise ValueError("The atlas does not contain CC200 labels 1 through 200.")

    return coordinates

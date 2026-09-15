import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from nilearn.plotting import find_parcellation_cut_coords

CC200_ATLAS_URL = (
    "https://raw.githubusercontent.com/preprocessed-connectomes-project/"
    "abide/master/preprocessing/resources/abide_rois/CC200.nii.gz"
)
ABIDE_PCP_BASE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_PHENOTYPIC_FILE = "Phenotypic_V1_0b_preprocessed1.csv"


@dataclass(frozen=True)
class ABIDESubject:
    roi_time_series: np.ndarray
    dx_group: int | None
    subject_id: str
    site_id: str | None = None


@dataclass(frozen=True)
class ABIDESubjectIndex:
    """Participant metadata and a deferred ROI path for split-safe loading."""

    roi_path: str | Path
    dx_group: int
    subject_id: str
    site_id: str | None = None

    @property
    def label(self):
        if self.dx_group == 1:
            return 1
        if self.dx_group == 2:
            return 0
        raise ValueError(f"Unexpected DX_GROUP value: {self.dx_group}")


def _load_roi_time_series(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            url = (
                f"{ABIDE_PCP_BASE_URL}/Outputs/cpac/filt_noglobal/rois_cc200/"
                f"{path.name}"
            )
            urlretrieve(url, temporary_path)
            temporary_path.replace(path)
        return np.loadtxt(path)
    return np.asarray(value)


def index_abide_subjects(data_dir, n_subjects=None):
    """Read only ABIDE metadata and ROI paths, without loading time series."""
    if n_subjects is not None and n_subjects < 1:
        raise ValueError("n_subjects must be a positive integer or None.")
    dataset_dir = Path(data_dir) / "ABIDE_pcp"
    phenotypic_path = dataset_dir / ABIDE_PHENOTYPIC_FILE
    if not phenotypic_path.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = phenotypic_path.with_suffix(".tmp")
        urlretrieve(
            f"{ABIDE_PCP_BASE_URL}/{ABIDE_PHENOTYPIC_FILE}",
            temporary_path,
        )
        temporary_path.replace(phenotypic_path)

    roi_dir = dataset_dir / "cpac" / "filt_noglobal"
    quality_filters = {
        "qc_rater_1": {"OK"},
        "qc_anat_rater_2": {"OK", "maybe"},
        "qc_func_rater_2": {"OK", "maybe"},
        "qc_anat_rater_3": {"OK"},
        "qc_func_rater_3": {"OK"},
    }
    subjects = []
    with phenotypic_path.open(newline="", encoding="utf-8-sig") as input_file:
        participants = csv.DictReader(input_file)
        for participant in participants:
            if participant["FILE_ID"] == "no_filename":
                continue
            if any(
                participant[column] not in accepted
                for column, accepted in quality_filters.items()
            ):
                continue
            file_id = participant["FILE_ID"]
            roi_path = roi_dir / f"{file_id}_rois_cc200.1D"
            subjects.append(
                ABIDESubjectIndex(
                    roi_path=roi_path,
                    dx_group=int(participant["DX_GROUP"]),
                    subject_id=file_id,
                    site_id=participant["SITE_ID"],
                )
            )
            if n_subjects is not None and len(subjects) >= n_subjects:
                break

    if not subjects:
        raise ValueError("No quality-checked ABIDE participants were indexed.")
    return subjects


def materialize_abide_subjects(subject_index):
    """Load ROI time series only for the previously selected participants."""
    return [
        ABIDESubject(
            roi_time_series=_load_roi_time_series(subject.roi_path),
            dx_group=subject.dx_group,
            subject_id=subject.subject_id,
            site_id=subject.site_id,
        )
        for subject in subject_index
    ]


def load_abide_subjects(data_dir, n_subjects=None):
    """Load CC200 time series and core metadata for ABIDE participants."""
    return materialize_abide_subjects(
        index_abide_subjects(data_dir, n_subjects=n_subjects)
    )


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

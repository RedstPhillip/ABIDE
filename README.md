# ABIDE Brain Network Classification

Resting-state fMRI classification using Python, PyTorch, PyTorch Geometric,
and Nilearn. The final model is a three-seed Brain Network Transformer (BNT)
ensemble with probability calibration fitted on training-only out-of-fold
predictions.

## Model and results

Each participant is represented by a Pearson connectivity matrix from 200 CC200
regions. BNT processes its connectivity profiles in fixed atlas order. The
ensemble averages seeds 42, 52, and 62, applies Platt calibration, and uses a
fixed decision threshold of 0.5.

| Partition | Participants | ROC-AUC | Average precision | Balanced accuracy | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 131 | 0.7485 | 0.6765 | 0.6842 | 0.2043 |
| Final test | 131 | 0.7021 | 0.6651 | 0.6445 | 0.2238 |

These are the recorded results from the completed experiment, not a new run.
The [model manifest](configs/final_bnt_calibrated/ensemble.json) contains exact
checkpoint versions, SHA-256 hashes, calibration coefficients, and validation
metrics. [Aggregate test results](configs/final_bnt_calibrated/test_metrics.json)
record the final evaluation from September 4, 2026.

The data are quality-checked ABIDE I / PCP time series processed with CPAC,
band-pass filtering, and no global signal regression. The subject split contains
609 training, 131 validation, and 131 test participants. Development loading
selects participants from metadata before opening their ROI files.

This is a research model, not a diagnostic tool. Sites overlap across the subject
split, so the results do not establish generalization to unseen sites. Earlier
classical experiments also used this test partition; it is not a pristine
project-wide holdout. An earlier eager-loading defect was corrected and affected
development results were rerun before the final model was selected. The recorded
test evaluation was performed after selection and was not used for further tuning.

## Setup

Use Python 3.11 or newer and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

`requirements.txt` records the pinned research environment. On a GPU machine,
install a CUDA-enabled PyTorch build before installing this package.

## Predict

Download the three checkpoints from the W&B artifacts named in the manifest.
This requires access to the referenced artifacts and authentication where needed.

```bash
python download_manifest_checkpoints.py configs/final_bnt_calibrated/ensemble.json
python -m abide_gnn.inference --ensemble configs/final_bnt_calibrated/ensemble.json --roi-file path/to/participant_rois_cc200.1D --device cpu
```

Input files contain one row per time point and 200 numeric columns in CC200 atlas
order. Output includes calibrated class probabilities, the raw ensemble
probability, and the predicted research class. Checkpoints load sequentially.

## Training and evaluation

Train a BNT member without online logging:

```bash
python train.py --gnn-type bnt --node-feature-mode connectivity --no-coordinates --split-strategy subject --seed 42 --training-seed 42 --lr 0.0001 --weight-decay 0.0001 --epochs 200 --min-epochs 1 --patience 0 --lr-scheduler-patience 1000 --no-wandb
```

Training downloads missing development data on first use. Repeat with training
seeds 52 and 62 for three members. The saved manifest is the source of truth for
the completed ensemble; retraining does not replace its frozen calibration.
`python train.py --help` lists supported training options, including retained
graph-model variants for checkpoint compatibility and comparisons.

`evaluate_ensemble.py` evaluates a supplied manifest and requires
`--confirm-test-evaluation`. The completed model's recorded test results are
already included above; another test evaluation is unnecessary for code checks.

## Code layout

- `abide_gnn/`: data, models, training, calibration, and inference.
- `configs/`: frozen ensemble manifest and aggregate results.
- `notebooks/data_exploration.ipynb`: cohort and time-series exploration.
- `tests/`: regression tests for the retained implementation.

Data, downloaded weights, and generated outputs are ignored by Git.

To open the notebook, install `pip install -e ".[notebooks]"`, register the kernel
with `python -m ipykernel install --user --name abide`, and run `jupyter lab`.
For development checks:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

## References

- [ABIDE Preprocessed Connectomes Project](https://preprocessed-connectomes-project.org/abide/)
- [Brain Network Transformer](https://github.com/Wayfear/BrainNetworkTransformer)

The BNT implementation adapts the released architecture to 200 CC200 tokens and
100 readout clusters, using QR initialization for orthonormal cluster centers.

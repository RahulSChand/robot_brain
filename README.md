# BCI Motor Imagery Classifier (b_rob)

Brain-Computer Interface classifier using the [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) dataset. Classifies 5 motor imagery actions from EEG + TD-NIRS brain recordings:

**Labels:** Left Fist, Right Fist, Both Fists, Tongue Tapping, Relax

## Dataset

- **Source:** `hf download KernelCo/robot_control --repo-type dataset`
- **1395 .npz files**, 17 subjects, 31 sessions, 279 samples per class (balanced)
- Each file contains:
  - `feature_eeg`: (7499, 6) — 6 EEG channels at 500 Hz, 15 seconds
  - `feature_moments`: (72, 40, 3, 2, 3) — TD-NIRS at 4.76 Hz (time × modules × SDS × wavelength × moment)
  - `label`: dict with `label`, `subject_id`, `session_id`, `duration`

### Data Quality Findings

| Issue | Detail |
|---|---|
| EEG channels | `AFF6, AFp2, AFp1, AFF5, FCz, CPz` — 4 are frontal (useless for motor imagery), only 2 near motor cortex (midline only, can't distinguish left/right) |
| NIRS NaN | 87% of files have some NaN, but it's **all-or-nothing** per channel (dead sensor = 100% NaN, alive sensor = 0% NaN). 91.3% of channels are perfectly clean. |
| NIRS dead modules | Modules 34-39 (back of head) worst affected. Motor cortex modules (10-23) work 80-94% of the time |
| EEG accuracy | ~19-20% (random chance) — electrode placement can't capture motor imagery signals |
| NIRS accuracy | 40-50% — this is where the useful signal lives |

## Files

### Scripts (run in this order)

| File | What It Does | Run Where |
|---|---|---|
| `analyze_data.py` | Scans all 1395 files for NaN, shape issues, label distribution, per-module quality, value ranges. **Run this first** to understand the data. | Local or remote |
| `train_bci_classifier.py` | **V1 classifier.** Hand-crafted features (mean, std, band powers) from NIRS+EEG → LDA/RF/SVM. Stratified + subject-wise CV. | Local or remote |
| `train_bci_v2.py` | **V2 classifier.** Richer features (slope, time windows, laterality, skew, kurtosis), per-subject z-norm, feature selection, XGBoost, hyperparameter tuning, stacking ensemble. | Local or remote |
| `train_clean_nirs.py` | **Dead channel experiment.** Tests 6 strategies for handling dead NIRS channels (drop unreliable, SDS1-only, motor-only, mean imputation). Compares all with RF/LDA/SVM. | Local or remote |
| `train_transformer.py` | **Deep learning.** Trains a small CNN and Transformer on raw NIRS time-series (72 timesteps × 80 channels). Stratified + subject-wise CV. Needs GPU. | Remote (CUDA) |

### Other Files

| File | Purpose |
|---|---|
| `main.py` | Placeholder (unused) |
| `notes.txt` | Quick reference commands (realsync, dataset download) |
| `.realsync` | Config for syncing local workspace to remote GPU machine |
| `r_1.txt` | Saved terminal output from remote transformer training run |
| `bci_model_lda.pkl`, `bci_model_rf.pkl`, `bci_scaler.pkl`, `bci_label_encoder.pkl` | Saved V1 sklearn models |
| `nirs_transformer.pt` | Saved transformer model (on remote) |
| `pyproject.toml`, `uv.lock` | Python dependencies (managed by `uv`) |

## Results Summary

| Method | Stratified 5-fold CV | Subject-wise CV |
|---|---|---|
| V1: Hand-crafted features (RF, 288 feats) | 44.0% | 33.8% |
| V2: Richer features + XGBoost (1551 feats) | ~45-50% | ~35-40% |
| Clean NIRS: SVM (660 feats, NaN→0) | **50.3%** | **40.2%** |
| CNN on raw NIRS (156K params) | 41.8% | 34.4% |
| Transformer on raw NIRS (114K params) | 41.6% | 31.6% |

**Key finding:** Hand-crafted features + SVM outperformed deep learning. The dataset is too small (1395 samples) for transformers to learn better representations than domain-specific feature engineering. The transformer's best class was Tongue Tapping (60% F1), worst was Both Fists/Right Fist (~33% F1).

## NIRS Preprocessing Pipeline

All scripts follow the same pipeline (from dataset hints):

1. **Short channel regression** — regress SDS=0 (scalp signal) out of SDS=1,2 (brain signal) to remove systemic noise
2. **Baseline correction** — subtract mean of first 14 samples (~3s rest period)
3. **NaN handling** — dead channels (all-or-nothing) replaced with 0
4. **Feature extraction** (classical) or **raw input** (deep learning)

## Setup

```bash
# Local (Mac)
uv sync
uv run python analyze_data.py
uv run python train_clean_nirs.py

# Remote (GPU)
# Dataset at: /mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control
pip install torch numpy scipy scikit-learn xgboost
python train_transformer.py
```

## Why EEG Doesn't Work

The 6 EEG channels (AFF6, AFp2, AFp1, AFF5, FCz, CPz) follow the 10-20 naming system:
- 4 channels are on the **forehead** (anterior-frontal) — irrelevant for motor imagery
- 2 channels are **midline** (FCz, CPz) — near motor cortex but can't distinguish left vs right hemisphere
- The critical positions **C3** (left motor cortex) and **C4** (right motor cortex) are **missing from the headset**


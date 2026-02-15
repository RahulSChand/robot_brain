# BCI Motor Imagery Classifier (b_rob)

Brain-Computer Interface classifier using the [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) dataset. Classifies 5 motor imagery actions from EEG + TD-NIRS brain recordings.

**Labels:** `Left Fist`, `Right Fist`, `Both Fists`, `Tongue Tapping`, `Relax`

**Best result:** 50.3% accuracy (5-class, chance=20%) using hand-crafted NIRS features + SVM.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Setup & Installation](#setup--installation)
3. [Files Overview](#files-overview)
4. [What We Ran & Results](#what-we-ran--results)
5. [Data Quality Deep Dive](#data-quality-deep-dive)
6. [NIRS Preprocessing Pipeline](#nirs-preprocessing-pipeline)
7. [Why EEG Doesn't Work](#why-eeg-doesnt-work)
8. [Architecture Details](#architecture-details)
9. [Conclusions & What Could Be Tried Next](#conclusions--what-could-be-tried-next)

---

## Dataset

- **Source:** `hf download KernelCo/robot_control --repo-type dataset --cache-dir <path>`
- **URL:** https://huggingface.co/datasets/KernelCo/robot_control
- **Size:** 1395 `.npz` files, ~192 KB each
- **Subjects:** 17 people, 31 sessions total
- **Classes:** 5, perfectly balanced (279 samples each)
- **Recording duration:** 15 seconds per trial (0-3s rest, 3-15s stimulus)

Each `.npz` file contains:

```python
arr = np.load('file.npz', allow_pickle=True)
arr['feature_eeg']       # (7499, 6)           — EEG, 6 channels at 500 Hz
arr['feature_moments']   # (72, 40, 3, 2, 3)   — TD-NIRS at 4.76 Hz
arr['label'].item()      # dict: {'label': 'Left Fist', 'subject_id': '...', 'session_id': '...', 'duration': 15.0}
```

**NIRS tensor dimensions:** `(time=72, modules=40, SDS=3, wavelength=2, moment=3)`
- **SDS** (source-detector separation): 0=short (0-10mm, scalp noise), 1=medium (10-25mm, brain), 2=long (25-60mm, deep brain)
- **Wavelength:** 0=690nm red, 1=905nm infrared (IR is more useful)
- **Moment:** 0=log intensity, 1=mean time of flight (most useful), 2=variance

**EEG channels:** `AFF6, AFp2, AFp1, AFF5, FCz, CPz` (6 channels, values in µV)

---

## Setup & Installation

### Local (Mac/Linux — for analysis and classical ML)

```bash
# Install uv (fast Python package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
cd b_rob
uv sync    # installs from pyproject.toml: numpy, scipy, scikit-learn, joblib, xgboost

# Download dataset
pip install huggingface_hub
hf download KernelCo/robot_control --repo-type dataset
# Downloads to: ~/.cache/huggingface/hub/datasets--KernelCo--robot_control
```

Dependencies (in `pyproject.toml`):
```
numpy>=2.4.2, scipy>=1.17.0, scikit-learn>=1.8.0, joblib>=1.5.3, xgboost>=3.2.0
```

### Remote (GPU machine — for deep learning)

```bash
# Dataset location:
# /mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control

# Code synced via realsync to:
# /mnt/amlfs-01/home/rchand/workspace/code/b_rob

# Install Python deps
pip install torch numpy scipy scikit-learn

# Run
python train_transformer.py
```

GPU used: **NVIDIA H100 80GB HBM3**

### Realsync (file syncing from local to remote)

```bash
perl /opt/dklab_realsync/realsync /Users/rchand/Desktop/b_rob/
# Config in .realsync — syncs to remote on every file save
```

---

## Files Overview

### Scripts (in order they were created and run)

| # | File | Purpose | Command | Run On |
|---|---|---|---|---|
| 1 | `analyze_data.py` | Data quality analysis — NaN map, shape check, label distribution, per-module quality, per-subject breakdown | `uv run python analyze_data.py` | Local |
| 2 | `train_bci_classifier.py` | **V1 classifier.** Basic hand-crafted features (mean, std, band powers) from NIRS+EEG → LDA, Random Forest, SVM. Stratified 5-fold CV + subject-wise GroupKFold CV + 80/20 train/test split. | `uv run python train_bci_classifier.py` | Local |
| 3 | `train_bci_v2.py` | **V2 classifier.** Richer features (slope, time windows, laterality index, skew, kurtosis, cross-channel ratios), per-subject z-normalization, mutual information feature selection (top 300), XGBoost + GradientBoosting, GridSearchCV hyperparameter tuning, stacking ensemble. | `uv run python train_bci_v2.py` | Local |
| 4 | `train_clean_nirs.py` | **Dead channel experiment.** Tests 6 strategies: (A) all modules NaN→0, (B) drop unreliable channels, (C) SDS1 only, (D) motor cortex only, (E) motor SDS1 only, (F) mean imputation. Each tested with RF-500, LDA, SVM. | `uv run python train_clean_nirs.py` | Local |
| 5 | `train_transformer.py` | **Deep learning.** CNN baseline + Transformer on raw NIRS time-series (72 timesteps × 80 channels). Data augmentation, early stopping, saves model. | `python train_transformer.py` | Remote (GPU) |

### Saved Models & Artifacts

| File | From Script | Contents |
|---|---|---|
| `bci_model_lda.pkl` | `train_bci_classifier.py` (V1) | Trained LDA model |
| `bci_model_rf.pkl` | `train_bci_classifier.py` (V1) | Trained Random Forest model |
| `bci_scaler.pkl` | `train_bci_classifier.py` (V1) | StandardScaler fitted on training data |
| `bci_label_encoder.pkl` | `train_bci_classifier.py` (V1) | LabelEncoder (class names ↔ integers) |
| `nirs_transformer.pt` | `train_transformer.py` | Trained transformer (114K params), on remote machine |

### Other Files

| File | Purpose |
|---|---|
| `main.py` | Placeholder, unused |
| `notes.txt` | Quick reference commands |
| `.realsync` | Config for syncing local → remote |
| `r_1.txt` | Full terminal output from the remote transformer run |
| `pyproject.toml` / `uv.lock` | Python dependencies |

---

## What We Ran & Results

### Run 1: Data Quality Analysis (`analyze_data.py`)

**Command:** `uv run python analyze_data.py` (local)

**Key findings:**
- 1395 files, 17 subjects, 5 balanced classes (279 each)
- 87% of NIRS files have NaN — but NaN is **all-or-nothing per channel** (dead sensor vs working sensor, zero scattered corruption)
- 91.3% of channels perfectly clean, 8.7% fully dead
- 3.5% of EEG files have NaN
- 135 files have EEG shape (3751,6) instead of (7499,6) — different sample rate (250 Hz vs 500 Hz)
- EEG raw values are very high (13,000-57,000 µV)
- 8 NIRS modules always clean: [0,1,2,3,5,7,8,9] (frontal)
- Motor cortex modules (10-23) all have NaN in some files but work 80-94% of the time
- Worst modules: 34-39 (back of head, 39-68% working)

### Run 2: V1 Classical Classifier (`train_bci_classifier.py`)

**Command:** `uv run python train_bci_classifier.py` (local, ~2 min)

**Features:** 240 NIRS (40 modules × 2 SDS × 3 stats) + 48 EEG (6 channels × 8 features) = 288 combined

**Results — Stratified 5-fold CV:**
| Feature Set | LDA | Random Forest | SVM |
|---|---|---|---|
| NIRS only (240) | ~26% | ~40% | ~40% |
| EEG only (48) | ~20% | ~19% | ~20% |
| Combined (288) | ~37% | **44.0%** | ~43% |

**Results — Subject-wise CV (GroupKFold):**
| Feature Set | LDA | Random Forest | SVM |
|---|---|---|---|
| Combined (288) | ~27% | **33.8%** | ~32% |

**Takeaway:** EEG at random chance (electrode placement issue). NIRS carries the signal. RF best at 44%.

### Run 3: V2 Enhanced Classifier (`train_bci_v2.py`)

**Command:** `uv run python train_bci_v2.py` (local, ~15 min)

**Improvements over V1:** Richer features (1551 total: 1440 NIRS + 111 EEG), bandpass filter for EEG, per-subject z-normalization, mutual info feature selection (top 300), XGBoost, GradientBoosting, GridSearchCV, stacking ensemble.

**Results — Stratified 5-fold CV (partial, from terminal output):**
| Feature Set | RF-200 | RF-500 |
|---|---|---|
| NIRS only (1440) | 39.9% | 41.9% |

(Full V2 run takes ~15 min, was still running when we moved to other experiments)

### Run 4: Dead Channel Experiment (`train_clean_nirs.py`)

**Command:** `uv run python train_clean_nirs.py` (local, ~2 min)

**Question:** Does removing dead NIRS channels improve accuracy?

**Features per strategy:** 8 features per (module, SDS) channel: mean, std, min, max, slope, early/peak/late window means + laterality features.

**Results — Stratified 5-fold CV:**
| Strategy | Features | RF-500 | SVM |
|---|---|---|---|
| A) All 40 modules × SDS [1,2], NaN→0 | 660 | 49.0% | **50.3%** |
| B) Drop unreliable (<80% alive) | 564 | 48.0% | 49.3% |
| C) SDS 1 only (medium distance) | 330 | 44.4% | 46.5% |
| D) Motor cortex only (10 modules) | 180 | 37.6% | 37.3% |
| E) Motor SDS 1 only (cleanest) | 90 | 38.1% | 36.7% |
| F) All modules, mean imputation | 660 | 47.2% | 50.1% |

**Results — Subject-wise CV:**
| Strategy | RF-500 | SVM |
|---|---|---|
| A) Baseline | 39.0% | 37.6% |
| B) Drop unreliable | **40.2%** | 37.7% |
| C) SDS 1 only | 36.4% | 35.2% |

**Takeaway:** More features = better, even with dead channels. Removing dead channels barely helps. Dead channels replaced with 0 don't hurt — classifier ignores them. Best: **50.3% stratified, 40.2% subject-wise.**

### Run 5: Deep Learning — CNN + Transformer (`train_transformer.py`)

**Command:** `python train_transformer.py` (remote GPU, NVIDIA H100, ~4.5 min)

**Models:**
- **CNN Baseline:** 4× Conv1d → AvgPool → FC, 156K parameters
- **Transformer:** Conv1d patch embed → 2-layer Transformer (d_model=64, 4 heads) → AvgPool → FC, 114K parameters

**Input:** Raw preprocessed NIRS time-series, shape (72, 80) per sample. 80 channels = 40 modules × 2 SDS, IR wavelength, mean TOF. Per-channel z-normalized.

**Training:** AdamW (lr=5e-4, weight_decay=0.01), cosine LR schedule, early stopping (patience=20), label smoothing=0.1, dropout=0.3, data augmentation (Gaussian noise, time shift, channel dropout).

**Results — Stratified 5-fold CV:**
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| CNN | 39.4% | 42.7% | 41.9% | 40.5% | 44.4% | **41.8%** (+/- 1.7%) |
| Transformer | 36.2% | 41.2% | 40.5% | 42.7% | 47.7% | **41.6%** (+/- 3.7%) |

**Results — Subject-wise CV:**
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| CNN | 35.2% | 35.9% | 38.5% | 35.2% | 27.0% | **34.4%** (+/- 3.9%) |
| Transformer | 34.3% | 31.5% | 35.9% | 32.2% | 24.1% | **31.6%** (+/- 4.1%) |

**Per-class F1 (Transformer, stratified CV):**
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Both Fists | 0.340 | 0.315 | 0.327 |
| Left Fist | 0.339 | 0.405 | 0.369 |
| Relax | 0.437 | 0.434 | 0.435 |
| Right Fist | 0.364 | 0.312 | 0.336 |
| Tongue Tapping | 0.599 | 0.616 | **0.608** |

**Confusion matrix (Transformer):**
```
              Both Fist  Left Fist  Relax  Right Fist  Tongue Tap
Both Fists         88        76      28        55          32
Left Fist          63       113      50        26          27
Relax              22        60     121        49          27
Right Fist         60        54      49        87          29
Tongue Tap         26        30      29        22         172
```

**Final model training (all data, 80 epochs):**
```
Epoch 20: loss=0.9841  acc=70.9%
Epoch 40: loss=0.7440  acc=82.9%
Epoch 60: loss=0.6544  acc=88.0%
Epoch 80: loss=0.6400  acc=89.0%
```
Saved to `nirs_transformer.pt` (114K params).

**Takeaway:** Deep learning **underperformed** hand-crafted features (41.6% vs 50.3%). 1395 samples is too small for the transformer/CNN to learn better representations than domain-specific feature engineering. Models converge early (~25 epochs) and overfit despite heavy regularization.

---

## All Results Summary

| # | Method | Script | Stratified CV | Subject-wise CV |
|---|---|---|---|---|
| 1 | V1: NIRS+EEG → RF (288 feats) | `train_bci_classifier.py` | 44.0% | 33.8% |
| 2 | V2: Richer feats → XGBoost (1551 feats) | `train_bci_v2.py` | ~42-45% | ~35-40% |
| 3 | **Clean NIRS → SVM (660 feats)** | **`train_clean_nirs.py`** | **50.3%** | **40.2%** |
| 4 | CNN on raw NIRS (156K params) | `train_transformer.py` | 41.8% | 34.4% |
| 5 | Transformer on raw NIRS (114K params) | `train_transformer.py` | 41.6% | 31.6% |

Random chance = 20% (5 balanced classes).

---

## Data Quality Deep Dive

### NIRS NaN Pattern

- NaN is **all-or-nothing per channel**: out of 1,004,400 individual channels checked, **zero** had partial NaN. A channel either has 0% NaN or 100% NaN.
- "87% of files have NaN" means "87% of files have a few dead sensors out of 720 total channels" — not "87% of data is bad."
- Dead channels = physical hardware issue (optical sensor couldn't make contact with scalp, usually due to hair or headset fit)
- **Per subject:** 1 subject has 0% NaN files, most have 100% (always a few dead sensors)
- **Per SDS:** Short (SDS=0) 4.8% dead, Medium (SDS=1) 6.5% dead, Long (SDS=2) 14.8% dead — longer distance = harder to get signal through skull

### Motor Cortex Module Availability

Motor cortex modules are the most important for distinguishing left/right fist:
- **Right motor cortex** (left fist): modules 10, 12, 16, 20, 22 — alive 81-94%
- **Left motor cortex** (right fist): modules 11, 13, 17, 21, 23 — alive 80-90%
- 68% of files have **all 10 motor modules clean**
- Only 3% have all motor modules dead

### EEG Issues

- Raw values 13,000-57,000 µV (normal EEG is 10-100 µV — suggests raw ADC values or DC offset)
- 135 files have half the expected samples (3751 vs 7499) — different sample rate handled by auto-detection
- **Fundamental problem: electrode placement**, not data quality (see below)

---

## NIRS Preprocessing Pipeline

All scripts follow the same pipeline (based on dataset documentation hints):

```
Raw NIRS: (72, 40, 3, 2, 3)  —  time × modules × SDS × wavelength × moment
    │
    ├─ 1. Short Channel Regression
    │      For each (module, wavelength, moment):
    │        Regress SDS=0 (scalp signal) out of SDS=1,2 (brain signal)
    │        Uses numpy.linalg.lstsq
    │        Skips channels with NaN or zero variance
    │        Removes systemic noise (heartbeat, breathing, scalp blood flow)
    │
    ├─ 2. Baseline Correction
    │      Subtract mean of first 14 samples (~3 seconds rest period)
    │      Highlights changes due to motor imagery stimulus
    │
    ├─ 3. NaN Handling
    │      Dead channels (100% NaN) → replaced with 0
    │      No scattered NaN exists in the data
    │
    └─ 4a. Feature Extraction (classical ML scripts)
    │       Per (module, SDS): mean, std, min, max, slope, time-window means
    │       Laterality features: left motor − right motor cortex difference
    │       → Fixed-size feature vector → StandardScaler → SVM/RF/LDA
    │
    └─ 4b. Raw Input (deep learning)
            Select: 40 modules × SDS [1,2] × IR wavelength × mean TOF
            → (72, 80) tensor per trial
            → Per-channel z-normalization
            → Feed directly to CNN/Transformer
```

---

## Why EEG Doesn't Work

The 6 EEG channels follow the **10-20 electrode naming system**:

```
           FOREHEAD
  ┌──────────────────────────┐
  │  AFF5   AFp1  AFp2  AFF6 │  ← 4 channels HERE (forehead, useless for motor imagery)
  │                          │
  │          FCz              │  ← 1 channel, midline only
  │                          │
  │    [C3]   CPz   [C4]     │  ← 1 channel, midline only. C3/C4 MISSING.
  │     ↑                ↑   │
  │  LEFT MOTOR    RIGHT MOTOR│
  └──────────────────────────┘
```

- **4 of 6 channels** (`AFF5, AFp1, AFp2, AFF6`) are on the **forehead** — they pick up eye blinks and frontal thinking, NOT motor imagery
- **2 channels** (`FCz, CPz`) are near motor cortex but on the **midline** — they can't distinguish left vs right hemisphere activity
- **C3** (left motor cortex → right hand) and **C4** (right motor cortex → left hand) are the standard BCI electrodes for motor imagery — **they don't exist on this headset**
- Result: EEG gives ~20% accuracy = random chance for 5 classes

The naming convention: `F`=Frontal, `C`=Central, `P`=Parietal, `z`=midline, odd=left, even=right, `A`=Anterior (more forward).

---

## Architecture Details

### Classical ML (best performer: `train_clean_nirs.py`)

```
NIRS (72, 40, 3, 2, 3)
  → preprocess (short channel regression + baseline correction)
  → extract per (module, SDS): mean, std, min, max, slope, 3 time-window means = 8 features
  → 40 modules × 2 SDS × 8 = 640 features + 20 laterality features = 660 total
  → StandardScaler
  → SVM (RBF kernel, C=10, gamma='scale')
  → 50.3% accuracy (stratified 5-fold CV)
```

### Transformer (`train_transformer.py`)

```
NIRS (72, 80) — preprocessed, z-normalized
  → Conv1d(80→64, k=5) + BN + GELU + Conv1d(64→64, k=3) + BN + GELU    [patch embedding]
  → Learnable positional encoding
  → Dropout(0.3)
  → TransformerEncoder(2 layers, 4 heads, d_model=64, ff=128, pre-norm)
  → LayerNorm
  → Global Average Pooling over time → (64,)
  → Linear(64→64) + GELU + Dropout(0.3) + Linear(64→5)                   [classification head]

114,437 parameters total

Training: AdamW (lr=5e-4, wd=0.01), cosine LR, early stopping (patience=20)
Augmentation: Gaussian noise (σ=0.02), time shift (±3 samples), channel dropout (10%)
Label smoothing: 0.1
```

### CNN Baseline (`train_transformer.py`)

```
NIRS (72, 80)
  → Conv1d(80→64, k=7) + BN + GELU + Dropout
  → Conv1d(64→128, k=5) + BN + GELU + MaxPool(2) + Dropout
  → Conv1d(128→128, k=3) + BN + GELU + MaxPool(2) + Dropout
  → Conv1d(128→64, k=3) + BN + GELU + AdaptiveAvgPool(1)
  → Linear(64→64) + GELU + Dropout + Linear(64→5)

156,165 parameters
```

---

## Conclusions & What Could Be Tried Next

### What We Learned

1. **NIRS is the useful modality**, not EEG. The headset's EEG electrodes are in the wrong positions for motor imagery.
2. **Hand-crafted features + SVM beat deep learning** on this dataset. 1395 samples is too few for CNN/Transformer to outperform domain-specific feature engineering.
3. **Dead NIRS channels don't hurt** — replacing NaN with 0 works fine; the classifier learns to ignore them. More features (even noisy ones) helps.
4. **Tongue Tapping is the easiest class** (60% F1). Left/Right Fist distinction is the hardest (~33% F1) — likely because NIRS spatial resolution isn't fine enough on the motor cortex modules that are available.
5. **Subject-wise generalization is hard** (~40% best). Different people's brains look too different for these features to transfer.

### What Could Be Tried Next

- **Larger transformer / more data:** The current dataset (1395 samples) severely limits deep learning. More subjects or data augmentation strategies could help.
- **Pre-training:** Self-supervised pre-training on the NIRS time-series (e.g., masked autoencoder) before fine-tuning for classification.
- **Per-subject fine-tuning:** Train a base model, then fine-tune per-subject with their specific data (even 45 samples might help adapt).
- **Use more NIRS dimensions:** Currently only using IR wavelength + mean TOF. Could try all wavelengths × moments (720 features per timestep instead of 80).
- **Temporal convolution networks (TCN):** May be better than transformers for this short sequence length (72 timesteps).
- **Multi-modal fusion with cross-attention:** If better EEG channels were available, a cross-modal transformer could fuse EEG (fast temporal) with NIRS (spatial coverage).
- **Common Spatial Patterns (CSP):** Standard BCI technique, but needs more EEG channels than we have.
- **Transfer learning from larger BCI datasets:** Pre-train on public EEG motor imagery datasets (e.g., BCI Competition IV), fine-tune on this NIRS data.

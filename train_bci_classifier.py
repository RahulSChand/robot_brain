"""
BCI Classifier Training Script
==============================
Trains a classifier on the KernelCo/robot_control dataset (EEG + TD-NIRS)
to predict motor imagery actions:
  - Left Fist, Right Fist, Both Fists, Tongue Tapping, Relax

Data quality findings (from analyze_data.py):
  - 1395 files, 17 subjects, 31 sessions, 5 classes (279 each)
  - 87% of files have NaN in NIRS (32/40 modules affected)
  - 3.5% of files have NaN in EEG
  - 135 files have EEG shape (3751,6) instead of (7499,6) — different sample rate
  - Only NIRS modules 0,1,2,3,5,7,8,9 are fully NaN-free across all files
  - Motor cortex modules (10-23) ALL have NaN in some files

Approach:
  - Use ALL 40 NIRS modules but skip NaN values (nanmean/nanstd/nanmin)
  - Handle variable EEG lengths (detect actual sample rate per file)
  - Robust feature extraction that tolerates missing data

Dataset: https://huggingface.co/datasets/KernelCo/robot_control
Download: hf download KernelCo/robot_control --repo-type dataset
"""

import numpy as np
from numpy.linalg import lstsq
from glob import glob
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
import sys
import time
import joblib
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress nanmean warnings

# =============================================
# CONFIG
# =============================================

# Remote path (uncomment when running on remote machine):
# DEFAULT_CACHE_DIR = '/mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control'

# Local path (Mac):
DEFAULT_CACHE_DIR = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--KernelCo--robot_control'
)

# NIRS modules to use for features.
# Motor cortex modules (from Kernel Flow headset map):
#   Right motor cortex (left fist):  10, 12, 16, 20, 22
#   Left motor cortex (right fist):  11, 13, 17, 21, 23
# These have NaN in many files, but we use nanmean/nanstd to handle it.
# We also include clean frontal modules (0-9) for additional signal.
ALL_MODULES = list(range(40))

# NIRS: 4.76 Hz, so 3 seconds baseline ≈ 14 samples
NIRS_BASELINE_SAMPLES = 14

# EEG: some files are 500Hz (7499 samples), others 250Hz (3751 samples)
# We detect per file and adjust baseline accordingly.
DURATION_SECONDS = 15
BASELINE_SECONDS = 3


# =============================================
# FIND DATA FILES
# =============================================

def find_data_files(data_dir):
    for pattern in [
        os.path.join(data_dir, 'snapshots', '*', 'data', '*.npz'),
        os.path.join(data_dir, '*.npz'),
        os.path.join(data_dir, '**', '*.npz'),
    ]:
        files = sorted(glob(pattern, recursive=True))
        if files:
            return files
    return []


# =============================================
# PREPROCESSING FUNCTIONS
# =============================================

def preprocess_nirs(nirs_data):
    """
    NIRS preprocessing:
    1. Short channel regression (only where data is valid)
    2. Baseline correction

    Input:  (72, 40, 3, 2, 3) = (time, modules, SDS, wavelength, moment)
    Output: same shape, cleaned. NaN values are left as NaN
            (handled by nanmean/nanstd in feature extraction).
    """
    cleaned = nirs_data.copy().astype(np.float64)

    # Step 1: Short channel regression (where possible)
    for mod in range(40):
        for wl in range(2):
            for mom in range(3):
                short = cleaned[:, mod, 0, wl, mom]

                # Skip if short channel has NaN or no variance
                if np.any(~np.isfinite(short)) or np.nanstd(short) < 1e-12:
                    continue

                short_2d = short.reshape(-1, 1)

                for sds in [1, 2]:
                    brain = cleaned[:, mod, sds, wl, mom]

                    # Skip if brain channel has NaN or no variance
                    if np.any(~np.isfinite(brain)) or np.nanstd(brain) < 1e-12:
                        continue

                    try:
                        fit, _, _, _ = lstsq(short_2d, brain, rcond=None)
                        result = brain - (short_2d @ fit).flatten()
                        if np.all(np.isfinite(result)):
                            cleaned[:, mod, sds, wl, mom] = result
                    except np.linalg.LinAlgError:
                        pass

    # Step 2: Baseline correction (using nanmean for robustness)
    baseline = np.nanmean(cleaned[:NIRS_BASELINE_SAMPLES], axis=0)
    cleaned = cleaned - baseline

    return cleaned


def extract_nirs_features(nirs_cleaned):
    """
    Extract features from preprocessed NIRS.

    For each module, extracts from medium+long channels,
    IR wavelength (idx 1), mean time of flight (idx 1):
      - nanmean, nanmin, nanstd of post-stimulus signal

    Uses ALL 40 modules (NaN-robust via nanmean/nanstd).
    40 modules x 2 SDS x 3 stats = 240 NIRS features
    (NaN channels produce 0 features — still informative as "no signal")
    """
    features = []
    stimulus_data = nirs_cleaned[NIRS_BASELINE_SAMPLES:]

    for mod in ALL_MODULES:
        for sds in [1, 2]:  # medium and long channels
            signal = stimulus_data[:, mod, sds, 1, 1]  # IR, mean TOF

            # Use nan-safe functions — if all NaN, result is 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_val = np.nanmean(signal) if np.any(np.isfinite(signal)) else 0.0
                min_val = np.nanmin(signal) if np.any(np.isfinite(signal)) else 0.0
                std_val = np.nanstd(signal) if np.any(np.isfinite(signal)) else 0.0

            features.append(mean_val)
            features.append(min_val)
            features.append(std_val)

    return np.array(features)


def extract_eeg_features(eeg_data):
    """
    Extract features from EEG data.

    Handles variable sample rates:
      - 7499 samples → 500 Hz
      - 3751 samples → 250 Hz

    For each of the 6 channels, computes:
      - Basic stats: mean, std, min, max (4)
      - Frequency band powers: delta, theta, alpha, beta (4)

    6 channels x 8 features = 48 EEG features
    """
    n_samples = eeg_data.shape[0]

    # Detect sample rate from number of samples
    # 15 seconds: 500Hz → 7499 samples, 250Hz → 3751 samples
    if n_samples > 5000:
        fs = 500
    else:
        fs = 250

    baseline_samples = int(BASELINE_SECONDS * fs)

    # Replace NaN with 0 for EEG (only 3.5% of files affected)
    eeg_clean = np.nan_to_num(eeg_data, nan=0.0).astype(np.float64)

    # Baseline correction
    baseline = eeg_clean[:baseline_samples].mean(axis=0)
    eeg_stimulus = eeg_clean[baseline_samples:] - baseline

    features = []
    nperseg = min(256, len(eeg_stimulus) // 2)  # adjust for shorter recordings
    if nperseg < 64:
        nperseg = 64

    for ch in range(eeg_stimulus.shape[1]):
        signal = eeg_stimulus[:, ch]

        # Basic statistics
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))

        # Frequency band powers
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        features.append(psd[(freqs >= 0.5) & (freqs < 4)].mean())    # delta
        features.append(psd[(freqs >= 4) & (freqs < 8)].mean())      # theta
        features.append(psd[(freqs >= 8) & (freqs < 13)].mean())     # alpha
        features.append(psd[(freqs >= 13) & (freqs < 30)].mean())    # beta

    return np.array(features)


# =============================================
# MAIN
# =============================================

if __name__ == '__main__':

    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DATA_DIR', DEFAULT_CACHE_DIR)

    print("=" * 60)
    print("BCI Classifier Training")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")

    data_files = find_data_files(data_dir)
    if not data_files:
        print(f"\nERROR: No .npz files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(data_files)} .npz files")

    # =============================================
    # LOAD & PREPROCESS ALL DATA
    # =============================================

    print("\nLoading and preprocessing data...")
    t_start = time.time()

    X_nirs_list = []
    X_eeg_list = []
    X_combined_list = []
    y_labels = []
    subjects = []
    sessions = []
    n_skipped = 0

    for i, f in enumerate(data_files):
        arr = np.load(f, allow_pickle=True)

        label_info = arr['label'].item()
        label = label_info['label']
        subject_id = label_info.get('subject_id', 'unknown')
        session_id = label_info.get('session_id', 'unknown')

        # Preprocess NIRS
        nirs_clean = preprocess_nirs(arr['feature_moments'])
        nirs_feats = extract_nirs_features(nirs_clean)

        # Extract EEG features
        eeg_feats = extract_eeg_features(arr['feature_eeg'])

        # Check for any remaining bad values
        combined = np.concatenate([nirs_feats, eeg_feats])
        if not np.all(np.isfinite(combined)):
            # Replace any remaining NaN/Inf with 0
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            nirs_feats = np.nan_to_num(nirs_feats, nan=0.0, posinf=0.0, neginf=0.0)
            eeg_feats = np.nan_to_num(eeg_feats, nan=0.0, posinf=0.0, neginf=0.0)

        X_nirs_list.append(nirs_feats)
        X_eeg_list.append(eeg_feats)
        X_combined_list.append(combined)
        y_labels.append(label)
        subjects.append(subject_id)
        sessions.append(session_id)

        if (i + 1) % 200 == 0 or (i + 1) == len(data_files):
            elapsed = time.time() - t_start
            print(f"  Processed {i+1}/{len(data_files)} files  ({elapsed:.1f}s)")

    X_nirs = np.array(X_nirs_list)
    X_eeg = np.array(X_eeg_list)
    X_all = np.array(X_combined_list)
    subjects_arr = np.array(subjects)

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Quick sanity check
    assert np.all(np.isfinite(X_all)), "Feature matrix still has NaN/Inf!"

    print(f"\nDataset Summary:")
    print(f"  Total samples:        {X_all.shape[0]}")
    print(f"  NIRS features/sample: {X_nirs.shape[1]}")
    print(f"  EEG features/sample:  {X_eeg.shape[1]}")
    print(f"  Combined features:    {X_all.shape[1]}")
    print(f"  Classes:              {list(le.classes_)}")
    print(f"  Unique subjects:      {len(set(subjects))}")
    print(f"  Unique sessions:      {len(set(sessions))}")
    print(f"  Label distribution:")
    for cls in le.classes_:
        count = y_labels.count(cls)
        print(f"    {cls}: {count} samples")

    # =============================================
    # METHOD 1: Stratified 5-Fold Cross-Validation
    # =============================================

    print("\n" + "=" * 60)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    feature_sets = {
        'NIRS only': X_nirs,
        'EEG only': X_eeg,
        'Combined (NIRS+EEG)': X_all,
    }

    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='auto'),
    }

    best_score = 0
    best_combo = None

    for feat_name, X_feat in feature_sets.items():
        print(f"\n--- {feat_name} ({X_feat.shape[1]} features) ---")
        for clf_name, clf_template in classifiers.items():
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf_template),
            ])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X_feat, y, cv=skf, scoring='accuracy')
            print(f"  {clf_name:20s}: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_combo = (feat_name, clf_name)

    print(f"\nBest: {best_combo[1]} with {best_combo[0]} -> {best_score*100:.1f}%")

    # =============================================
    # METHOD 2: Subject-Wise Cross-Validation
    # =============================================

    print("\n" + "=" * 60)
    print("SUBJECT-WISE CROSS-VALIDATION (GroupKFold)")
    print("  Ensures no subject appears in both train and test")
    print("=" * 60)

    unique_subjects = np.unique(subjects_arr)
    n_folds = min(5, len(unique_subjects))
    subject_encoder = LabelEncoder()
    groups = subject_encoder.fit_transform(subjects)

    for feat_name, X_feat in feature_sets.items():
        print(f"\n--- {feat_name} ({X_feat.shape[1]} features) ---")
        for clf_name, clf_template in classifiers.items():
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf_template),
            ])
            gkf = GroupKFold(n_splits=n_folds)
            scores = cross_val_score(pipe, X_feat, y, cv=gkf, groups=groups, scoring='accuracy')
            print(f"  {clf_name:20s}: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    # =============================================
    # METHOD 3: Train/Test Split (80/20)
    # =============================================

    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT (80/20, stratified)")
    print("=" * 60)

    X = X_all
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nResults (Combined NIRS+EEG features):")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_s, y_train)
    print(f"  LDA           - Train: {lda.score(X_train_s, y_train)*100:.1f}%  Test: {lda.score(X_test_s, y_test)*100:.1f}%")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    print(f"  Random Forest - Train: {rf.score(X_train_s, y_train)*100:.1f}%  Test: {rf.score(X_test_s, y_test)*100:.1f}%")

    svm = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm.fit(X_train_s, y_train)
    print(f"  SVM (RBF)     - Train: {svm.score(X_train_s, y_train)*100:.1f}%  Test: {svm.score(X_test_s, y_test)*100:.1f}%")

    # =============================================
    # DETAILED RESULTS
    # =============================================

    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT (LDA)")
    print("=" * 60)

    y_pred = lda.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    header = "  " + "  ".join([f"{c[:6]:>6s}" for c in le.classes_])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:6d}" for v in row])
        print(f"  {row_str}  <- {le.classes_[i]}")

    # =============================================
    # SAVE MODELS
    # =============================================

    save_dir = os.path.dirname(os.path.abspath(__file__))

    joblib.dump(lda, os.path.join(save_dir, 'bci_model_lda.pkl'))
    joblib.dump(rf, os.path.join(save_dir, 'bci_model_rf.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'bci_scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'bci_label_encoder.pkl'))

    print(f"\nModels saved to {save_dir}/")
    print("  - bci_model_lda.pkl  - bci_model_rf.pkl")
    print("  - bci_scaler.pkl     - bci_label_encoder.pkl")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

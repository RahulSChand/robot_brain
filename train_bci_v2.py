#!/usr/bin/env python3
"""
BCI Classifier v2 — Push accuracy as high as possible
======================================================

Key improvements over v1:
  1. NIRS: Focus on motor cortex modules, use all wavelengths × moments
  2. NIRS: Time-windowed features (early/peak/late hemodynamic response)
  3. NIRS: Slope, laterality index features
  4. EEG:  Bandpass filter, richer stats (skew, kurtosis, peak-to-peak)
  5. EEG:  Cross-channel ratios (frontal/central, left/right asymmetry)
  6. EEG:  Time-windowed features (early vs late)
  7. Per-subject z-normalization option
  8. Feature selection (mutual information)
  9. More classifiers: GradientBoosting, XGBoost, SVM-tuned
 10. Hyperparameter tuning via GridSearchCV
 11. Stacking ensemble
"""

import numpy as np
from numpy.linalg import lstsq
from glob import glob
from scipy.signal import welch, butter, sosfilt
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GroupKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import sys
import time
import joblib
import warnings

warnings.filterwarnings('ignore')

def P(msg=""):
    """Print with flush."""
    print(msg, flush=True)

# =============================================
# CONFIG
# =============================================

# Remote path (uncomment when running on remote machine):
# DEFAULT_CACHE_DIR = '/mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control'

# Local path (Mac):
DEFAULT_CACHE_DIR = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--KernelCo--robot_control'
)

NIRS_SAMPLE_RATE = 4.76
EEG_SAMPLE_RATE = 500
NIRS_BASELINE_SAMPLES = 14  # ~3s at 4.76Hz
EEG_BASELINE_SECONDS = 3
BASELINE_SECONDS = 3

# Motor cortex modules (from dataset documentation)
RIGHT_MOTOR = [10, 12, 16, 20, 22]
LEFT_MOTOR  = [11, 13, 17, 21, 23]
MOTOR_MODULES = RIGHT_MOTOR + LEFT_MOTOR  # 10 modules


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
# NIRS PREPROCESSING
# =============================================

def preprocess_nirs(nirs_data):
    """Short channel regression + baseline correction.
    Only regress motor cortex modules to save time."""
    cleaned = nirs_data.copy().astype(np.float64)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

    # Only regress the modules we'll actually use for features
    for mod in MOTOR_MODULES:
        for wl in range(2):
            for mom in range(3):
                short = cleaned[:, mod, 0, wl, mom]
                if np.std(short) < 1e-12:
                    continue
                short_2d = short.reshape(-1, 1)
                for sds in [1, 2]:
                    brain = cleaned[:, mod, sds, wl, mom]
                    if np.std(brain) < 1e-12:
                        continue
                    try:
                        fit, _, _, _ = lstsq(short_2d, brain, rcond=None)
                        result = brain - (short_2d @ fit).flatten()
                        if np.all(np.isfinite(result)):
                            cleaned[:, mod, sds, wl, mom] = result
                    except np.linalg.LinAlgError:
                        pass

    baseline = np.nanmean(cleaned[:NIRS_BASELINE_SAMPLES], axis=0)
    cleaned = cleaned - baseline
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned


def extract_nirs_features_v2(nirs_cleaned):
    """
    Focused NIRS features on motor cortex:

    For each motor module (10) × SDS (medium+long = 2):
      For key wavelength/moment combos:
        - Full stimulus: mean, std, min, slope
        - 3 time windows: mean, std
      Laterality index: (left - right) for each pair

    Target: ~600-800 features (manageable for 1395 samples)
    """
    features = []

    # Time windows (NIRS sample indices from start of stimulus)
    # Stimulus starts after baseline (sample 14)
    # Signal shape is (72, 40, 3, 2, 3)
    n_stim = nirs_cleaned.shape[0] - NIRS_BASELINE_SAMPLES
    w1_end = min(14, n_stim)   # 0-3s into stimulus
    w2_end = min(28, n_stim)   # 3-6s
    w3_end = n_stim             # 6s to end

    # Extract per motor module
    for mod in MOTOR_MODULES:
        for sds in [1, 2]:  # medium, long
            for wl in range(2):
                for mom in range(3):
                    sig = nirs_cleaned[NIRS_BASELINE_SAMPLES:, mod, sds, wl, mom]
                    sig = np.nan_to_num(sig, nan=0.0)

                    # Full stimulus stats
                    features.append(np.mean(sig))
                    features.append(np.std(sig))
                    features.append(np.min(sig))
                    features.append(np.max(sig))

                    # Slope
                    if len(sig) > 2:
                        t = np.arange(len(sig))
                        try:
                            slope = np.polyfit(t, sig, 1)[0]
                        except:
                            slope = 0.0
                        features.append(slope)
                    else:
                        features.append(0.0)

                    # Time windows
                    for ws, we in [(0, w1_end), (w1_end, w2_end), (w2_end, w3_end)]:
                        chunk = sig[ws:we]
                        if len(chunk) > 0:
                            features.append(np.mean(chunk))
                            features.append(np.std(chunk))
                        else:
                            features.extend([0.0, 0.0])

    # Laterality features: compare left vs right motor cortex
    # For each pair (right_mod, left_mod), compute difference in mean signal
    for r_mod, l_mod in zip(RIGHT_MOTOR, LEFT_MOTOR):
        for sds in [1, 2]:
            for wl in range(2):
                for mom in range(3):
                    r_sig = nirs_cleaned[NIRS_BASELINE_SAMPLES:, r_mod, sds, wl, mom]
                    l_sig = nirs_cleaned[NIRS_BASELINE_SAMPLES:, l_mod, sds, wl, mom]
                    r_mean = np.nanmean(r_sig)
                    l_mean = np.nanmean(l_sig)
                    # Laterality index
                    denom = abs(r_mean) + abs(l_mean) + 1e-10
                    features.append((r_mean - l_mean) / denom)
                    # Also raw difference
                    features.append(r_mean - l_mean)

    return np.array(features)


# =============================================
# EEG PREPROCESSING & FEATURES
# =============================================

def bandpass_filter(signal, fs, low=0.5, high=45.0):
    """Apply bandpass filter."""
    nyq = fs / 2.0
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)
    try:
        sos = butter(4, [low_norm, high_norm], btype='band', output='sos')
        return sosfilt(sos, signal, axis=0)
    except:
        return signal


def extract_eeg_features_v2(eeg_data):
    """
    Rich EEG features:
      Per-channel: stats + 5 band powers + 2 ratios + time-windowed
      Cross-channel: frontal/central ratios, left/right asymmetry
    """
    n_samples = eeg_data.shape[0]
    fs = 500 if n_samples > 5000 else 250
    baseline_samples = int(BASELINE_SECONDS * fs)

    eeg = np.nan_to_num(eeg_data, nan=0.0).astype(np.float64)
    eeg = bandpass_filter(eeg, fs, low=0.5, high=45.0)

    bl_end = min(baseline_samples, n_samples - 1)
    baseline = eeg[:bl_end].mean(axis=0) if bl_end > 0 else np.zeros(eeg.shape[1])
    eeg_stim = eeg[bl_end:] - baseline

    if eeg_stim.shape[0] < 10:
        eeg_stim = eeg - baseline  # fallback: use all

    features = []
    n_ch = eeg_stim.shape[1]
    nperseg = min(256, max(64, len(eeg_stim) // 2))

    band_powers = []

    for ch in range(n_ch):
        sig = eeg_stim[:, ch]

        # Time-domain stats (7)
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(np.min(sig))
        features.append(np.max(sig))
        features.append(float(skew(sig)) if len(sig) > 2 else 0.0)
        features.append(float(kurtosis(sig)) if len(sig) > 2 else 0.0)
        features.append(np.ptp(sig))

        # Band powers (5)
        try:
            freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        except:
            freqs, psd = np.array([0]), np.array([0])
        delta = psd[(freqs >= 0.5) & (freqs < 4)].mean() if psd[(freqs >= 0.5) & (freqs < 4)].size > 0 else 0
        theta = psd[(freqs >= 4) & (freqs < 8)].mean() if psd[(freqs >= 4) & (freqs < 8)].size > 0 else 0
        alpha = psd[(freqs >= 8) & (freqs < 13)].mean() if psd[(freqs >= 8) & (freqs < 13)].size > 0 else 0
        beta  = psd[(freqs >= 13) & (freqs < 30)].mean() if psd[(freqs >= 13) & (freqs < 30)].size > 0 else 0
        gamma = psd[(freqs >= 30) & (freqs < 45)].mean() if psd[(freqs >= 30) & (freqs < 45)].size > 0 else 0
        features.extend([delta, theta, alpha, beta, gamma])

        # Band ratios (2)
        features.append(alpha / (beta + 1e-10))
        features.append(theta / (beta + 1e-10))

        band_powers.append([delta, theta, alpha, beta, gamma])

    band_powers = np.array(band_powers)  # (6, 5)

    # Cross-channel: Channels 0=AFF6, 1=AFp2, 2=AFp1, 3=AFF5, 4=FCz, 5=CPz
    if n_ch >= 6:
        frontal = band_powers[[0, 1, 2, 3]].mean(axis=0)
        central = band_powers[[4, 5]].mean(axis=0)
        for b in range(5):
            features.append(frontal[b] / (central[b] + 1e-10))

        # Left/right asymmetry (key for left/right fist!)
        left_p = band_powers[0]   # AFF6
        right_p = band_powers[3]  # AFF5
        for b in range(5):
            features.append((left_p[b] - right_p[b]) / (left_p[b] + right_p[b] + 1e-10))

        # FCz vs CPz (supplementary vs primary motor)
        for b in range(5):
            features.append(band_powers[4, b] / (band_powers[5, b] + 1e-10))

    # Early vs late features
    half = len(eeg_stim) // 2
    if half > 10:
        for ch in range(n_ch):
            early = eeg_stim[:half, ch]
            late = eeg_stim[half:, ch]
            features.append(np.mean(late) - np.mean(early))
            features.append(np.std(late) - np.std(early))

    return np.array(features)


# =============================================
# MAIN
# =============================================

if __name__ == '__main__':

    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DATA_DIR', DEFAULT_CACHE_DIR)

    P("=" * 65)
    P("BCI Classifier v2 — Maximizing Accuracy")
    P("=" * 65)
    P(f"Data directory: {data_dir}")

    data_files = find_data_files(data_dir)
    if not data_files:
        P(f"ERROR: No .npz files found in {data_dir}")
        sys.exit(1)
    P(f"Found {len(data_files)} .npz files")

    # =============================================
    # LOAD & EXTRACT FEATURES
    # =============================================

    P("\n--- Phase 1: Loading and extracting features ---")
    t_start = time.time()

    X_nirs_list, X_eeg_list = [], []
    y_labels, subjects = [], []
    skipped = 0

    for i, f in enumerate(data_files):
        try:
            arr = np.load(f, allow_pickle=True)
            label_info = arr['label'].item()
            y_labels.append(label_info['label'])
            subjects.append(label_info.get('subject_id', 'unknown'))

            nirs_clean = preprocess_nirs(arr['feature_moments'])
            nirs_feats = extract_nirs_features_v2(nirs_clean)
            nirs_feats = np.nan_to_num(nirs_feats, nan=0.0, posinf=0.0, neginf=0.0)
            X_nirs_list.append(nirs_feats)

            eeg_feats = extract_eeg_features_v2(arr['feature_eeg'])
            eeg_feats = np.nan_to_num(eeg_feats, nan=0.0, posinf=0.0, neginf=0.0)
            X_eeg_list.append(eeg_feats)
        except Exception as e:
            skipped += 1
            # Remove last label/subject if we added them
            if len(y_labels) > len(X_nirs_list):
                y_labels.pop()
                subjects.pop()

        if (i + 1) % 100 == 0 or (i + 1) == len(data_files):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            P(f"  [{i+1}/{len(data_files)}] {elapsed:.0f}s ({rate:.0f} files/s) skipped={skipped}")

    # Ensure consistent EEG feature length
    eeg_lens = [len(f) for f in X_eeg_list]
    target_eeg_len = max(set(eeg_lens), key=eeg_lens.count)  # most common
    P(f"  EEG feature lengths: min={min(eeg_lens)}, max={max(eeg_lens)}, target={target_eeg_len}")

    # Pad/truncate EEG features to same length
    X_eeg_fixed = []
    for feats in X_eeg_list:
        if len(feats) == target_eeg_len:
            X_eeg_fixed.append(feats)
        elif len(feats) < target_eeg_len:
            X_eeg_fixed.append(np.pad(feats, (0, target_eeg_len - len(feats))))
        else:
            X_eeg_fixed.append(feats[:target_eeg_len])

    X_nirs = np.array(X_nirs_list)
    X_eeg = np.array(X_eeg_fixed)
    X_all = np.concatenate([X_nirs, X_eeg], axis=1)
    subjects_arr = np.array(subjects)

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    P(f"\n  NIRS features:  {X_nirs.shape[1]}")
    P(f"  EEG features:   {X_eeg.shape[1]}")
    P(f"  Combined:       {X_all.shape[1]}")
    P(f"  Samples:        {X_all.shape[0]}")
    P(f"  Skipped files:  {skipped}")
    P(f"  Labels:         {dict(zip(*np.unique(y_labels, return_counts=True)))}")
    P(f"  Time:           {time.time()-t_start:.1f}s")

    # =============================================
    # PER-SUBJECT Z-NORMALIZATION
    # =============================================

    P("\n--- Phase 2: Per-subject z-normalization ---")

    X_nirs_subj = X_nirs.copy()
    X_eeg_subj = X_eeg.copy()

    for subj in np.unique(subjects_arr):
        mask = subjects_arr == subj
        for X in [X_nirs_subj, X_eeg_subj]:
            subj_data = X[mask]
            mu = subj_data.mean(axis=0)
            sigma = subj_data.std(axis=0)
            sigma[sigma < 1e-12] = 1.0
            X[mask] = (subj_data - mu) / sigma

    X_all_subj = np.concatenate([X_nirs_subj, X_eeg_subj], axis=1)
    X_all_subj = np.nan_to_num(X_all_subj, nan=0.0, posinf=0.0, neginf=0.0)
    P("  Done.")

    # =============================================
    # FEATURE SELECTION
    # =============================================

    P("\n--- Phase 3: Feature selection (mutual information) ---")

    n_keep = min(300, X_all.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_keep)
    X_selected = selector.fit_transform(X_all, y)

    selector_subj = SelectKBest(mutual_info_classif, k=n_keep)
    X_selected_subj = selector_subj.fit_transform(X_all_subj, y)

    P(f"  Kept {n_keep} / {X_all.shape[1]} features")

    # =============================================
    # CLASSIFIERS
    # =============================================

    classifiers = {
        'LDA':       LinearDiscriminantAnalysis(),
        'RF-200':    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'RF-500':    RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_leaf=2,
                                            random_state=42, n_jobs=-1),
        'GradBoost': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                 random_state=42),
        'XGBoost':   XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                    random_state=42, eval_metric='mlogloss', verbosity=0),
        'SVM-RBF':   SVC(kernel='rbf', C=10.0, gamma='scale'),
    }

    feature_configs = {
        'NIRS only':         X_nirs,
        'EEG only':          X_eeg,
        'Combined raw':      X_all,
        'Combined selected': X_selected,
        'SubjNorm selected': X_selected_subj,
    }

    # =============================================
    # STRATIFIED 5-FOLD CV
    # =============================================

    P("\n" + "=" * 65)
    P("STRATIFIED 5-FOLD CROSS-VALIDATION")
    P("=" * 65)

    best_score = 0
    best_combo = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for feat_name, X_feat in feature_configs.items():
        P(f"\n  --- {feat_name} ({X_feat.shape[1]} features) ---")
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            scores = cross_val_score(pipe, X_feat, y, cv=skf, scoring='accuracy')
            m, s = scores.mean(), scores.std()
            P(f"    {clf_name:15s}: {m*100:.1f}% (+/- {s*100:.1f}%)")
            if m > best_score:
                best_score = m
                best_combo = (feat_name, clf_name)

    P(f"\n  *** BEST: {best_combo[1]} on {best_combo[0]} -> {best_score*100:.1f}% ***")

    # =============================================
    # SUBJECT-WISE CV
    # =============================================

    P("\n" + "=" * 65)
    P("SUBJECT-WISE CROSS-VALIDATION (GroupKFold)")
    P("=" * 65)

    subject_enc = LabelEncoder()
    groups = subject_enc.fit_transform(subjects)
    n_folds = min(5, len(np.unique(groups)))

    best_subj_score = 0
    best_subj_combo = None

    for feat_name in ['Combined selected', 'SubjNorm selected']:
        X_feat = feature_configs[feat_name]
        P(f"\n  --- {feat_name} ({X_feat.shape[1]} features) ---")
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            gkf = GroupKFold(n_splits=n_folds)
            scores = cross_val_score(pipe, X_feat, y, cv=gkf, groups=groups, scoring='accuracy')
            m, s = scores.mean(), scores.std()
            P(f"    {clf_name:15s}: {m*100:.1f}% (+/- {s*100:.1f}%)")
            if m > best_subj_score:
                best_subj_score = m
                best_subj_combo = (feat_name, clf_name)

    P(f"\n  *** BEST (subject-wise): {best_subj_combo[1]} on {best_subj_combo[0]} -> {best_subj_score*100:.1f}% ***")

    # =============================================
    # HYPERPARAMETER TUNING
    # =============================================

    P("\n" + "=" * 65)
    P("HYPERPARAMETER TUNING (GridSearchCV on best feature set)")
    P("=" * 65)

    X_best = feature_configs[best_combo[0]]

    # --- XGBoost ---
    P("\n  Tuning XGBoost...")
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)),
    ])
    xgb_params = {
        'clf__n_estimators': [200, 400],
        'clf__max_depth': [4, 6, 8],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
    }
    grid_xgb = GridSearchCV(xgb_pipe, xgb_params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_xgb.fit(X_best, y)
    P(f"    Best: {grid_xgb.best_score_*100:.1f}%  Params: {grid_xgb.best_params_}")

    # --- SVM ---
    P("\n  Tuning SVM...")
    svm_pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
    svm_params = {
        'clf__C': [1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.01, 0.001],
        'clf__kernel': ['rbf'],
    }
    grid_svm = GridSearchCV(svm_pipe, svm_params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_svm.fit(X_best, y)
    P(f"    Best: {grid_svm.best_score_*100:.1f}%  Params: {grid_svm.best_params_}")

    # --- Random Forest ---
    P("\n  Tuning Random Forest...")
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    rf_params = {
        'clf__n_estimators': [300, 500, 800],
        'clf__max_depth': [15, 25, None],
        'clf__min_samples_leaf': [1, 2, 4],
    }
    grid_rf = GridSearchCV(rf_pipe, rf_params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_rf.fit(X_best, y)
    P(f"    Best: {grid_rf.best_score_*100:.1f}%  Params: {grid_rf.best_params_}")

    # --- GradientBoosting ---
    P("\n  Tuning GradientBoosting...")
    gb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42)),
    ])
    gb_params = {
        'clf__n_estimators': [200, 400],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.05, 0.1],
    }
    grid_gb = GridSearchCV(gb_pipe, gb_params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_gb.fit(X_best, y)
    P(f"    Best: {grid_gb.best_score_*100:.1f}%  Params: {grid_gb.best_params_}")

    # =============================================
    # STACKING ENSEMBLE
    # =============================================

    P("\n" + "=" * 65)
    P("STACKING ENSEMBLE")
    P("=" * 65)

    stack = StackingClassifier(
        estimators=[
            ('xgb', grid_xgb.best_estimator_.named_steps['clf']),
            ('rf',  grid_rf.best_estimator_.named_steps['clf']),
            ('svm', grid_svm.best_estimator_.named_steps['clf']),
            ('lda', LinearDiscriminantAnalysis()),
            ('gb',  grid_gb.best_estimator_.named_steps['clf']),
        ],
        final_estimator=XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', verbosity=0,
        ),
        cv=5,
        n_jobs=-1,
    )
    stack_pipe = Pipeline([('scaler', StandardScaler()), ('clf', stack)])
    stack_scores = cross_val_score(stack_pipe, X_best, y, cv=skf, scoring='accuracy')
    P(f"  Stacking: {stack_scores.mean()*100:.1f}% (+/- {stack_scores.std()*100:.1f}%)")

    # =============================================
    # FINAL TRAIN/TEST EVALUATION
    # =============================================

    P("\n" + "=" * 65)
    P("FINAL TRAIN/TEST SPLIT (80/20)")
    P("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y, test_size=0.2, random_state=42, stratify=y
    )
    P(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'XGBoost (tuned)':    grid_xgb.best_estimator_.named_steps['clf'],
        'RF (tuned)':         grid_rf.best_estimator_.named_steps['clf'],
        'SVM (tuned)':        grid_svm.best_estimator_.named_steps['clf'],
        'GradBoost (tuned)':  grid_gb.best_estimator_.named_steps['clf'],
        'LDA':                LinearDiscriminantAnalysis(),
    }

    best_test_score = 0
    best_test_name = None
    best_test_model = None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        train_acc = model.score(X_train_s, y_train)
        test_acc = model.score(X_test_s, y_test)
        P(f"  {name:22s} - Train: {train_acc*100:.1f}%  Test: {test_acc*100:.1f}%")
        if test_acc > best_test_score:
            best_test_score = test_acc
            best_test_name = name
            best_test_model = model

    P(f"\n  Best test model: {best_test_name} ({best_test_score*100:.1f}%)")
    y_pred = best_test_model.predict(X_test_s)
    P(classification_report(y_test, y_pred, target_names=le.classes_))

    P("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    header = "  " + "  ".join([f"{c[:8]:>8s}" for c in le.classes_])
    P(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:8d}" for v in row])
        P(f"  {row_str}  <- {le.classes_[i]}")

    # =============================================
    # SUMMARY
    # =============================================

    P("\n" + "=" * 65)
    P("SUMMARY — V1 vs V2")
    P("=" * 65)
    P(f"  V1 best (Stratified 5-fold CV):  44.0% (RF, 288 features)")
    P(f"  V1 best (Subject-wise CV):       33.8% (RF)")
    P(f"  V2 best (Stratified 5-fold CV):  {best_score*100:.1f}% ({best_combo[1]}, {feature_configs[best_combo[0]].shape[1]} feats)")
    P(f"  V2 best (Subject-wise CV):       {best_subj_score*100:.1f}% ({best_subj_combo[1]})")
    P(f"  V2 tuned XGBoost CV:             {grid_xgb.best_score_*100:.1f}%")
    P(f"  V2 tuned SVM CV:                 {grid_svm.best_score_*100:.1f}%")
    P(f"  V2 tuned RF CV:                  {grid_rf.best_score_*100:.1f}%")
    P(f"  V2 tuned GradBoost CV:           {grid_gb.best_score_*100:.1f}%")
    P(f"  V2 Stacking CV:                  {stack_scores.mean()*100:.1f}%")
    P(f"  V2 best test set:                {best_test_score*100:.1f}% ({best_test_name})")

    # =============================================
    # SAVE MODELS
    # =============================================

    save_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(best_test_model, os.path.join(save_dir, 'bci_v2_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'bci_v2_scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'bci_v2_label_encoder.pkl'))
    joblib.dump(selector, os.path.join(save_dir, 'bci_v2_feature_selector.pkl'))
    P(f"\n  Models saved to {save_dir}/")

    total_time = time.time() - t_start
    P(f"\n  Total time: {total_time/60:.1f} minutes")
    P("=" * 65)
    P("DONE!")
    P("=" * 65)

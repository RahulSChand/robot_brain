"""
Clean NIRS Classifier — Testing if removing dead channels improves accuracy
============================================================================

Data quality findings:
  - NaN is all-or-nothing: a channel either works perfectly or is 100% dead
  - 91.3% of channels are perfectly clean, 8.7% are fully dead
  - Worst offenders: modules 34-39 (back of head), SDS=2 (long distance)
  - Motor cortex SDS=1 (medium): ~100% alive for all motor modules
  - Motor cortex SDS=2 (long): 80-93% alive

We compare:
  A) BASELINE: All 40 modules × SDS [1,2], NaN→0 (what got 42%)
  B) CLEAN: Only reliable channels (≥80% alive), NaN→0
  C) SDS1 ONLY: All 40 modules × SDS [1] only (nearly 100% alive)
  D) MOTOR ONLY: 10 motor modules × SDS [1,2] (all reliable)
  E) MOTOR SDS1: 10 motor modules × SDS [1] only (cleanest)
  F) MEAN IMPUTE: All 40 × SDS [1,2], but NaN→column mean instead of 0
"""

import numpy as np
from numpy.linalg import lstsq
from glob import glob
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

def P(msg=""):
    print(msg, flush=True)

# =============================================
# CONFIG
# =============================================

DEFAULT_CACHE_DIR = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--KernelCo--robot_control'
)

NIRS_BASELINE_SAMPLES = 14
NIRS_SR = 4.76

MOTOR_RIGHT = [10, 12, 16, 20, 22]
MOTOR_LEFT  = [11, 13, 17, 21, 23]
MOTOR_ALL   = MOTOR_RIGHT + MOTOR_LEFT

# Channels that are alive <80% of the time (should be dropped)
UNRELIABLE_SDS2 = {26, 27, 34, 35, 36, 37, 38, 39}
UNRELIABLE_SDS1 = {34, 35, 36, 37}  # <80% alive even for medium distance


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
    """Short channel regression + baseline correction."""
    cleaned = nirs_data.copy().astype(np.float64)

    for mod in range(40):
        for wl in range(2):
            for mom in range(3):
                short = cleaned[:, mod, 0, wl, mom]
                if np.any(~np.isfinite(short)) or np.nanstd(short) < 1e-12:
                    continue
                short_2d = short.reshape(-1, 1)
                for sds in [1, 2]:
                    brain = cleaned[:, mod, sds, wl, mom]
                    if np.any(~np.isfinite(brain)) or np.nanstd(brain) < 1e-12:
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
    return cleaned


# =============================================
# FEATURE EXTRACTION (configurable channels)
# =============================================

def extract_nirs_features(nirs_cleaned, modules, sds_list, use_nan=False):
    """
    Extract features from specific (module, SDS) combinations.
    
    For each (module, SDS): IR wavelength (1), mean TOF (1)
      - mean, std, min, max, slope of post-stimulus signal
    
    If use_nan=True, returns NaN for dead channels (for later imputation).
    If use_nan=False, returns 0 for dead channels.
    """
    features = []
    stimulus = nirs_cleaned[NIRS_BASELINE_SAMPLES:]
    n_stim = len(stimulus)
    
    # Time windows for hemodynamic response
    t_early = min(14, n_stim)   # 0-3s
    t_peak = min(28, n_stim)    # 3-6s
    
    for mod in modules:
        for sds in sds_list:
            sig = stimulus[:, mod, sds, 1, 1]  # IR, mean TOF
            
            is_dead = np.all(np.isnan(sig))
            
            if is_dead:
                fill = np.nan if use_nan else 0.0
                features.extend([fill] * 8)  # 8 features per channel
                continue
            
            sig = np.nan_to_num(sig, nan=0.0)
            
            # Basic stats (4)
            features.append(np.mean(sig))
            features.append(np.std(sig))
            features.append(np.min(sig))
            features.append(np.max(sig))
            
            # Slope (1)
            if len(sig) > 2:
                t = np.arange(len(sig))
                try:
                    slope = np.polyfit(t, sig, 1)[0]
                except:
                    slope = 0.0
                features.append(slope)
            else:
                features.append(0.0)
            
            # Time-windowed means (3): early, peak, late
            features.append(np.mean(sig[:t_early]) if t_early > 0 else 0.0)
            features.append(np.mean(sig[t_early:t_peak]) if t_peak > t_early else 0.0)
            features.append(np.mean(sig[t_peak:]) if t_peak < n_stim else 0.0)
    
    return np.array(features)


def extract_laterality_features(nirs_cleaned, sds_list):
    """Left vs right motor cortex difference — key for left/right fist."""
    features = []
    stimulus = nirs_cleaned[NIRS_BASELINE_SAMPLES:]
    
    for r_mod, l_mod in zip(MOTOR_RIGHT, MOTOR_LEFT):
        for sds in sds_list:
            r_sig = stimulus[:, r_mod, sds, 1, 1]
            l_sig = stimulus[:, l_mod, sds, 1, 1]
            
            r_dead = np.all(np.isnan(r_sig))
            l_dead = np.all(np.isnan(l_sig))
            
            if r_dead or l_dead:
                features.extend([0.0, 0.0])
                continue
            
            r_sig = np.nan_to_num(r_sig, nan=0.0)
            l_sig = np.nan_to_num(l_sig, nan=0.0)
            
            r_mean = np.mean(r_sig)
            l_mean = np.mean(l_sig)
            
            # Laterality index
            denom = abs(r_mean) + abs(l_mean) + 1e-10
            features.append((r_mean - l_mean) / denom)
            features.append(r_mean - l_mean)
    
    return np.array(features)


# =============================================
# MAIN
# =============================================

if __name__ == '__main__':

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CACHE_DIR

    P("=" * 65)
    P("Clean NIRS Classifier — Dead Channel Removal Experiment")
    P("=" * 65)
    P(f"Data directory: {data_dir}")

    data_files = find_data_files(data_dir)
    if not data_files:
        P(f"ERROR: No .npz files found")
        sys.exit(1)
    P(f"Found {len(data_files)} files\n")

    # =============================================
    # LOAD DATA
    # =============================================

    P("--- Loading and preprocessing ---")
    t0 = time.time()

    # We'll extract features for ALL strategies in one pass
    strategies = {
        'A) Baseline (40mod×SDS1,2→0)': {
            'modules': list(range(40)), 'sds': [1, 2], 'use_nan': False, 'laterality': True
        },
        'B) Clean ≥80% (drop unreliable)': {
            'modules': [m for m in range(40) if m not in UNRELIABLE_SDS1],
            'sds': [1],  # Only SDS 1 for clean
            'sds2_modules': [m for m in range(40) if m not in UNRELIABLE_SDS2],
            'use_nan': False, 'laterality': True
        },
        'C) SDS1 only (40 modules)': {
            'modules': list(range(40)), 'sds': [1], 'use_nan': False, 'laterality': True
        },
        'D) Motor only (10mod×SDS1,2)': {
            'modules': MOTOR_ALL, 'sds': [1, 2], 'use_nan': False, 'laterality': True
        },
        'E) Motor SDS1 only (cleanest)': {
            'modules': MOTOR_ALL, 'sds': [1], 'use_nan': False, 'laterality': True
        },
        'F) All→NaN + mean impute': {
            'modules': list(range(40)), 'sds': [1, 2], 'use_nan': True, 'laterality': True
        },
    }

    all_features = {name: [] for name in strategies}
    y_labels = []
    subjects = []

    for i, f in enumerate(data_files):
        arr = np.load(f, allow_pickle=True)
        label_info = arr['label'].item()
        y_labels.append(label_info['label'])
        subjects.append(label_info.get('subject_id', 'unknown'))

        nirs_clean = preprocess_nirs(arr['feature_moments'])

        for name, cfg in strategies.items():
            if name == 'B) Clean ≥80% (drop unreliable)':
                # Special: SDS1 for all non-unreliable, SDS2 only for reliable modules
                feats_s1 = extract_nirs_features(nirs_clean, cfg['modules'], [1], cfg['use_nan'])
                feats_s2 = extract_nirs_features(nirs_clean, cfg['sds2_modules'], [2], cfg['use_nan'])
                lat = extract_laterality_features(nirs_clean, [1, 2])
                feats = np.concatenate([feats_s1, feats_s2, lat])
            else:
                feats = extract_nirs_features(nirs_clean, cfg['modules'], cfg['sds'], cfg['use_nan'])
                lat = extract_laterality_features(nirs_clean, cfg['sds'])
                feats = np.concatenate([feats, lat])

            all_features[name].append(feats)

        if (i + 1) % 200 == 0 or (i + 1) == len(data_files):
            P(f"  [{i+1}/{len(data_files)}] {time.time()-t0:.0f}s")

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    subjects_arr = np.array(subjects)
    groups = LabelEncoder().fit_transform(subjects)

    P(f"\nLabels: {list(le.classes_)}")
    P(f"Samples: {len(y)}, Subjects: {len(set(subjects))}")

    # =============================================
    # CLASSIFICATION
    # =============================================

    classifiers = {
        'RF-500': RandomForestClassifier(n_estimators=500, max_depth=30, 
                                          min_samples_leaf=2, random_state=42, n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='rbf', C=10.0, gamma='scale'),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_group_folds = min(5, len(np.unique(groups)))

    P(f"\n{'=' * 65}")
    P("STRATIFIED 5-FOLD CV — COMPARING STRATEGIES")
    P("=" * 65)

    best_overall = 0
    best_name = ""

    for strat_name, feat_list in all_features.items():
        X = np.array(feat_list)
        
        # Handle NaN for strategy F (mean imputation)
        if 'NaN' in strat_name:
            # Replace NaN with column mean
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for c in range(X.shape[1]):
                mask = np.isnan(X[:, c])
                X[mask, c] = col_means[c]
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        P(f"\n  {strat_name}  ({X.shape[1]} features)")
        
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
            m, s = scores.mean(), scores.std()
            marker = ""
            if m > best_overall:
                best_overall = m
                best_name = f"{strat_name} + {clf_name}"
                marker = " ★"
            P(f"    {clf_name:8s}: {m*100:.1f}% (+/- {s*100:.1f}%){marker}")

    P(f"\n  ★ BEST: {best_name} → {best_overall*100:.1f}%")

    # =============================================
    # SUBJECT-WISE CV (hardest test)
    # =============================================

    P(f"\n{'=' * 65}")
    P("SUBJECT-WISE CV (GroupKFold) — Best 3 strategies")
    P("=" * 65)

    # Re-test top strategies with subject-wise CV
    for strat_name, feat_list in all_features.items():
        X = np.array(feat_list)
        if 'NaN' in strat_name:
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for c in range(X.shape[1]):
                mask = np.isnan(X[:, c])
                X[mask, c] = col_means[c]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        P(f"\n  {strat_name}  ({X.shape[1]} features)")
        gkf = GroupKFold(n_splits=n_group_folds)
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring='accuracy')
            P(f"    {clf_name:8s}: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    P(f"\nTotal time: {time.time()-t0:.0f}s")
    P("=" * 65)
    P("DONE!")
    P("=" * 65)


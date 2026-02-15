"""
Data Quality Analysis for KernelCo/robot_control dataset
=========================================================
Run this BEFORE training to understand the data and check for issues.

Checks:
  - NaN / Inf / zero counts per file
  - Shape consistency
  - Value ranges (min, max, mean)
  - Per-module quality (which NIRS modules have bad data)
  - Per-subject / session breakdown
  - Label distribution
"""

import numpy as np
from glob import glob
from collections import Counter, defaultdict
import os
import sys

# =============================================
# CONFIG
# =============================================

DEFAULT_CACHE_DIR = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--KernelCo--robot_control'
)

# Remote path (uncomment when running on remote machine):
# DEFAULT_CACHE_DIR = '/mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control'


def find_data_files(data_dir):
    """Find all .npz files."""
    for pattern in [
        os.path.join(data_dir, 'snapshots', '*', 'data', '*.npz'),
        os.path.join(data_dir, '*.npz'),
        os.path.join(data_dir, '**', '*.npz'),
    ]:
        files = sorted(glob(pattern, recursive=True))
        if files:
            return files
    return []


if __name__ == '__main__':

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CACHE_DIR

    print("=" * 70)
    print("DATA QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Data directory: {data_dir}\n")

    data_files = find_data_files(data_dir)
    if not data_files:
        print(f"ERROR: No .npz files found in {data_dir}")
        sys.exit(1)

    print(f"Total .npz files: {len(data_files)}\n")

    # =============================================
    # 1. SCAN ALL FILES
    # =============================================

    print("-" * 70)
    print("1. SCANNING ALL FILES")
    print("-" * 70)

    labels = Counter()
    subjects = Counter()
    sessions = Counter()
    subject_labels = defaultdict(Counter)  # subject -> label counts

    # Per-file stats
    files_with_nirs_nan = []
    files_with_nirs_inf = []
    files_with_eeg_nan = []
    files_with_eeg_inf = []
    files_with_bad_shape = []

    # Per-module NaN/Inf counts (across all files)
    # nirs shape: (72, 40, 3, 2, 3)
    module_nan_count = np.zeros(40, dtype=int)
    module_inf_count = np.zeros(40, dtype=int)
    module_zero_frac = np.zeros(40, dtype=float)

    # Aggregate value ranges
    eeg_mins, eeg_maxs = [], []
    nirs_mins, nirs_maxs = [], []

    for i, f in enumerate(data_files):
        arr = np.load(f, allow_pickle=True)
        fname = os.path.basename(f)

        eeg = arr['feature_eeg']
        nirs = arr['feature_moments']
        label_info = arr['label'].item()

        # Labels / subjects / sessions
        lab = label_info['label']
        subj = label_info.get('subject_id', 'unknown')
        sess = label_info.get('session_id', 'unknown')
        labels[lab] += 1
        subjects[subj] += 1
        sessions[sess] += 1
        subject_labels[subj][lab] += 1

        # Shape check
        if eeg.shape != (7499, 6) or nirs.shape != (72, 40, 3, 2, 3):
            files_with_bad_shape.append((fname, eeg.shape, nirs.shape))

        # EEG quality
        if np.any(np.isnan(eeg)):
            files_with_eeg_nan.append(fname)
        if np.any(np.isinf(eeg)):
            files_with_eeg_inf.append(fname)

        # NIRS quality
        if np.any(np.isnan(nirs)):
            files_with_nirs_nan.append(fname)
        if np.any(np.isinf(nirs)):
            files_with_nirs_inf.append(fname)

        # Per-module quality
        for mod in range(40):
            mod_data = nirs[:, mod, :, :, :]
            module_nan_count[mod] += np.isnan(mod_data).sum()
            module_inf_count[mod] += np.isinf(mod_data).sum()
            total_vals = mod_data.size
            module_zero_frac[mod] += (mod_data == 0).sum() / total_vals

        # Value ranges
        eeg_clean = eeg[np.isfinite(eeg)]
        nirs_clean = nirs[np.isfinite(nirs)]
        if len(eeg_clean) > 0:
            eeg_mins.append(eeg_clean.min())
            eeg_maxs.append(eeg_clean.max())
        if len(nirs_clean) > 0:
            nirs_mins.append(nirs_clean.min())
            nirs_maxs.append(nirs_clean.max())

        if (i + 1) % 200 == 0 or (i + 1) == len(data_files):
            print(f"  Scanned {i+1}/{len(data_files)} files...")

    n = len(data_files)
    module_zero_frac /= n  # average fraction of zeros per module

    # =============================================
    # 2. RESULTS
    # =============================================

    print(f"\n{'=' * 70}")
    print("2. LABEL DISTRIBUTION")
    print("=" * 70)
    for lab, cnt in sorted(labels.items()):
        print(f"  {lab:20s}: {cnt:5d}  ({cnt/n*100:.1f}%)")
    print(f"  {'TOTAL':20s}: {n:5d}")

    print(f"\n{'=' * 70}")
    print("3. SUBJECTS & SESSIONS")
    print("=" * 70)
    print(f"  Unique subjects: {len(subjects)}")
    print(f"  Unique sessions: {len(sessions)}")
    print(f"\n  Samples per subject:")
    for subj, cnt in sorted(subjects.items(), key=lambda x: -x[1]):
        label_str = ", ".join(f"{l}:{c}" for l, c in sorted(subject_labels[subj].items()))
        print(f"    {subj}: {cnt:4d} samples  ({label_str})")

    print(f"\n{'=' * 70}")
    print("4. SHAPE CONSISTENCY")
    print("=" * 70)
    if files_with_bad_shape:
        print(f"  WARNING: {len(files_with_bad_shape)} files have unexpected shapes!")
        for fname, eshape, nshape in files_with_bad_shape[:10]:
            print(f"    {fname}: EEG={eshape}, NIRS={nshape}")
    else:
        print(f"  All {n} files have correct shapes:")
        print(f"    EEG:  (7499, 6)")
        print(f"    NIRS: (72, 40, 3, 2, 3)")

    print(f"\n{'=' * 70}")
    print("5. NaN / Inf CHECK")
    print("=" * 70)
    print(f"  EEG  files with NaN: {len(files_with_eeg_nan):4d}/{n}")
    print(f"  EEG  files with Inf: {len(files_with_eeg_inf):4d}/{n}")
    print(f"  NIRS files with NaN: {len(files_with_nirs_nan):4d}/{n}")
    print(f"  NIRS files with Inf: {len(files_with_nirs_inf):4d}/{n}")

    if files_with_nirs_nan:
        print(f"\n  First 10 NIRS NaN files:")
        for f in files_with_nirs_nan[:10]:
            print(f"    {f}")

    if files_with_nirs_inf:
        print(f"\n  First 10 NIRS Inf files:")
        for f in files_with_nirs_inf[:10]:
            print(f"    {f}")

    if files_with_eeg_nan:
        print(f"\n  First 10 EEG NaN files:")
        for f in files_with_eeg_nan[:10]:
            print(f"    {f}")

    print(f"\n{'=' * 70}")
    print("6. PER-MODULE NIRS QUALITY (across all files)")
    print("=" * 70)
    print(f"  {'Module':>8s}  {'NaN count':>10s}  {'Inf count':>10s}  {'Avg zero %':>10s}  {'Status':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for mod in range(40):
        status = "OK"
        if module_nan_count[mod] > 0 or module_inf_count[mod] > 0:
            status = "BAD"
        elif module_zero_frac[mod] > 0.5:
            status = "WARN"
        print(f"  {mod:8d}  {module_nan_count[mod]:10d}  {module_inf_count[mod]:10d}  "
              f"{module_zero_frac[mod]*100:9.1f}%  {status:>8s}")

    print(f"\n{'=' * 70}")
    print("7. VALUE RANGES")
    print("=" * 70)
    if eeg_mins:
        print(f"  EEG  range: [{min(eeg_mins):.4f}, {max(eeg_maxs):.4f}] uV")
    if nirs_mins:
        print(f"  NIRS range: [{min(nirs_mins):.6f}, {max(nirs_maxs):.6f}]")

    # =============================================
    # 3. SAMPLE A FEW FILES IN DETAIL
    # =============================================

    print(f"\n{'=' * 70}")
    print("8. DETAILED SAMPLE (first 3 files)")
    print("=" * 70)
    for f in data_files[:3]:
        arr = np.load(f, allow_pickle=True)
        fname = os.path.basename(f)
        eeg = arr['feature_eeg']
        nirs = arr['feature_moments']
        label_info = arr['label'].item()

        print(f"\n  File: {fname}")
        print(f"  Label: {label_info['label']}, Subject: {label_info['subject_id']}, "
              f"Session: {label_info['session_id']}, Duration: {label_info['duration']:.2f}s")

        print(f"  EEG:  shape={eeg.shape}  dtype={eeg.dtype}")
        print(f"        min={np.nanmin(eeg):.4f}  max={np.nanmax(eeg):.4f}  "
              f"mean={np.nanmean(eeg):.4f}  std={np.nanstd(eeg):.4f}")
        print(f"        NaN={np.isnan(eeg).sum()}  Inf={np.isinf(eeg).sum()}  "
              f"Zero={np.sum(eeg==0)}")

        # EEG per-channel
        print(f"        Per-channel means: {[f'{np.nanmean(eeg[:,c]):.2f}' for c in range(6)]}")

        print(f"  NIRS: shape={nirs.shape}  dtype={nirs.dtype}")
        nirs_finite = nirs[np.isfinite(nirs)]
        if len(nirs_finite) > 0:
            print(f"        min={nirs_finite.min():.6f}  max={nirs_finite.max():.6f}  "
                  f"mean={nirs_finite.mean():.6f}")
        print(f"        NaN={np.isnan(nirs).sum()}  Inf={np.isinf(nirs).sum()}  "
              f"Zero={np.sum(nirs==0)}")

    # =============================================
    # SUMMARY
    # =============================================

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if files_with_nirs_nan:
        issues.append(f"{len(files_with_nirs_nan)} files have NaN in NIRS")
    if files_with_nirs_inf:
        issues.append(f"{len(files_with_nirs_inf)} files have Inf in NIRS")
    if files_with_eeg_nan:
        issues.append(f"{len(files_with_eeg_nan)} files have NaN in EEG")
    if files_with_eeg_inf:
        issues.append(f"{len(files_with_eeg_inf)} files have Inf in EEG")
    if files_with_bad_shape:
        issues.append(f"{len(files_with_bad_shape)} files have unexpected shapes")

    bad_modules = [m for m in range(40) if module_nan_count[m] > 0 or module_inf_count[m] > 0]
    if bad_modules:
        issues.append(f"Modules with NaN/Inf: {bad_modules}")

    warn_modules = [m for m in range(40) if module_zero_frac[m] > 0.5]
    if warn_modules:
        issues.append(f"Modules with >50% zeros (possibly inactive): {warn_modules}")

    if issues:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(f"    - {iss}")
        print(f"\n  The training script handles these by replacing NaN/Inf with 0")
        print(f"  and skipping regression for bad channels. Training should still work.")
    else:
        print("  DATA IS CLEAN! No NaN, Inf, or shape issues found.")
        print("  All 40 NIRS modules appear to have valid data.")
        print("  You're good to train!")

    print()


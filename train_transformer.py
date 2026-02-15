#!/usr/bin/env python3
"""
NIRS Transformer Classifier
============================
Trains a small transformer on raw NIRS time-series data for motor imagery
classification (Left Fist, Right Fist, Both Fists, Tongue Tapping, Relax).

Instead of hand-crafted features (mean, std, slope → 50% accuracy),
the transformer learns temporal patterns directly from the 72-timestep
hemodynamic response.

Architecture:
  Raw NIRS (72, 80) → Conv1d patch embed → Transformer Encoder → Classify

Data findings:
  - 1395 samples, 17 subjects, 5 balanced classes (279 each)
  - NaN is all-or-nothing per channel (dead sensor → replace with 0)
  - 91.3% of channels work perfectly, 8.7% fully dead
  - Hand-crafted features plateau at ~50% (SVM, stratified CV)

Requirements:
  pip install torch numpy scipy scikit-learn
  (Run on CUDA GPU for reasonable speed)

Usage:
  python train_transformer.py
  python train_transformer.py /path/to/dataset
"""

import numpy as np
from numpy.linalg import lstsq
from glob import glob
import os
import sys
import time
import warnings
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

def P(msg=""):
    print(msg, flush=True)


# =============================================
# CONFIG
# =============================================

# Remote path (GPU machine):
DEFAULT_CACHE_DIR = '/mnt/amlfs-01/home/rchand/workspace/code/r_dataset/datasets--KernelCo--robot_control'

# Local path (Mac, no GPU — uncomment if testing locally):
# DEFAULT_CACHE_DIR = os.path.expanduser(
#     '~/.cache/huggingface/hub/datasets--KernelCo--robot_control'
# )

# NIRS config
NIRS_BASELINE_SAMPLES = 14      # ~3s at 4.76 Hz
N_MODULES = 40
SDS_LIST = [1, 2]               # medium + long distance channels
WAVELENGTH = 1                  # IR (index 1)
MOMENT = 1                      # mean time of flight (index 1)
N_CHANNELS = N_MODULES * len(SDS_LIST)  # 80

# Training config
BATCH_SIZE = 32
EPOCHS = 150
LR = 5e-4
WEIGHT_DECAY = 0.01
PATIENCE = 20                   # early stopping
N_FOLDS = 5
SEED = 42

# Model config
D_MODEL = 64                    # transformer hidden dimension
N_HEADS = 4                     # attention heads
N_LAYERS = 2                    # transformer layers
FF_DIM = 128                    # feedforward dimension
DROPOUT = 0.3                   # regularization (aggressive for small dataset)

# Augmentation
AUG_NOISE_STD = 0.02            # Gaussian noise
AUG_TIME_SHIFT = 3              # max time shift (samples)
AUG_CHANNEL_DROP = 0.1          # probability of zeroing a channel


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
    """
    Short channel regression + baseline correction.
    Input:  (72, 40, 3, 2, 3)
    Output: (72, 80) — 40 modules × 2 SDS, using IR wavelength + mean TOF
    """
    cleaned = nirs_data.copy().astype(np.float64)

    # Short channel regression
    for mod in range(N_MODULES):
        for wl in range(2):
            for mom in range(3):
                short = cleaned[:, mod, 0, wl, mom]
                if np.any(~np.isfinite(short)) or np.nanstd(short) < 1e-12:
                    continue
                short_2d = short.reshape(-1, 1)
                for sds in SDS_LIST:
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

    # Baseline correction
    baseline = np.nanmean(cleaned[:NIRS_BASELINE_SAMPLES], axis=0)
    cleaned = cleaned - baseline

    # Extract channels: 40 modules × 2 SDS (medium, long), IR, mean TOF
    # Result shape: (72, 80)
    channels = []
    for mod in range(N_MODULES):
        for sds in SDS_LIST:
            ch = cleaned[:, mod, sds, WAVELENGTH, MOMENT]
            ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
            channels.append(ch)

    return np.stack(channels, axis=1)  # (72, 80)


# =============================================
# DATASET
# =============================================

class NIRSDataset(Dataset):
    """PyTorch dataset for NIRS time-series data."""

    def __init__(self, data, labels, subjects, augment=False):
        """
        data:     np.array (N, 72, 80) — preprocessed NIRS
        labels:   np.array (N,) — integer labels
        subjects: np.array (N,) — subject IDs (for subject-wise CV)
        augment:  bool — whether to apply data augmentation
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.subjects = subjects
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].clone()  # (72, 80)
        y = self.labels[idx]

        if self.augment:
            # 1. Gaussian noise
            x = x + torch.randn_like(x) * AUG_NOISE_STD

            # 2. Random time shift (circular)
            shift = torch.randint(-AUG_TIME_SHIFT, AUG_TIME_SHIFT + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

            # 3. Random channel dropout
            mask = torch.rand(x.shape[1]) > AUG_CHANNEL_DROP
            x = x * mask.unsqueeze(0)

        return x, y


# =============================================
# MODEL: NIRS TRANSFORMER
# =============================================

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for 72 timesteps."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pos_embedding[:, :x.size(1), :]


class NIRSTransformer(nn.Module):
    """
    Small transformer for NIRS time-series classification.

    Architecture:
      Input (72, 80) → Conv1d patch embed → (72, d_model)
                     → Positional encoding
                     → Transformer Encoder (N layers)
                     → Global Average Pooling → (d_model,)
                     → Classification head → (5,)
    """

    def __init__(self, n_channels=80, n_classes=5, seq_len=72,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()

        # Patch embedding: Conv1d to project 80 input channels → d_model
        # Kernel size 5 captures ~1 second of NIRS data at 4.76 Hz
        self.patch_embed = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable for small datasets)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len=72, n_channels=80)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)          # (batch, 80, 72)
        x = self.patch_embed(x)         # (batch, d_model, 72)
        x = x.transpose(1, 2)          # (batch, 72, d_model)

        # Positional encoding
        x = self.pos_encoding(x)
        x = self.input_dropout(x)

        # Transformer
        x = self.transformer(x)        # (batch, 72, d_model)
        x = self.norm(x)

        # Global average pooling over time
        x = x.mean(dim=1)              # (batch, d_model)

        # Classify
        x = self.classifier(x)         # (batch, n_classes)
        return x


# =============================================
# MODEL: CNN BASELINE (for comparison)
# =============================================

class NIRSCNN(nn.Module):
    """
    Simple 1D CNN baseline for comparison.
    """

    def __init__(self, n_channels=80, n_classes=5, dropout=DROPOUT):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (batch, 80, 72)
        x = self.features(x)           # (batch, 64, 1)
        x = x.squeeze(-1)              # (batch, 64)
        x = self.classifier(x)
        return x


# =============================================
# TRAINING UTILITIES
# =============================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping (helps with transformer stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================
# CROSS-VALIDATION TRAINING
# =============================================

def train_cv(model_class, model_name, data, labels, subjects, groups,
             device, n_folds=N_FOLDS, subject_wise=False):
    """
    Train with K-fold cross-validation.
    Returns per-fold accuracies.
    """
    if subject_wise:
        n_actual_folds = min(n_folds, len(np.unique(groups)))
        kf = GroupKFold(n_splits=n_actual_folds)
        splits = list(kf.split(data, labels, groups))
        cv_name = f"Subject-wise {n_actual_folds}-fold"
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(kf.split(data, labels))
        cv_name = f"Stratified {n_folds}-fold"

    P(f"\n  {model_name} — {cv_name} CV")
    P(f"  {'─' * 50}")

    fold_accs = []
    all_test_preds = []
    all_test_labels = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Split train into train + val (85/15)
        n_train = len(train_idx)
        n_val = max(1, int(n_train * 0.15))
        np.random.seed(SEED + fold_idx)
        perm = np.random.permutation(n_train)
        val_subset = train_idx[perm[:n_val]]
        train_subset = train_idx[perm[n_val:]]

        # Create datasets
        train_ds = NIRSDataset(data[train_subset], labels[train_subset],
                               subjects[train_subset], augment=True)
        val_ds = NIRSDataset(data[val_subset], labels[val_subset],
                             subjects[val_subset], augment=False)
        test_ds = NIRSDataset(data[test_idx], labels[test_idx],
                              subjects[test_idx], augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Create model
        model = model_class(n_channels=N_CHANNELS, n_classes=5).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        # Load best model and evaluate on test set
        model.load_state_dict(best_state)
        model.to(device)
        test_loss, test_acc, preds, true_labels = evaluate(
            model, test_loader, criterion, device
        )

        fold_accs.append(test_acc)
        all_test_preds.extend(preds)
        all_test_labels.extend(true_labels)

        stopped = epoch + 1
        P(f"    Fold {fold_idx+1}: test acc = {test_acc*100:.1f}%  "
          f"(stopped at epoch {stopped}, best val loss = {best_val_loss:.4f})")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    P(f"  ──────────────────────────────────────────────────")
    P(f"  Mean: {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")

    return mean_acc, std_acc, np.array(all_test_preds), np.array(all_test_labels)


# =============================================
# MAIN
# =============================================

if __name__ == '__main__':

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CACHE_DIR

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        P(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        P(f"Using Apple MPS")
    else:
        device = torch.device('cpu')
        P(f"Using CPU (will be slow!)")

    P("=" * 65)
    P("NIRS Transformer Classifier")
    P("=" * 65)
    P(f"Data directory: {data_dir}")

    data_files = find_data_files(data_dir)
    if not data_files:
        P(f"ERROR: No .npz files found in {data_dir}")
        sys.exit(1)
    P(f"Found {len(data_files)} files")

    # =============================================
    # LOAD & PREPROCESS
    # =============================================

    P("\n--- Loading and preprocessing NIRS data ---")
    t0 = time.time()

    all_data = []       # (N, 72, 80)
    all_labels = []
    all_subjects = []

    for i, f in enumerate(data_files):
        arr = np.load(f, allow_pickle=True)
        label_info = arr['label'].item()
        all_labels.append(label_info['label'])
        all_subjects.append(label_info.get('subject_id', 'unknown'))

        nirs_processed = preprocess_nirs(arr['feature_moments'])  # (72, 80)
        all_data.append(nirs_processed)

        if (i + 1) % 200 == 0 or (i + 1) == len(data_files):
            P(f"  [{i+1}/{len(data_files)}] {time.time()-t0:.0f}s")

    data = np.array(all_data, dtype=np.float32)    # (1395, 72, 80)
    subjects = np.array(all_subjects)

    le = LabelEncoder()
    labels = le.fit_transform(all_labels)

    # Per-channel z-normalization across the full dataset
    # (each of the 80 channels gets zero mean, unit variance)
    for ch in range(data.shape[2]):
        ch_data = data[:, :, ch]
        mu = ch_data.mean()
        sigma = ch_data.std()
        if sigma > 1e-10:
            data[:, :, ch] = (ch_data - mu) / sigma

    # Group encoding for subject-wise CV
    group_enc = LabelEncoder()
    groups = group_enc.fit_transform(all_subjects)

    P(f"\nDataset:")
    P(f"  Shape:    {data.shape}  (samples, timesteps, channels)")
    P(f"  Classes:  {list(le.classes_)}")
    P(f"  Samples:  {len(labels)}")
    P(f"  Subjects: {len(set(all_subjects))}")
    P(f"  Labels:   {dict(Counter(all_labels))}")
    P(f"  Time:     {time.time()-t0:.0f}s")

    # =============================================
    # MODEL SUMMARIES
    # =============================================

    P(f"\n--- Model architectures ---")

    dummy_transformer = NIRSTransformer(n_channels=N_CHANNELS, n_classes=5)
    dummy_cnn = NIRSCNN(n_channels=N_CHANNELS, n_classes=5)

    P(f"\n  Transformer: {count_parameters(dummy_transformer):,} parameters")
    P(f"    d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, ff={FF_DIM}")
    P(f"    dropout={DROPOUT}, patch_embed=Conv1d(5)+Conv1d(3)")

    P(f"\n  CNN Baseline: {count_parameters(dummy_cnn):,} parameters")
    P(f"    Conv1d(7)→Conv1d(5)→Conv1d(3)→Conv1d(3)→AvgPool→FC")

    del dummy_transformer, dummy_cnn

    # =============================================
    # TRAINING: STRATIFIED 5-FOLD CV
    # =============================================

    P(f"\n{'=' * 65}")
    P("STRATIFIED 5-FOLD CROSS-VALIDATION")
    P("=" * 65)

    results = {}

    # CNN baseline
    cnn_acc, cnn_std, cnn_preds, cnn_labels = train_cv(
        NIRSCNN, "CNN Baseline", data, labels, subjects, groups, device,
        n_folds=N_FOLDS, subject_wise=False
    )
    results['CNN (stratified)'] = (cnn_acc, cnn_std)

    # Transformer
    tf_acc, tf_std, tf_preds, tf_labels = train_cv(
        NIRSTransformer, "Transformer", data, labels, subjects, groups, device,
        n_folds=N_FOLDS, subject_wise=False
    )
    results['Transformer (stratified)'] = (tf_acc, tf_std)

    # =============================================
    # TRAINING: SUBJECT-WISE CV
    # =============================================

    P(f"\n{'=' * 65}")
    P("SUBJECT-WISE CROSS-VALIDATION (GroupKFold)")
    P("  Tests generalization to completely new subjects")
    P("=" * 65)

    cnn_subj_acc, cnn_subj_std, _, _ = train_cv(
        NIRSCNN, "CNN Baseline", data, labels, subjects, groups, device,
        n_folds=N_FOLDS, subject_wise=True
    )
    results['CNN (subject-wise)'] = (cnn_subj_acc, cnn_subj_std)

    tf_subj_acc, tf_subj_std, tf_subj_preds, tf_subj_labels = train_cv(
        NIRSTransformer, "Transformer", data, labels, subjects, groups, device,
        n_folds=N_FOLDS, subject_wise=True
    )
    results['Transformer (subject-wise)'] = (tf_subj_acc, tf_subj_std)

    # =============================================
    # DETAILED REPORT (best transformer fold)
    # =============================================

    P(f"\n{'=' * 65}")
    P("CLASSIFICATION REPORT (Transformer, stratified CV, all folds combined)")
    P("=" * 65)

    P(classification_report(
        tf_labels, tf_preds, target_names=le.classes_, digits=3
    ))

    P("Confusion Matrix:")
    cm = confusion_matrix(tf_labels, tf_preds)
    header = "          " + "  ".join([f"{c[:8]:>8s}" for c in le.classes_])
    P(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:8d}" for v in row])
        P(f"          {row_str}  ← {le.classes_[i]}")

    # =============================================
    # SUMMARY
    # =============================================

    P(f"\n{'=' * 65}")
    P("SUMMARY — COMPARISON")
    P("=" * 65)

    P(f"\n  Previous best (hand-crafted features):")
    P(f"    Stratified 5-fold: 50.3% (SVM)")
    P(f"    Subject-wise:      40.2% (RF)")

    P(f"\n  Deep learning results:")
    for name, (acc, std) in results.items():
        marker = " ★" if acc > 0.503 else ""
        P(f"    {name:35s}: {acc*100:.1f}% (+/- {std*100:.1f}%){marker}")

    # =============================================
    # SAVE BEST MODEL (full training on all data)
    # =============================================

    P(f"\n{'=' * 65}")
    P("TRAINING FINAL MODEL ON ALL DATA")
    P("=" * 65)

    # Train on all data for deployment
    full_ds = NIRSDataset(data, labels, subjects, augment=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=True)

    final_model = NIRSTransformer(n_channels=N_CHANNELS, n_classes=5).to(device)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    P(f"  Training for 80 epochs on all {len(labels)} samples...")
    for epoch in range(80):
        train_loss, train_acc = train_one_epoch(
            final_model, full_loader, optimizer, criterion, device
        )
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            P(f"    Epoch {epoch+1:3d}: loss={train_loss:.4f}  acc={train_acc*100:.1f}%")

    # Save
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, 'nirs_transformer.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'label_encoder_classes': le.classes_.tolist(),
        'n_channels': N_CHANNELS,
        'n_classes': 5,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'ff_dim': FF_DIM,
    }, save_path)
    P(f"\n  Model saved to {save_path}")

    total_time = time.time() - t0
    P(f"\n  Total time: {total_time/60:.1f} minutes")
    P("=" * 65)
    P("DONE!")
    P("=" * 65)


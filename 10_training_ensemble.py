import schedulefree
import zipfile
import os
import atexit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import gc
import time
from tqdm import tqdm
import torchvision.models as models
from datetime import datetime
import torch.nn.functional as F
from google.colab import userdata
import random
import seaborn as sns
import shutil
import smtplib
from email.mime.text import MIMEText
from torch.cuda.amp import autocast, GradScaler
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from sklearn.metrics import auc as sklearn_auc
import json
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image, UnidentifiedImageError
import torchvision.models as models
import re
from collections import defaultdict
from torchmetrics import PrecisionRecallCurve, AUROC, AveragePrecision
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
import math
from collections import defaultdict
import contextlib
import sys
from torch.utils.data import WeightedRandomSampler
from typing import Optional, Dict, Tuple
import traceback
import h5py


print("Libraries imported.")

# =============================================================================
# 1) Environment & Hardware
# =============================================================================
print("Configuring environment...")

# Device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# 2) Training Loop Control
# =============================================================================
NUM_EPOCHS = 100                 # Max epochs per architecture
PATIENCE = 5                    # Early stopping patience
UNLEASHED = False               # If True → disable early stopping
SEED = 24                       # For reproducibility

# =============================================================================
# 3) Data Loading & Performance
# =============================================================================
BATCH_SIZE = 32                 # Adjust to fit GPU memory
WORKERS = os.cpu_count()        # >0 enables background loading & prefetching
ACCUMULATION_STEPS = 4  # 8 * 4 = 32 effective batch size

# Val loader uses a larger batch size
VAL_BATCH_SIZE = BATCH_SIZE * 3  # e.g., 8 * 3 = 24

# =============================================================================
# 4) Model & Optimization
# =============================================================================
ENCODER_WEIGHTS = "imagenet"     # Pretrained encoders
OPTIMIZER_NAME = "AdamWScheduleFree"   # Options: AdamWScheduleFree, AdamW

# =============================================================================
# 5) Loss Function (from Khened et al., Scientific Reports 2021)
# =============================================================================
# Hybrid BCE + Dice loss weights
ALPHA_BCE    = 0.125
BETA_DICE_BG = 0.157
GAMMA_DICE_FG = 0.173


# =============================================================================
# 6) Operating Point & Sensitivity Design
# =============================================================================
SENSITIVITY_TARGET = 0.95        # Required TPR for threshold calibration


# =============================================================================
# 7) Dataset & Normalization
# =============================================================================
# Path on Google Drive containing TRAIN.h5 and VALIDATION.h5
HDF5_DRIVE_DIR  = "IA_MEDICA_SAMPLES/CAMELYON16"
METADATA_DIR    = "METADATA_CHECKPOINTS/CAMELYON16"
IDENTIFIER      = "CAMELYON16_HDF5_OPTIMIZED"

# Local temporary directory on Colab (fast I/O)
LOCAL_DATA_DIR = "/content/dataset"


# =============================================================================
# 9) Checkpoints, Tracking & Notifications
# =============================================================================
CHECKPOINT_PATH = "Checkpoints"        # Where models are saved
AIM_REPO_PATH   = "aim_repo_prostate"  # Experiment tracking

# Email notifications
sender     = "bruno.nunes.1987@gmail.com"
recipients = ["bruno.nunes.1987@gmail.com"]
password   = userdata.get("APP_PASSWORD")  # Must exist in Colab secrets

# =============================================================================
# 0) Execution Mode & Reproducibility Profiles
# =============================================================================
# Options: "FAST_DEV" (Speed prioritized) | "PAPER" (Strict Determinism)
EXECUTION_MODE = "FAST_DEV"
SMART_SAMPLING = True

if EXECUTION_MODE == "FAST_DEV":
    print("⚠️ RUNNING IN FAST_DEV MODE: Benchmarking ON, Determinism OFF")
    # Speed optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    # Defaults for dev
    SUBSET_RATIO = 1.0
    USE_SUBSET = False

elif EXECUTION_MODE == "PAPER":
    print("🛡️ RUNNING IN PAPER MODE: Strict Determinism Enforced")
    # Reproducibility settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Defaults for paper
    SUBSET_RATIO = 1.0
    USE_SUBSET = False # Force full dataset unless explicitly overridden

# Global AMP setting
AMP_PRECISION = "fp16" # or "bf16" or "fp32"
AMP_LOG = ""

# --- Seeding & Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

def worker_init_fn(worker_id):
    # Ensure DataLoader workers have distinct seeds
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # 2. CRITICAL: Disable OpenCV threading in workers
    # We want the worker to use 1 core fully, not spawn sub-threads.
    # This reduces contention/jitter significantly.
    cv2.setNumThreads(0)

print(f"Reproducibility configured. SEED={SEED}")

def get_learning_rate(architecture: str):
    CONFIGS = {
        "SWIN":       {"lr": 3e-4, "wd": 1e-4},
        "SEGFORMER":  {"lr": 1e-3, "wd": 1e-4},
        "DPT":        {"lr": 3e-4, "wd": 1e-4},
        "UPERNET":    {"lr": 2e-4, "wd": 1e-4},
        "DEEPLABV3PLUS": {"lr": 1e-3, "wd": 1e-4},
        "UNET++":        {"lr": 5e-4, "wd": 1e-4},
        "FPN":           {"lr": 3e-4, "wd": 1e-4},
        "MANET":         {"lr": 4e-4, "wd": 1e-4},
    }

    try:
        cfg = CONFIGS[architecture.upper()]
    except KeyError:
        raise ValueError(f"Unknown architecture: {architecture}")

    return cfg["lr"], cfg["wd"]

def autocast_ctx(x: torch.Tensor, amp_dtype):
    use_cuda_amp = x.is_cuda and (amp_dtype is not torch.float32)
    return torch.amp.autocast("cuda", dtype=amp_dtype) if use_cuda_amp else contextlib.nullcontext()

def change_paths(path, possible_paths=None):
    if possible_paths is None:
        possible_paths = [
            '/content/drive/MyDrive/Personal_Drive_Bruno/',
            '/content/drive/MyDrive/'
        ]

    for p in possible_paths:
        # Create a new variable here so 'path' stays original
        full_path = os.path.join(p, path)

        if os.path.exists(full_path):
            return full_path

    # Use the original path in the error message for clarity
    raise ValueError(f"The path {path} does not match any known environments.")

CHECKPOINT_PATH=change_paths(CHECKPOINT_PATH)
AIM_REPO_PATH=change_paths(AIM_REPO_PATH)
HDF5_DRIVE_DIR=change_paths(HDF5_DRIVE_DIR)
METADATA_DIR=change_paths(METADATA_DIR)

class TrainingHealthTracker:
    """
    Tracks training/validation health issues:
    - NaN/Inf loss counts
    - Skipped batch counts (with reasons)
    - Validation collapse count (v_results is None)
    - Invalid metric counts (AUPRC/AUROC/MCC* NaN/Inf)
    """

    def __init__(self, name="run",patience_collapse=3):
        self.name = name
        self.patience_collapse = patience_collapse
        self.current_consecutive_collapses = 0 # <--- PERSISTENT COUNTER
        self.reset_run()
        self.reset_epoch()

    def reset_run(self):
        self.run = {
            "train_naninf_loss": 0,
            "val_naninf_loss": 0,
            "train_skipped_batches": 0,
            "val_skipped_batches": 0,
            "train_skip_reasons": defaultdict(int),
            "val_skip_reasons": defaultdict(int),
            "val_collapse_epochs": 0,
            "val_invalid_metric_epochs": 0,
        }

    def reset_epoch(self):
        self.epoch = {
            "train_naninf_loss": 0,
            "val_naninf_loss": 0,
            "train_skipped_batches": 0,
            "val_skipped_batches": 0,
            "train_skip_reasons": defaultdict(int),
            "val_skip_reasons": defaultdict(int),
            "val_collapsed": 0,           # 0/1 for this epoch
            "val_invalid_metrics": 0,     # 0/1 for this epoch
        }

    # ---------------------------
    # Logging helpers
    # ---------------------------
    def train_skip(self, reason: str):
        self.epoch["train_skipped_batches"] += 1
        self.epoch["train_skip_reasons"][reason] += 1
        self.run["train_skipped_batches"] += 1
        self.run["train_skip_reasons"][reason] += 1

    def val_skip(self, reason: str):
        self.epoch["val_skipped_batches"] += 1
        self.epoch["val_skip_reasons"][reason] += 1
        self.run["val_skipped_batches"] += 1
        self.run["val_skip_reasons"][reason] += 1

    def train_naninf_loss(self):
        self.epoch["train_naninf_loss"] += 1
        self.run["train_naninf_loss"] += 1

    def val_naninf_loss(self):
        self.epoch["val_naninf_loss"] += 1
        self.run["val_naninf_loss"] += 1

    # --- UPDATED LOGIC ---
    def mark_val_collapsed(self):
        self.epoch["val_collapsed"] = 1
        self.run["val_collapse_epochs"] += 1

        # Increment consecutive counter
        self.current_consecutive_collapses += 1

    def mark_val_invalid_metrics(self):
        self.epoch["val_invalid_metrics"] = 1
        self.run["val_invalid_metric_epochs"] += 1
        # Invalid metrics (NaNs) are treated as a collapse
        self.current_consecutive_collapses += 1

    def reset_collapse_counter(self):
        """Call this when a validation run is successful/healthy"""
        self.current_consecutive_collapses = 0

    def should_emergency_stop(self):
        """Returns True if we hit the collapse limit"""
        return self.current_consecutive_collapses >= self.patience_collapse

    # ---------------------------
    # Pretty printing
    # ---------------------------
    @staticmethod
    def _fmt_reasons(reason_dict):
        if not reason_dict:
            return "-"
        items = sorted(reason_dict.items(), key=lambda x: (-x[1], x[0]))
        return ", ".join([f"{k}:{v}" for k, v in items])

    def log_epoch(self, epoch_num: int, prefix="[Health]"):
        print(
            f"{prefix} Epoch {epoch_num} | "
            f"Train skip={self.epoch['train_skipped_batches']} "
            f"(reasons: {self._fmt_reasons(self.epoch['train_skip_reasons'])}) | "
            f"Train NaN/Inf loss={self.epoch['train_naninf_loss']} | "
            f"Val skip={self.epoch['val_skipped_batches']} "
            f"(reasons: {self._fmt_reasons(self.epoch['val_skip_reasons'])}) | "
            f"Val NaN/Inf loss={self.epoch['val_naninf_loss']} | "
            f"Val collapsed={self.epoch['val_collapsed']} | "
            f"Val invalid_metrics={self.epoch['val_invalid_metrics']}"
        )

    def log_run(self, prefix="[Health-SUMMARY]"):
        print(
            f"{prefix} Run totals | "
            f"Train skip={self.run['train_skipped_batches']} "
            f"(reasons: {self._fmt_reasons(self.run['train_skip_reasons'])}) | "
            f"Train NaN/Inf loss={self.run['train_naninf_loss']} | "
            f"Val skip={self.run['val_skipped_batches']} "
            f"(reasons: {self._fmt_reasons(self.run['val_skip_reasons'])}) | "
            f"Val NaN/Inf loss={self.run['val_naninf_loss']} | "
            f"Val collapse_epochs={self.run['val_collapse_epochs']} | "
            f"Val invalid_metric_epochs={self.run['val_invalid_metric_epochs']}"
        )

class BCEDiceHybridLossPaper(nn.Module):
    """
    Paper-faithful implementation of the hybrid loss from:

    Khened, M., Kori, A., Rajkumar, H. et al.
    A generalized deep learning framework for whole-slide image segmentation and analysis.
    Scientific Reports 11, 11579 (2021).
    https://doi.org/10.1038/s41598-021-90444-8

    Loss = alpha * CE + beta * Dice_BG + gamma * Dice_FG

    - CE is binary cross-entropy on the tumor posterior p_i
    - Dice uses squared denominator: sum(p^2) + sum(g^2)
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.25,
                 gamma: float = 0.25,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    @staticmethod
    def _flatten(x):
        # [B,H,W] -> [B,N]
        return x.reshape(x.size(0), -1)

    def _dice_loss_khened(self, p, g):
        """
        p, g: [B,H,W] probabilities and binary GT for ONE class
        Implements:
            DL = 1 - (2 sum(p g)) / (sum(p^2) + sum(g^2))
        """
        p = self._flatten(p)
        g = self._flatten(g)

        intersection = (p * g).sum(dim=1)
        denom = (p.pow(2).sum(dim=1) + g.pow(2).sum(dim=1))

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, target):
        """
        logits: [B, 2, H, W] raw model outputs
        target: [B, H, W] or [B, 1, H, W] Class Indices (0 or 1).
                NOTE: This replaces the external One-Hot requirement to speed up training.
        """

        # -------------------------------------------------
        # 0) INTERNAL OPTIMIZATION: One-Hot on GPU
        #    Moves complexity out of the training loop to fix the bottleneck.
        # -------------------------------------------------
        # Ensure target is [B, H, W]
        # canonicalize target to [B,H,W]
        if target.ndim == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        elif target.ndim == 4 and target.size(-1) == 1:
            target = target[..., 0]

        assert target.ndim == 3, (
            "[BCEDiceHybridLossPaper] Target tensor invariant violated: "
            f"expected [B,H,W] class-index mask, got shape={tuple(target.shape)} "
            f"dtype={target.dtype}. One-hot encoding must NOT be applied outside the loss."
        )

        # Create One-Hot [B, H, W] -> [B, H, W, 2] -> [B, 2, H, W]
        target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        if logits.shape != target_one_hot.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape}, target {target_one_hot.shape}"
            )

        # -------------------------------------------------
        # 1) Posterior probabilities (softmax)
        # -------------------------------------------------
        probs = torch.softmax(logits, dim=1)

        # Tumor posterior p_i and GT g_i (paper notation)
        p_fg = probs[:, 1, :, :]
        g_fg = target_one_hot[:, 1, :, :]

        p_fg = p_fg.clamp(1e-7, 1.0 - 1e-7)

        # -------------------------------------------------
        # 2) Binary Cross-Entropy (Eq. 2)
        # -------------------------------------------------
        ce_loss = -(g_fg * torch.log(p_fg) +
                    (1.0 - g_fg) * torch.log(1.0 - p_fg))
        ce_loss = ce_loss.mean()

        # -------------------------------------------------
        # 3) Dice losses (Eq. 1)
        # -------------------------------------------------
        # Background
        dice_bg = self._dice_loss_khened(
            probs[:, 0, :, :],
            target_one_hot[:, 0, :, :]
        )

        # Foreground (tumor)
        dice_fg = self._dice_loss_khened(
            probs[:, 1, :, :],
            target_one_hot[:, 1, :, :]
        )

        # -------------------------------------------------
        # 4) Hybrid combination (Eq. 3)
        # -------------------------------------------------
        loss = (
            self.alpha * ce_loss +
            self.beta  * dice_bg +
            self.gamma * dice_fg
        )

        return loss

class AdvancedMetricTracker:
    """
    Memory-safe, incremental validation tracker for binary segmentation.
    OPTIMIZED VERSION: Keeps all accumulation on GPU to prevent pipeline stalls.

    - AUPRC / AUROC computed incrementally using binned thresholds.
    - Collapse detection via FG prevalence @ 0.5.
    - MCC* computed incrementally over a fixed threshold grid [0.1..0.9].
    """

    def __init__(
        self,
        device: torch.device,
        metric_bins: int = 2048,   # Reduced to 2048 for speed (scientifically sufficient for Val)
        mcc_thresholds=None,       # defaults to 0.1..0.9 step 0.1
        collapse_low: float = 0.001,
        collapse_high: float = 0.99,
        from_logits: bool = True,
    ):
        self.device = device
        self.from_logits = from_logits

        # Torchmetrics streaming (binned) metrics - initialized on GPU
        self.auprc = BinaryAveragePrecision(thresholds=metric_bins).to(device)
        self.auroc = BinaryAUROC(thresholds=metric_bins).to(device)

        # MCC* thresholds - move to GPU immediately
        if mcc_thresholds is None:
            mcc_thresholds = torch.arange(0.1, 1.0, 0.1)
        self.mcc_thresholds = torch.as_tensor(mcc_thresholds, dtype=torch.float32).to(device)

        self.collapse_low = float(collapse_low)
        self.collapse_high = float(collapse_high)

        self.reset()

    def reset(self):
        # Reset torchmetrics internal state
        self.auprc.reset()
        self.auroc.reset()

        # FG prevalence @ 0.5 stats
        self._pred_pos_at_05 = 0
        self._total_pixels = 0

        # MCC* confusion counts per-threshold
        # CRITICAL OPTIMIZATION: Initialize these on the GPU (device)
        # We do not want to move data to CPU during the training loop.
        K = int(self.mcc_thresholds.numel())
        self._tp = torch.zeros(K, dtype=torch.int64, device=self.device)
        self._fp = torch.zeros(K, dtype=torch.int64, device=self.device)
        self._tn = torch.zeros(K, dtype=torch.int64, device=self.device)
        self._fn = torch.zeros(K, dtype=torch.int64, device=self.device)

    @staticmethod
    def _extract_probs_fg(pred_logits: torch.Tensor) -> torch.Tensor:
        if pred_logits.ndim != 4:
            raise ValueError(f"Expected pred_logits [B,C,H,W], got {tuple(pred_logits.shape)}")

        B, C, H, W = pred_logits.shape
        if C == 1:
            probs = torch.sigmoid(pred_logits[:, 0, ...])
        elif C == 2:
            probs = torch.softmax(pred_logits, dim=1)[:, 1, ...]
        else:
            raise ValueError(f"Expected C=1 or C=2 for binary segmentation, got C={C}")

        return probs.to(dtype=torch.float32)

    @staticmethod
    def _extract_target_fg(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            tgt = target
        elif target.ndim == 4 and target.shape[1] == 2:
            tgt = target[:, 1, ...]
        elif target.ndim == 4 and target.shape[1] == 1:
            tgt = target[:, 0, ...]
        else:
            raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")

        if tgt.dtype != torch.bool:
            tgt = tgt > 0.5
        return tgt

    @torch.no_grad()
    def update(self, pred_logits: torch.Tensor, target: torch.Tensor):
        """
        Incrementally updates AUPRC/AUROC + collapse stats + MCC* counts.
        Everything stays on GPU.
        """
        probs_fg = self._extract_probs_fg(pred_logits)  # [B,H,W], float32
        tgt_fg = self._extract_target_fg(target)        # [B,H,W], bool

        p = probs_fg.reshape(-1)
        y = tgt_fg.reshape(-1)

        # Update torchmetrics (keeps data on GPU)
        self.auprc.update(p, y)
        self.auroc.update(p, y)

        # Collapse stats (scalars are cheap)
        pred_pos = (p >= 0.5).sum().item()
        total = p.numel()
        self._pred_pos_at_05 += int(pred_pos)
        self._total_pixels += int(total)

        # MCC* counts per threshold
        # 'thresholds' is already on GPU from __init__

        # Broadcasting: [1, N] vs [K, 1] -> [K, N]
        # This operation is heavy but highly parallelizable on GPU
        preds_k = p.unsqueeze(0) >= self.mcc_thresholds.unsqueeze(1)
        y_k = y.unsqueeze(0)

        # Accumulate strictly on GPU
        # REMOVED: .to("cpu")
        self._tp += (preds_k & y_k).sum(dim=1)
        self._fp += (preds_k & ~y_k).sum(dim=1)
        self._tn += (~preds_k & ~y_k).sum(dim=1)
        self._fn += (~preds_k & y_k).sum(dim=1)

    @staticmethod
    def _mcc_from_counts(tp, fp, tn, fn, eps: float = 1e-12) -> torch.Tensor:
        # Cast to float64 for precision during division
        tp = tp.to(torch.float64)
        fp = fp.to(torch.float64)
        tn = tn.to(torch.float64)
        fn = fn.to(torch.float64)
        num = tp * tn - fp * fn
        den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        return (num / den).to(torch.float32)

    @staticmethod
    def _extract_probs_from_probs_fg(probs_fg: torch.Tensor) -> torch.Tensor:
        if probs_fg.ndim == 4 and probs_fg.shape[1] == 1:
            probs_fg = probs_fg[:, 0, ...]
        if probs_fg.ndim != 3:
            raise ValueError(f"Expected probs_fg [B,H,W], got {tuple(probs_fg.shape)}")
        return probs_fg.to(dtype=torch.float32).clamp(0.0, 1.0)

    @torch.no_grad()
    def update_from_probs_fg(self, probs_fg: torch.Tensor, target: torch.Tensor):
        """
        Update using pre-computed probabilities (e.g. from TTA).
        """
        probs_fg = self._extract_probs_from_probs_fg(probs_fg)
        tgt_fg = self._extract_target_fg(target)

        p = probs_fg.reshape(-1)
        y = tgt_fg.reshape(-1)

        self.auprc.update(p, y)
        self.auroc.update(p, y)

        pred_pos = (p >= 0.5).sum().item()
        total = p.numel()
        self._pred_pos_at_05 += int(pred_pos)
        self._total_pixels += int(total)

        # GPU broadcasting
        preds_k = p.unsqueeze(0) >= self.mcc_thresholds.unsqueeze(1)
        y_k = y.unsqueeze(0)

        # GPU Accumulation
        self._tp += (preds_k & y_k).sum(dim=1)
        self._fp += (preds_k & ~y_k).sum(dim=1)
        self._tn += (~preds_k & ~y_k).sum(dim=1)
        self._fn += (~preds_k & y_k).sum(dim=1)

    def compute_and_reset(self, health: TrainingHealthTracker = None):
        """
        Moves accumulated stats to CPU *once* at end of epoch to compute final scores.
        """
        if self._total_pixels == 0:
            print("[VAL GUARDRAIL] No pixels processed in validation epoch.")
            self.reset()
            return None

        fg_prev = self._pred_pos_at_05 / float(self._total_pixels)

        # Guardrail 1: FG prevalence collapse
        if fg_prev < self.collapse_low:
            print(
                f"[VAL GUARDRAIL] COLLAPSE DETECTED — "
                f"FG prevalence @0.5 too LOW: {fg_prev:.6f} "
                f"(threshold {self.collapse_low:.6f})"
            )
            if health is not None:health.mark_val_collapsed();
            self.reset()
            return None

        if fg_prev > self.collapse_high:
            print(
                f"[VAL GUARDRAIL] COLLAPSE DETECTED — "
                f"FG prevalence @0.5 too HIGH: {fg_prev:.6f} "
                f"(threshold {self.collapse_high:.6f})"
            )
            if health is not None:health.mark_val_collapsed();
            self.reset()
            return None

        # Compute streaming AUPRC / AUROC
        # This automatically handles the transfer to CPU for the result
        auprc_t = self.auprc.compute().detach().cpu()
        auroc_t = self.auroc.compute().detach().cpu()

        # Guardrail 2: NaN / Inf AUPRC
        if (auprc_t.numel() == 0) or (not torch.isfinite(auprc_t).all()):
            print(f"[VAL GUARDRAIL] INVALID METRIC — AUPRC is NaN/Inf: {auprc_t}")
            if health is not None:
                health.mark_val_invalid_metrics()
            self.reset()
            return None

        # Guardrail 3: NaN / Inf AUROC
        if (auroc_t.numel() == 0) or (not torch.isfinite(auroc_t).all()):
            print(f"[VAL GUARDRAIL] INVALID METRIC — AUROC is NaN/Inf: {auroc_t}")
            if health is not None:
                health.mark_val_invalid_metrics()
            self.reset()
            return None

        val_auprc = float(auprc_t.item())
        val_auroc = float(auroc_t.item())

        # Compute MCC*
        # OPTIMIZATION: This is the ONLY time we move large tensors to CPU
        mccs = self._mcc_from_counts(
            self._tp.cpu(),
            self._fp.cpu(),
            self._tn.cpu(),
            self._fn.cpu()
        )

        # Guardrail 4: NaN / Inf MCC*
        if (mccs.numel() == 0) or (not torch.isfinite(mccs).all()):
            print(f"[VAL GUARDRAIL] INVALID METRIC — MCC* contains NaN/Inf: {mccs}")
            if health is not None:
                health.mark_val_invalid_metrics()
            self.reset()
            return None

        val_mcc_star = float(torch.max(mccs).item())

        # Final defensive check
        if not (math.isfinite(val_auprc) and math.isfinite(val_auroc) and math.isfinite(val_mcc_star)):
            print(f"[VAL GUARDRAIL] INVALID METRIC — AUPRC={val_auprc}, AUROC={val_auroc}, MCC*={val_mcc_star}")
            if health is not None: health.mark_val_invalid_metrics()
            self.reset()
            return None

        out = {
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "val_mcc_star": val_mcc_star,
            "fg_prevalence_at_05": float(fg_prev),
        }

        self.reset()
        return out

class RunningWeightedMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cumulative_score = 0.0
        self.sample_count = 0

    def update(self, batch_score_sum, batch_sample_count):
        """
        batch_score_sum: The sum of metrics in the batch (e.g., sum of Dices)
        batch_sample_count: The number of valid samples in that batch (e.g., count of tumor images)
        """
        self.cumulative_score += batch_score_sum
        self.sample_count += batch_sample_count

    def get_average(self):
        return self.cumulative_score / self.sample_count if self.sample_count > 0 else 0.0

# --- AUGMENTATION PIPELINE (Strictly Updated for Albumentations 1.4+ & nnU-Net) ---
def get_transforms(mode="train", img_size=224):
    if mode == "train":
        return A.Compose([
            # 2. GEOMETRIC INVARIANCE
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # Affine replacement for ShiftScaleRotate (Strict 1.4 API)
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-45, 45),
                fill=0,
                fill_mask=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            # 3. COLOR & INTENSITY ROBUSTNESS
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                # Gamma Augmentation (nnU-Net Page 11, Item 7)
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.8),

            # 4. ARTIFACT & QUALITY SIMULATION
            # 4. ARTIFACT & QUALITY SIMULATION (CPU Side)
            # Original: OneOf([Blur, Noise, Downscale], p=0.2) -> ~6.6% each
            # New: OneOf([Blur, Noise], p=0.14) -> 0.14 * 0.5 = ~7% each (Close enough)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1.0),
            ], p=0.14),

            # 5. REGULARIZATION (CoarseDropout / Cutout)
            # Strict 1.4 API: num_holes_range, fill
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, int(img_size/10)),
                hole_width_range=(1, int(img_size/10)),
                fill=0,
                fill_mask=0,
                p=0.2
            ),

            ToTensorV2(),
        ])

    else: # Validation / Test (Deterministic)
        return A.Compose([
            ToTensorV2(),
        ])

import psutil

class HybridProstateDataset(Dataset):
    def __init__(self, hdf5_path, mode="train", subset_indices=None):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.transform = get_transforms(mode=mode, img_size=224)

        # -------------------------------------------------------
        # 1. Metadata Loading & Size Estimation
        # -------------------------------------------------------
        print(f"Opening {hdf5_path}...")
        with h5py.File(self.hdf5_path, 'r') as f:
            self.full_labels = f["labels"][:]
            self.full_pids   = f["patient_ids"][:]

            # Get raw shapes to calculate memory footprint
            img_shape = f["images"].shape  # (N, H, W, C)
            mask_shape = f["masks"].shape  # (N, H, W)

            # Determine subset size
            if subset_indices is not None:
                self.indices = np.array(subset_indices)
            else:
                self.indices = np.arange(len(self.full_labels))

            n_samples = len(self.indices)

            # Calculate estimated RAM usage (uint8 = 1 byte)
            # Size = N * (H*W*C + H*W)
            bytes_per_sample = (img_shape[1] * img_shape[2] * img_shape[3]) + \
                               (mask_shape[1] * mask_shape[2])
            total_bytes_needed = n_samples * bytes_per_sample

        # -------------------------------------------------------
        # 2. Strategy Decision: RAM vs Disk
        # -------------------------------------------------------
        # Check available system RAM
        mem = psutil.virtual_memory()
        available_ram = mem.available

        # Heuristic: Load to RAM if it takes < 70% of available memory
        self.use_ram_cache = total_bytes_needed < (available_ram * 0.70)

        # Initialize Cache Variables
        self.images_cache = None
        self.masks_cache = None

        # Initialize Disk Variables
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._atexit_registered = False

        if self.use_ram_cache:
            print(f"✅ RAM Sufficient ({available_ram/1e9:.1f}GB avail). Loading {total_bytes_needed/1e9:.2f}GB into memory...")
            self._load_to_ram()
        else:
            print(f"⚠️ Dataset too large ({total_bytes_needed/1e9:.2f}GB) for RAM ({available_ram/1e9:.1f}GB). Using OPTIMIZED DISK MODE.")
            # Set metadata for the disk-based indices
            self.labels = self.full_labels[self.indices]
            self.patient_ids = self.full_pids[self.indices]

    def _load_to_ram(self):
        """Loads the dataset into Numpy arrays for instant access."""
        start_t = time.time()

        # Optimization: HDF5 reads are much faster when indices are sorted
        sorted_indices = np.sort(self.indices)

        with h5py.File(self.hdf5_path, 'r') as f:
            # Fancy indexing: Load only the subset directly into RAM
            self.images_cache = f["images"][sorted_indices]
            self.masks_cache = f["masks"][sorted_indices]

        # IMPORTANT: Since we sorted the data to read it, we must
        # sync the labels/pids to this sorted order.
        self.labels = self.full_labels[sorted_indices]
        self.patient_ids = self.full_pids[sorted_indices]

        # Reset indices to be 0..N (mapping to the cache directly)
        self.indices = np.arange(len(sorted_indices))

        print(f"Loaded {len(self.indices)} samples in {time.time()-start_t:.2f}s.")

    def _open_file(self):
        """Lazy loader for Disk Mode."""
        pid = os.getpid()
        # Handle fork safety
        if self.h5_file is not None and self._opened_pid != pid:
            self.close()

        if self.h5_file is None:
            # Optimization: 50MB chunk cache + latest libver
            self.h5_file = h5py.File(
                self.hdf5_path,
                "r",
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self._opened_pid = pid

            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # --- PATH A: RAM CACHE (Fast) ---
        if self.use_ram_cache:
            # Direct memory lookup
            # real_idx is just idx because we remapped indices in _load_to_ram
            image = self.images_cache[idx]
            mask = self.masks_cache[idx]

        # --- PATH B: DISK READ (Robust) ---
        else:
            if self.h5_file is None: self._open_file()

            # Map logical index to physical HDF5 index
            real_idx = self.indices[idx]

            image = self.images_dset[real_idx]
            mask = self.masks_dset[real_idx]

        # --- Common Augmentation ---
        try:
            augmented = self.transform(image=image, mask=mask)
            m = augmented["mask"]
            if m.ndim == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)
            return augmented["image"], m.long()
        except Exception as e:
            raise RuntimeError(f"Transform failed at idx={idx}") from e

    # --- Standard Getters ---
    def get_labels(self):
        return self.labels

    def get_patient_ids(self):
        return self.patient_ids

    def get_class_counts(self):
        counts = np.bincount(self.labels)
        return {
            'CANCER': counts[1] if len(counts) > 1 else 0,
            'NOT_CANCER': counts[0] if len(counts) > 0 else 0
        }

    # --- Cleanup & Pickling (Consolidated) ---
    def close(self):
        """Close HDF5 handle if it exists."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try: self.h5_file.close()
            except: pass
            self.h5_file = None
            self.images_dset = None
            self.masks_dset = None
            self._opened_pid = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        """
        Pickling:
        1. If RAM Mode: Pickle the numpy arrays (efficient copy-on-write in Linux).
        2. If Disk Mode: Drop file handles (cannot pickle open files).
        """
        state = self.__dict__.copy()
        # Always drop file handles
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # File handles remain None until next access
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._atexit_registered = False

class ProstateCancerDatasetHDF5(Dataset):
    def __init__(self, hdf5_path, mode="train", subset_indices=None):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.transform = get_transforms(mode=mode, img_size=224)

        # --- METADATA LOADING (Fast) ---
        with h5py.File(self.hdf5_path, 'r') as f:
            self.full_labels = f["labels"][:]      # Numpy array in RAM
            self.full_pids   = f["patient_ids"][:] # Numpy array in RAM
            self.total_len   = len(self.full_labels)

        # --- SUBSET LOGIC ---
        if subset_indices is not None:
            self.indices = np.array(subset_indices)
        else:
            self.indices = np.arange(self.total_len)

        # Filter metadata to match the subset (Critical for Sampler!)
        self.labels = self.full_labels[self.indices]
        self.patient_ids = self.full_pids[self.indices]

        # File handles (Lazy loading)
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._atexit_registered = False

    def _open_file(self):
        """Open the HDF5 file lazily (per-process safe).

        DataLoader with num_workers>0 forks/spawns worker processes. Each worker must
        hold its own independent HDF5 file handle. We therefore:
          - open on first access inside the current process
          - if the dataset object crosses a process boundary, close/reopen
          - register an atexit hook (once per process) to close cleanly
        """
        pid = os.getpid()

        # If we were opened in a different process (e.g., fork), close and reopen here.
        if self.h5_file is not None and self._opened_pid is not None and self._opened_pid != pid:
            self.close()

        if self.h5_file is None:
            # 50MB raw data chunk cache is sufficient for sequential-ish reading
            self.h5_file = h5py.File(
                self.hdf5_path,
                "r",
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024,
            )
            self.images_dset = self.h5_file["images"]
            self.masks_dset = self.h5_file["masks"]
            self._opened_pid = pid

            if not self._atexit_registered:
                # Ensure clean close when worker exits (best-effort).
                atexit.register(self.close)
                self._atexit_registered = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None: self._open_file()

        # Map logical index to physical HDF5 index
        real_idx = self.indices[idx]

        # Fast Read (No PNG decoding!)
        image = self.images_dset[real_idx] # uint8 [224,224,3]
        mask  = self.masks_dset[real_idx]  # uint8 [224,224]

        # Augment (using your existing transform pipeline)
        try:
            augmented = self.transform(image=image, mask=mask)
            # Returns Tensor [3, H, W] and Tensor [H, W] or [H, W, 1]
            return augmented['image'], augmented['mask'].long()
        except Exception as e:
            print(f"Error on index {idx}: {e}")
            return None, None

    # --- Methods required by Sampler/Stratification ---

    # --- Pickling Safety (Fix 3.1) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the file handle
        state['h5_file'] = None
        state['images_dset'] = None
        state['masks_dset'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Handle will be reopened on next __getitem__ call

    def get_labels(self):
        return self.labels

    def get_patient_ids(self):
        return self.patient_ids

    def get_class_counts(self):
        counts = np.bincount(self.labels)
        not_cancer_count = counts[0] if len(counts) > 0 else 0
        cancer_count = counts[1] if len(counts) > 1 else 0
        return {'CANCER': cancer_count, 'NOT_CANCER': not_cancer_count}

    def close(self):
        """Close any open HDF5 handle (safe to call multiple times)."""
        try:
            if getattr(self, "h5_file", None) is not None:
                try:
                    self.h5_file.close()
                except Exception:
                    pass
        finally:
            self.h5_file = None
            self.images_dset = None
            self.masks_dset = None
            self._opened_pid = None

    def __del__(self):
        # Best-effort cleanup (do not raise in GC).
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        """Make dataset picklable by dropping live HDF5 handles."""
        state = self.__dict__.copy()
        # Never pickle open file handles / datasets.
        state["h5_file"] = None
        state["images_dset"] = None
        state["masks_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure handles are closed until first access in this process.
        self.h5_file = None
        self.images_dset = None
        self.masks_dset = None
        self._opened_pid = None
        self._atexit_registered = False

class GPUNormalizer(torch.nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        # Register as buffers so they move to device automatically with the model
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, 3, H, W]
        """
        # CASE 1: Input is Byte/Uint8 (0-255).
        # This is what comes from ToTensorV2() with uint8 source.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # CASE 2: Input is Float, but large magnitude (0-255).
        # This happens if an augmentation inadvertently cast to float but didn't normalize.
        # We use a safe threshold. ImageNet data is never > 10.0 after normalization.
        elif x.max() > 2.0:
             x = x / 255.0

        # CASE 3: Input is already float [0, 1].
        # We do nothing.

        # Finally, normalize with Mean/Std
        return (x - self.mean) / self.std

class GPUDownscale(nn.Module):
    def __init__(self, p=0.07, scale_range=(0.5, 0.9)):
        # p=0.07 mimics the ~6.6% effective probability of the original setup
        super().__init__()
        self.p = p
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 3, H, W] FloatTensor

        # 1. Random Check
        if random.random() > self.p:
            return x

        B, C, H, W = x.shape

        # 2. Pick Scale
        scale = random.uniform(self.scale_min, self.scale_max)

        # 3. Downscale (Nearest Neighbor - simulates pixelation)
        # We perform this on the whole batch if triggered.
        # (Scientific Note: In A.OneOf, it was per-image.
        #  Doing it per-batch is much faster on GPU and statistically acceptable for LR Finder.
        #  If you strictly need per-image, we need a loop, but that slows it down slightly).

        # Let's stick to per-batch for maximum speed in LR Finder.
        h_small = int(H * scale)
        w_small = int(W * scale)

        # Down
        x_small = F.interpolate(x, size=(h_small, w_small), mode='nearest')

        # Up (Bicubic - simulates display)
        x_restored = F.interpolate(x_small, size=(H, W), mode='bicubic', align_corners=False)

        return x_restored

class SubsetView(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[int(self.indices[i])]

    # optional pass-through helpers if you rely on these
    def get_labels(self):
        labels = np.asarray(self.base_ds.get_labels())
        return labels[self.indices].tolist()

    def get_patient_ids(self):
        pids = np.asarray(self.base_ds.get_patient_ids())
        return pids[self.indices].tolist()

# --- Helper Functions ---
def get_formatted_datetime_string():
  now = datetime.now()
  return now.strftime("%d_%m_%Y_%H_%M_%S")

def clear_gpu():
    print("Clearing GPU cache...")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"[clear_gpu] Warning: {e}")
    print("GPU cache cleared.")

class EarlyStopping:
    """
    Early stops the training and saves only the single best model checkpoint,
    deleting the previous best to conserve disk space.
    """
    def __init__(self, patience=5, verbose=True, delta=0.0001, output_best_model_path='best_model.pth'):
        """
        Args:
            output_best_model_path (str): The fixed path where the single best model will be saved.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.delta = delta
        self.output_best_model_path = output_best_model_path # The fixed target path for saving
        self._current_best_checkpoint_on_disk_path = None # Tracks the path of the *actual* best file on disk

        os.makedirs(os.path.dirname(self.output_best_model_path), exist_ok=True)

    def set_initial_best_checkpoint_path(self, path):
        if path and os.path.exists(path):
            self._current_best_checkpoint_on_disk_path = path
            if self.verbose:
                print(f"EarlyStopping: Initial best checkpoint set to {os.path.basename(path)}")
        else:
            self._current_best_checkpoint_on_disk_path = None
            if self.verbose:
                print("EarlyStopping: No initial best checkpoint path provided/found.")

    def __call__(self, score, model, optimizer, epoch, val_loss, val_auprc, val_mcc_star, val_auroc):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, val_auprc, val_mcc_star, score, val_auroc)
            return True

        improvement_detected = False
        if score > self.best_score + self.delta:
            improvement_detected = True

        if improvement_detected:
            previous_best_score_for_log = self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, val_auprc, val_mcc_star, score, val_auroc, previous_best_score_for_log)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best score: {self.best_score:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True

        return improvement_detected

    def save_checkpoint(self, val_loss, model, optimizer, epoch, val_auprc, val_mcc_star, score, val_auroc, previous_best_score_for_log=None):
        if self.verbose:
            if previous_best_score_for_log is None:
                print(f'Initial best score: {score:.6f}. Saving model...')

        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_score': score, # This is AUPRC
            'val_auprc': val_auprc,
            'val_auroc': val_auroc,
            'val_mcc_star': val_mcc_star,
            'is_compiled': hasattr(model, '_orig_mod')
        }

        try:
            torch.save(save_dict, self.output_best_model_path)
            if self.verbose:
                print(f"  New best model saved to: {self.output_best_model_path}")

            self._current_best_checkpoint_on_disk_path = self.output_best_model_path

        except Exception as e:
            print(f"Error saving best model checkpoint to {self.output_best_model_path}: {e}")

def create_stratified_subset_within_patients(full_dataset, ratio, split_name="Unknown"):
    """
    Creates a scientifically robust, stratified random subsample of a dataset.

    This function performs a two-level stratification:
    1. It groups all patches by their source patient.
    2. Within each patient, it further groups patches by class (Cancer/Not Cancer).
    3. It then samples the specified ratio from each of these sub-groups.

    This ensures the final subset precisely preserves the class proportions
    within each patient from the original full dataset.
    """
    print(f"  Creating a {ratio:.0%} two-level stratified subsample for {split_name.upper()} (by patient and class)...")

    indices_by_patient_and_class = defaultdict(lambda: defaultdict(list))
    for i in range(len(full_dataset)):
        # These attributes must exist on the dataset object
        patient_id = full_dataset.patient_ids[i]
        label = full_dataset.labels[i]
        indices_by_patient_and_class[patient_id][label].append(i)

    print(f"    Found {len(indices_by_patient_and_class)} unique patients in the {split_name} split.")

    subset_indices = []
    generator = torch.Generator().manual_seed(SEED)

    for patient_id, class_groups in indices_by_patient_and_class.items():
        for label, indices in class_groups.items():
            num_to_sample = int(np.ceil(len(indices) * ratio))
            shuffled_indices = torch.randperm(len(indices), generator=generator).tolist()
            sampled_local_indices = shuffled_indices[:num_to_sample]
            subset_indices.extend([indices[i] for i in sampled_local_indices])

    # Calculate and print the "after" counts for verification
    subset_labels = [full_dataset.labels[i] for i in subset_indices]
    if subset_labels:
        subset_counts = np.bincount(subset_labels)
        not_cancer_count = subset_counts[0] if len(subset_counts) > 0 else 0
        cancer_count = subset_counts[1] if len(subset_counts) > 1 else 0
    else:
        not_cancer_count, cancer_count = 0, 0

    print(f"    {split_name.title()} subset class counts -> CANCER: {cancer_count}, NOT_CANCER: {not_cancer_count}")

    return Subset(full_dataset, subset_indices)

def _set_matmul_precision_for_arch(architecture: str) -> str:
    """
    Controls TF32 behavior for float32 matmul kernels on Ampere+ GPUs.
    This is independent from AMP autocast dtype.
    Returns the chosen matmul precision string for logging.
    """
    arch = (architecture or "").upper()

    # Default: safest "high" (TF32 enabled on Ampere when allowed by PyTorch settings)
    matmul_precision = "high"

    # If you have a known numerically sensitive model, keep it strict.
    # Note: this affects matmul kernels, not convolution kernels.
    if arch == "DPT":
        matmul_precision = "high"  # keep "high"; AMP will be forced to fp32 below anyway
    elif arch in {"SWIN", "SEGFORMER"}:
        # Transformers often do fine with "medium" and can benefit from it.
        matmul_precision = "medium"
    else:
        matmul_precision = "high"

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        # Older torch versions may not support it; safe to ignore
        pass

    return matmul_precision


def _resolve_amp_precision(
    amp_precision: str,
    architecture: str,
    prefer_bf16_if_available: bool = True
) -> Tuple[torch.dtype, Optional[torch.cuda.amp.GradScaler], Dict[str, str]]:
    """
    Decide AMP autocast dtype + scaler based on explicit user choice.
    Returns (amp_dtype, scaler, log_dict).
    """
    arch = (architecture or "").upper()
    choice = (amp_precision or "auto").lower()

    # CPU fallback
    if not torch.cuda.is_available():
        return torch.float32, None, {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "float32",
            "amp_reason": "cpu_no_cuda",
        }

    # If you have models you *must* keep in fp32 for stability, enforce here.
    if arch == "DPT":
        return torch.float32, None, {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "float32",
            "amp_reason": "arch_forced_fp32",
        }

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    # NOTE: is_bf16_supported exists on modern torch; if not, bf16_supported=False.

    if choice == "fp32":
        return torch.float32, None, {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "float32",
            "amp_reason": "user_forced_fp32",
        }

    if choice == "bf16":
        if not bf16_supported:
            # Fail fast: don't silently downgrade, reviewers hate that.
            raise RuntimeError("AMP_PRECISION='bf16' requested but CUDA BF16 is not supported on this GPU/torch build.")
        return torch.bfloat16, None, {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "bfloat16",
            "amp_reason": "user_forced_bf16",
        }

    if choice == "fp16":
        # FP16 typically needs scaling for stability.
        return torch.float16, torch.cuda.amp.GradScaler(), {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "float16",
            "amp_reason": "user_forced_fp16",
        }

    if choice == "auto":
        # Auto policy: prefer BF16 if supported, else FP16+scaler.
        if prefer_bf16_if_available and bf16_supported:
            return torch.bfloat16, None, {
                "amp_precision_requested": choice,
                "amp_dtype_effective": "bfloat16",
                "amp_reason": "auto_prefer_bf16_supported",
            }
        return torch.float16, torch.cuda.amp.GradScaler(), {
            "amp_precision_requested": choice,
            "amp_dtype_effective": "float16",
            "amp_reason": "auto_fallback_fp16",
        }

    raise ValueError(f"Invalid AMP_PRECISION='{amp_precision}'. Use one of: fp32, bf16, fp16, auto.")


def setup_precision(
    architecture: str,
    amp_precision: str = "auto",
) -> Tuple[torch.dtype, Optional[torch.cuda.amp.GradScaler], Dict[str, str]]:
    """
    Unified entry point:
      - sets matmul precision policy (TF32 control) per architecture
      - selects AMP autocast dtype per explicit configuration
    Returns:
      (amp_dtype, scaler, log_dict)
    """
    matmul_precision = _set_matmul_precision_for_arch(architecture)
    amp_dtype, scaler, amp_log = _resolve_amp_precision(amp_precision, architecture)

    log = {
        "architecture": (architecture or "").upper(),
        "matmul_precision": matmul_precision,
        **amp_log,
    }
    return amp_dtype, scaler, log

def train_model(model, optimizer, dataloader, device, current_epoch, loss_fn, health, architecture):
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()

    amp_dtype, scaler, prec_log = setup_precision(architecture, amp_precision=AMP_PRECISION)

    global AMP_LOG
    AMP_LOG = prec_log

    tracker_loss = RunningWeightedMetric()

    # Initialize gradients once before loop
    optimizer.zero_grad(set_to_none=True)

    # Counter for successful micro-batches
    current_accumulation_steps = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train E{current_epoch+1}", leave=False, mininterval=10.0)

    for batch_idx, batch_data in pbar:
        # --- Guardrails (Skips do not increment accumulation) ---
        if batch_data is None:
            health.train_skip("dataloader_none_batch")
            continue
        try:
            images, masks = batch_data
        except Exception:
            health.train_skip("unpack_failed")
            continue
        if images is None or masks is None:
            health.train_skip("images_or_masks_none")
            continue

        # Check batch size 0
        if images.shape[0] == 0:
            health.train_skip("batch_size_zero")
            continue

        # Move to device
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)

        images = gpu_normalizer(images) # Becomes Float32 here
        images = gpu_downscale(images)

        # --- Target canonicalization (keep class indices; one-hot happens inside the loss) ---
        if masks.ndim == 4 and masks.size(1) == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 4 and masks.size(-1) == 1:
            masks = masks[..., 0]

        if masks.ndim != 3:
            health.train_skip("mask_bad_shape")
            continue


        # --- Forward ---
        ctx = autocast_ctx(images, amp_dtype)
        with ctx:
            outputs_raw = model(images)

            # Output Guardrails
            if isinstance(outputs_raw, (tuple, list)):
                if len(outputs_raw) == 0:
                    health.train_skip("model_empty_tuple")
                    continue
                outputs = outputs_raw[0]
            else:
                outputs = outputs_raw

            if not isinstance(outputs, torch.Tensor):
                health.train_skip("model_output_not_tensor")
                continue

            # Shape mismatch check
            # Shape guardrails (post one-hot removal)
            # masks: [B,H,W] (class indices), outputs: [B,C,H,W]
            if not hasattr(masks, "shape") or masks.ndim != 3 or outputs.ndim != 4:
                health.train_skip("bad_tensor_rank")
                continue

            if outputs.shape[-2:] != masks.shape[-2:]:
                health.train_skip("spatial_mismatch")
                continue

            # This loss expects 2-class logits (softmax over C=2)
            if outputs.shape[1] != 2:
                health.train_skip("channel_mismatch")
                continue

            loss = loss_fn(outputs, masks)

        # NaN/Inf check
        if not torch.isfinite(loss):
            health.train_naninf_loss()
            health.train_skip("naninf_loss")
            # Do NOT increment accumulation, do NOT step.
            continue

        # --- SUCCESSFUL FORWARD PASS ---

        # Normalize loss for accumulation
        loss = loss / ACCUMULATION_STEPS

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update metrics (undo normalization for logging)
        real_loss_val = loss.item() * ACCUMULATION_STEPS
        tracker_loss.update(real_loss_val * images.shape[0], images.shape[0])

        # Increment logical batch counter
        current_accumulation_steps += 1

        # --- OPTIMIZER STEP (Only when we hit the target accumulation) ---
        if current_accumulation_steps % ACCUMULATION_STEPS == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Reset
            optimizer.zero_grad(set_to_none=True)
            current_accumulation_steps = 0

            # Update UI only on step
            pbar.set_postfix(loss=f"{tracker_loss.get_average():.4f}")

    # Note: We intentionally drop partial gradients at the end of epoch
    # if current_accumulation_steps > 0. This is statistically safer
    # than stepping with a "light" batch which can have high variance.

    return tracker_loss.get_average()

def validate_model(model, optimizer, dataloader, device, loss_fn, health, architecture):
    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()

    # Precision setup (match training)
    amp_dtype, _, _ = setup_precision(architecture, amp_precision=AMP_PRECISION)

    tracker = AdvancedMetricTracker(device=device, metric_bins=2048)

    running_loss = 0.0
    num_samples_processed = 0

    pbar = tqdm(dataloader, desc="Validate", leave=False)

    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(pbar):
            # --- Guardrail 1: DataLoader yielded None ---
            if batch_data is None:
                health.val_skip("dataloader_none_batch")
                continue

            # --- Guardrail 2: Unpack defensively ---
            try:
                images, masks = batch_data
            except Exception as e:
                health.val_skip("unpack_failed")
                continue

            # --- Guardrail 3: None images/masks ---
            if images is None or masks is None:
                health.val_skip("images_or_masks_none")
                continue

            # --- Guardrail 4: Empty batch / invalid batch size ---
            try:
                bsz = images.size(0)
            except Exception as e:
                health.val_skip("images_no_batch_dim")
                continue

            if bsz == 0:
                health.val_skip("batch_size_zero")
                continue

            # --- OPTIMIZATION 3: Channels Last for Val Images ---
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            images = gpu_normalizer(images) # Becomes Float32 here


            # --- OPTIMIZATION 2.1: On-GPU One-Hot Encoding for Val ---
            masks = masks.to(device, non_blocking=True, dtype=torch.long)


            # --- Forward + loss with AMP ---
            ctx = autocast_ctx(images, amp_dtype)
            with ctx:
                outputs_raw = model(images)

                # --- Guardrail 5: Handle tuple/list output (compiled models, etc.) ---
                if isinstance(outputs_raw, (tuple, list)):
                    if len(outputs_raw) == 0:
                        health.val_skip("model_empty_tuple")
                        continue
                    outputs = outputs_raw[0]
                else:
                    outputs = outputs_raw

                # --- Guardrail 6: Output must be a tensor ---
                if not isinstance(outputs, torch.Tensor):
                    health.val_skip("model_output_not_tensor")
                    continue


                # --- Guardrail 7: Shape mismatch ---
                # optional canonicalization if your val masks can come with singleton channel
                if masks.ndim == 4 and masks.size(1) == 1:
                    masks = masks[:, 0, :, :]
                elif masks.ndim == 4 and masks.size(-1) == 1:
                    masks = masks[..., 0]

                if masks.ndim != 3:
                    health.val_skip("mask_bad_shape")
                    continue

                # shape checks consistent with training
                if outputs.ndim != 4 or outputs.shape[1] != 2:
                    health.val_skip("channel_mismatch")
                    continue
                if outputs.shape[-2:] != masks.shape[-2:]:
                    health.val_skip("spatial_mismatch")
                    continue

                #add
                loss = loss_fn(outputs, masks)

            # --- Guardrail 8: NaN/Inf loss ---
            if not torch.isfinite(loss):
                health.val_naninf_loss()
                health.val_skip("naninf_loss")
                continue

            # --- Accumulate loss (sample-weighted) ---
            batch_loss = loss.item()
            running_loss += batch_loss * bsz
            num_samples_processed += bsz

            tracker.update(outputs, masks)

            # Pbar update (loss only; metrics at the end)
            if num_samples_processed > 0:
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    avg_loss=f"{running_loss / num_samples_processed:.4f}",
                )

    # End of epoch aggregation
    try:
      if num_samples_processed == 0:
        health.val_skip("no_samples_processed")
        return 0.0, None

      epoch_loss = running_loss / num_samples_processed
      epoch_results = tracker.compute_and_reset(health=health)

      return epoch_loss, epoch_results

    finally:
      del tracker

# --- Update create_email_body to include AUC ---
def create_email_body(checkpoint_path, encoder, architecture):
    body = f'Training {architecture} finished.\n\nCheckpoint Path: {checkpoint_path}\n\n--- ENCODER: {encoder} ---\n'
    return body

def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body); msg['Subject'] = subject; msg['From'] = sender; msg['To'] = ', '.join(recipients)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server: smtp_server.login(sender, password); smtp_server.sendmail(sender, recipients, msg.as_string())
        print("Email sent successfully!")
    except Exception as e: print(f"Error sending email: {e}")

# --- 12. Aim & Ngrok Setup ---
print("Setting up Aim repository...")
# --- Aim Setup ---
from aim import Run
import subprocess
import aim
if not os.path.exists(AIM_REPO_PATH):
  print(f"Aim repository not found at {AIM_REPO_PATH}. Initializing...")
  os.makedirs(AIM_REPO_PATH, exist_ok=True)
  try:
    repo = aim.Repo.init(AIM_REPO_PATH);
    print(f"Aim repository initialized at: {repo.path}")
  except Exception as e:
    print(f"An unexpected error occurred during Aim repo setup: {e}")
else:
  print(f"Using existing Aim repository at: {AIM_REPO_PATH}")
print("Aim/Ngrok setup complete.")

def save_metadata(best_val_score, checkpoint, encoder, architecture,
                  metadata_best_path, val_loss, val_mcc, val_auroc):

    # --- SAVE METADATA ---
    # Construct filename based on the checkpoint name
    meta_filename = os.path.join(METADATA_DIR, os.path.basename(metadata_best_path))
    meta_filename = meta_filename.replace(".pth", "_meta.json")

    metadata = {
        # General Best Metrics
        "best_val_auprc_pixel_score": best_val_score,
        "pixel_val_loss": val_loss,
        "pixel_val_mcc": val_mcc,
        "pixel_val_auroc": val_auroc,

        # --- SECTION 2: MODEL ARTIFACTS ---
        "best_model_epoch": checkpoint.get('epoch', '?'),
        "checkpoint_path": metadata_best_path,
        "encoder": encoder,
        "architecture": architecture,

        # --- SECTION 3: REPRODUCIBILITY CONFIGURATION ---
        "hyperparameters": {
            "amp_precision": AMP_LOG,
            "Learning_rate": BASE_LEARNING_RATE,
            "Weight_Decay": WEIGHT_DECAY,
            "Batch_Size": BATCH_SIZE,
            "Num_Epochs": NUM_EPOCHS,
            "Workers": WORKERS,
            "Seed": SEED,
            "Dataset": HDF5_DRIVE_DIR,
            "Patience": PATIENCE,
            "Optimizer": OPTIMIZER_NAME,
            "Loss_Function": "BCEDiceHybrid",
            "Loss_Weights": {
                "alpha_bce": ALPHA_BCE,
                "beta_dice_bg": BETA_DICE_BG,
                "gamma_dice_fg": GAMMA_DICE_FG
            }
        }
    }

    try:
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to: {meta_filename}")
    except Exception as e:
        print(f"Error saving metadata file {meta_filename}: {e}")

def get_model(architecture, encoder, validation=False):

    # Validation optimization: No need to download ImageNet weights if loading a checkpoint
    encoder_weights = None if validation else "imagenet"

    # Common args for all architectures
    common_args = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,        # Correct for your 2-channel One-Hot masks
        activation=None,  # Correct for BCEDiceHybridLoss (expects logits)
    )

    arch = architecture.upper()

    if arch == "SWIN":
        # Swin works best as a backbone for U-Net or UPerNet
        model = smp.Unet(**common_args, decoder_attention_type=None)

    elif arch == "DEEPLABV3PLUS":
        model = smp.DeepLabV3Plus(**common_args, decoder_attention_type=None)

    elif arch == "DPT":
        # DPT specific: Ignore the depth readout, strictly segmentation
        model = smp.DPT(**common_args, decoder_readout='ignore')

    elif arch == "UNET++":
        model = smp.UnetPlusPlus(**common_args, decoder_attention_type=None)

    elif arch == "FPN":
        model = smp.FPN(**common_args, decoder_attention_type=None)

    elif arch == "SEGFORMER":
        model = smp.Segformer(**common_args, decoder_attention_type=None)

    elif arch == "MANET":
        model = smp.MAnet(**common_args, decoder_attention_type=None)

    elif arch == "UPERNET":
        model = smp.UPerNet(**common_args, decoder_attention_type=None)

    elif arch == "INCEPTIONRESNETV2":
         model = smp.Unet(**common_args, decoder_attention_type=None)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model

def load_checkpoint_for_resume(model, optimizer, early_stopping, checkpoint_path, device):
    """
    Loads a training checkpoint (if provided) and returns the starting epoch index
    for the outer training loop. It also informs EarlyStopping about the initial best path.
    """
    start_epoch = 0

    if not checkpoint_path:
        print("No resume checkpoint provided: training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    checkpoint_path = checkpoint_path.strip()
    if checkpoint_path == "":
        print("Empty resume checkpoint string: training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    if not os.path.exists(checkpoint_path):
        print(f"Resume checkpoint not found at {checkpoint_path}. Training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    print(f"\n*** Resuming from checkpoint: {checkpoint_path}")
    chkpt = torch.load(checkpoint_path, map_location=device)

    state_dict = chkpt.get("model_state_dict", None)
    if state_dict is None:
        print("Checkpoint missing 'model_state_dict'. Training from scratch.")
        early_stopping.set_initial_best_checkpoint_path(None)
        return start_epoch

    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    if "optimizer_state_dict" in chkpt:
        try:
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            print("Optimizer state loaded from checkpoint.")
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")

    best_score = chkpt.get("best_val_score", None)
    if best_score is not None:
        early_stopping.best_score = best_score
        # previous_best_score is not needed for EarlyStopping logic, only for print messages
        # early_stopping.previous_best_score = best_score # Removed as it's not strictly part of ES logic
        early_stopping.counter = 0
        print(f"Loaded EarlyStopping best_val_score = {best_score:.6f}")


    # This is the "best checkpoint on disk" initially.
    early_stopping.set_initial_best_checkpoint_path(checkpoint_path)

    last_epoch = int(chkpt.get("epoch", 0))
    start_epoch = last_epoch
    print(f"Last finished epoch in checkpoint: {last_epoch}. "
          f"Next epoch will be {last_epoch + 1}.\n")

    return start_epoch

def define_optimizer():
  if OPTIMIZER_NAME == "AdamWScheduleFree":
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)
  elif OPTIMIZER_NAME == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)
  else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")
  return optimizer

def get_previous_metrics(chkpt,
                         best_val_score_for_architecture_auprc,
                         best_val_score_for_architecture_mcc,
                         best_val_score_for_architecture_auroc,
                         best_val_score_for_architecture_loss):
  best_val_score_for_architecture_auprc = chkpt.get('val_auprc', chkpt.get('best_val_score'))
  best_val_score_for_architecture_mcc = chkpt.get('val_mcc_star',0.0)
  best_val_score_for_architecture_auroc = chkpt.get('val_auroc', 0.0)
  best_val_score_for_architecture_loss = chkpt.get('val_loss', 0.0)

  print(f"""
  Restored best metrics from checkpoint: VAL_AUPRC={best_val_score_for_architecture_auprc:.4f},
  VAL_AUROC={best_val_score_for_architecture_auroc:.4f},
  VAL_MCC={best_val_score_for_architecture_mcc:.4f},
  VAL_LOSS={best_val_score_for_architecture_loss:.4f}
  """)

  return best_val_score_for_architecture_auprc, best_val_score_for_architecture_mcc, best_val_score_for_architecture_auroc, best_val_score_for_architecture_loss

def collate_fn(batch):
  batch = [x for x in batch if x is not None and x[0] is not None and x[1] is not None]
  return torch.utils.data.dataloader.default_collate(batch) if batch else None

def get_filename(drive_dir):
  original_src=os.path.join(drive_dir, "TRAIN.h5")
  filtered_src=os.path.join(drive_dir, "TRAIN_FILTERED.h5")
  filename = original_src
  if os.path.exists(drive_dir):
    if os.path.exists(original_src) and os.path.exists(filtered_src):
      if SMART_SAMPLING:
        filename = filtered_src
  else:
    raise FileNotFoundError(f"Missing {drive_dir}")
  return filename

def setup_local_hdf5(drive_dir, local_dir):
    """
    Copies pre-processed HDF5 files from Google Drive to local VM for speed.
    """
    print(f"\n{'='*25} Setting up HDF5 Data {'='*25}")

    # Ensure local directory exists and is clean
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    required_files = ["VALIDATION.h5"]

    # Handle path resolution (using your existing change_paths logic)
    resolved_drive_dir = change_paths(drive_dir)
    chosen_file_name = str(os.path.basename(get_filename(resolved_drive_dir)))
    required_files.append(chosen_file_name)

    print(f"Source Directory: {resolved_drive_dir}")

    start_time = time.time()
    for filename in required_files:
        src = os.path.join(resolved_drive_dir, filename)
        dst = os.path.join(local_dir, filename)

        if not os.path.exists(src):
            raise FileNotFoundError(f"Critical data missing: {src}")

        print(f"Copying {filename} to local disk...")
        shutil.copy2(src, dst)

        # Verify file size to ensure copy integrity
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        if src_size != dst_size:
            raise RuntimeError(f"Copy failed for {filename}: Size mismatch.")

    elapsed = time.time() - start_time
    print(f"Data transfer complete in {elapsed:.2f} seconds.")
    return chosen_file_name

# ==============================================================================
# --- 13. Main Training Loop ---
# ==============================================================================

# 1. Transfer HDF5 files from Drive to Local
chosen_file_name = setup_local_hdf5(HDF5_DRIVE_DIR, LOCAL_DATA_DIR)

# Define paths to the local copies
train_h5_path = os.path.join(LOCAL_DATA_DIR, chosen_file_name)
val_h5_path = os.path.join(LOCAL_DATA_DIR, "VALIDATION.h5")

print("\nCreating DataLoaders...")
try:
    # A) Full HDF5 Wrappers
    # Note: Ensure your preprocessing script included "patient_ids" in the HDF5
    # for the leakage check below to function.
    full_train_ds_h5 = HybridProstateDataset(train_h5_path, mode="train")
    full_val_ds_h5   = ProstateCancerDatasetHDF5(val_h5_path, mode="val")

    # B) Patient Leakage Check (Critical for scientific validity)
    print("Verifying data integrity...")
    train_pids = set(full_train_ds_h5.get_patient_ids())
    val_pids   = set(full_val_ds_h5.get_patient_ids())
    intersection = train_pids.intersection(val_pids)

    print(f"Patient Check: Train={len(train_pids)}, Val={len(val_pids)}, Intersection={len(intersection)}")
    if len(intersection) > 0:
        raise ValueError(f"CRITICAL DATA LEAKAGE: Patients {intersection} found in both Train and Val!")
    else:
        print("✅ Patient Separation Verified.")

    # C) Subsetting (Patient-Stratified)
    if USE_SUBSET and SUBSET_RATIO < 1.0:
        print(f"Subsampling enabled: {SUBSET_RATIO:.0%}")
        train_sub = create_stratified_subset_within_patients(full_train_ds_h5, SUBSET_RATIO, "train")
        val_sub   = create_stratified_subset_within_patients(full_val_ds_h5, SUBSET_RATIO, "val")

        train_ds = SubsetView(full_train_ds_h5, train_sub.indices)  # keeps RAM cache if enabled
        val_ds   = SubsetView(full_val_ds_h5,   val_sub.indices)    # still disk-backed; fine
    else:
        train_ds = full_train_ds_h5
        val_ds   = full_val_ds_h5

    # ---------------------------------------------------------
    # 3) IMPLEMENT WEIGHTED RANDOM SAMPLER (Unchanged)
    # ---------------------------------------------------------
    print("Calculating sampler weights...")

    # .get_labels() works on the HDF5 dataset just like the old one
    current_labels = np.array(train_ds.get_labels())

    # Calculate class weights (Inverse Frequency)
    class_counts = np.bincount(current_labels)
    class_counts[class_counts == 0] = 1

    weight_per_class = 1.0 / class_counts
    samples_weights = weight_per_class[current_labels]

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(SEED)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(samples_weights).double(),
        num_samples=len(samples_weights),
        replacement=True,
        generator=sampler_generator
    )

    print(f"Sampler Ready.")

    # 4) Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(WORKERS>0),
        prefetch_factor=4 if WORKERS>0 else None,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=WORKERS>0,
        prefetch_factor=4 if WORKERS>0 else None,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )

    print("DataLoaders created successfully (Train: HDF5, Val: PNG).")

except Exception as e:
  print(f"DataLoader Err: {e}")
  import traceback
  traceback.print_exc()
  clear_gpu()

# # # The new, refactored lists
LIST_ARCH = ['SWIN',
             'DEEPLABV3PLUS',
             'UNET++',
             'FPN',
             'SEGFORMER',
             'MANET',
             'DPT',
             'UPERNET'
             ]

LIST_ENCODER = [
     'tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k', # For Unet (acting as our new Swin model)
     'tu-resnest101e',                                    # For DEEPLABV3PLUS
     'efficientnet-b7',                                   # For UNET++
     'senet154',                                          # For FPN
     'mit_b5',                                            # For SEGFORMER
     'resnet152',                                         # For MANET
     'tu-vit_large_patch16_224.augreg_in21k_ft_in1k',     # For DPT
     'tu-hiera_large_224'                                 # For UPerNet
 ]


LIST_CHECKPOINT = [
     '',                                                 # For Unet (acting as our new Swin model)
     '',                                                 # For DEEPLABV3PLUS
     '',                                                 # For UNET++
     '',                                                 # For FPN
     '',                                                 # For SEGFORMER
     '',                                                 # For MANET
     '',                                                 # For DPT
     ''                                                  # For UPerNet
 ]

LIST_ARCH = ['FPN']

LIST_ENCODER = ['senet154']

LIST_CHECKPOINT = ['']

# Reviewer Fix: Assert lengths match
assert len(LIST_ARCH) == len(LIST_ENCODER) == len(LIST_CHECKPOINT), \
    "Configuration lists (ARCH, ENCODER, CHECKPOINT) must have the same length!"

# Instantiate globally
gpu_normalizer = GPUNormalizer(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    device=device
)

gpu_downscale = GPUDownscale(p=0.07).to(device)

for architecture, encoder, resume_checkpoint_path in zip(LIST_ARCH, LIST_ENCODER, LIST_CHECKPOINT):

  best_val_score_for_architecture_auprc = None
  best_val_score_for_architecture_mcc = None
  best_val_score_for_architecture_auroc = None
  best_val_score_for_architecture_loss = None

  health = TrainingHealthTracker(name=f"{architecture}_{encoder}")
  BASE_LEARNING_RATE,WEIGHT_DECAY  = get_learning_rate(architecture)

  # --- Init Aim Run ---
  experiment_name = f"{IDENTIFIER}_{architecture}_{encoder}_{get_formatted_datetime_string()}"
  print(f"Init Aim: {experiment_name}")
  run = None
  try:
      run = Run(experiment=experiment_name, repo=AIM_REPO_PATH)
      run["hparams"] = {
          "base_learning_rate": BASE_LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
          "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,"workers": WORKERS, "seed": SEED,"label": experiment_name,
          "optimizer": OPTIMIZER_NAME,"loss": "BCEDiceHybridLossPaper","model": f"{architecture}_{encoder}",
          "encoder_weights": "IMAGENET", "patience": PATIENCE,
          "train_len":len(train_ds),"val_len":len(val_ds),
          "loss_alpha_bce": ALPHA_BCE, "loss_beta_dice_bg": BETA_DICE_BG, "loss_gamma_dice_fg": GAMMA_DICE_FG,
      }
      print("Aim run initialized.")
  except Exception as e:
    print(f"Aim Init Err: {e}.")

  print(f"Architecture: {architecture}")
  # --- Initialize Model, Optimizer, Early Stopping ---
  print("Initializing SMP Model, Optimizer, ES...")

  # --- Instantiate SMP Model ---
  try:
    model = get_model(architecture = architecture, encoder = encoder, validation = False)
    model.to(device);
  except Exception as e:
    print(f"Model init err: {e}")
    raise ValueError(f"Unknown architecture: {architecture}")

  # --- Compile Model ---
  try:
    model = torch.compile(model);
    print("Model compiled.")
  except Exception as e:
    print(f"Compile failed: {e}.")

  print(f"Model->{device}")

  optimizer = define_optimizer()

  print(f"Optimizer initialized with {OPTIMIZER_NAME}")

  output_best_model_path_for_this_run = os.path.join(CHECKPOINT_PATH, f'BEST_MODEL_{experiment_name}.pth')

  # Initialize EarlyStopping with this fixed path
  early_stopping = EarlyStopping(
      patience=int(PATIENCE),
      verbose=True,
      delta=0.0001,
      output_best_model_path=output_best_model_path_for_this_run)

  full_resume_checkpoint_path = os.path.join(CHECKPOINT_PATH, resume_checkpoint_path) if resume_checkpoint_path else None

  start_epoch = load_checkpoint_for_resume(
      model=model,
      optimizer=optimizer,
      early_stopping=early_stopping,
      checkpoint_path=full_resume_checkpoint_path,
      device=device)

  if early_stopping._current_best_checkpoint_on_disk_path:
      # If we successfully loaded a resume checkpoint, that's our initial best.
      metadata_best_path = early_stopping._current_best_checkpoint_on_disk_path
  else:
      # If training from scratch, this will be the first best model saved.
      metadata_best_path = output_best_model_path_for_this_run


  print(f"Starting Training For {architecture} from epoch {start_epoch + 1}...")

  loss_fn = BCEDiceHybridLossPaper(
      alpha=ALPHA_BCE,
      beta=BETA_DICE_BG,
      gamma=GAMMA_DICE_FG
      )
  print(
      f"Using BCE+Dice Hybrid Loss "
      f"(alpha={ALPHA_BCE}, beta={BETA_DICE_BG}, gamma={GAMMA_DICE_FG})"
      )

  training_successful = True
  for epoch in range(start_epoch, NUM_EPOCHS):
      health.reset_epoch()
      current_epoch_num = epoch + 1
      epoch_start = time.time()

      try:
        t_loss = train_model(model, optimizer, train_loader, device, epoch, loss_fn, health, architecture)
      except Exception as e:
        print(f"\nTrain Err E{current_epoch_num}:{e}")
        training_successful=False
        break

      try:
          v_loss, v_results = validate_model(model, optimizer, val_loader, device, loss_fn, health, architecture)

          # 1. Check for Model Collapse (if tracker returned None)
          if v_results is None:
              if health.epoch["val_collapsed"] == 1:
                  reason = "COLLAPSE"
              elif health.epoch["val_invalid_metrics"] == 1:
                  reason = "INVALID_METRICS"
              else:
                  reason = "UNKNOWN"
              print(f"\n[Epoch {current_epoch_num}] Validation failed ({reason}). Skipping checkpoint.")

              # --- NEW: EMERGENCY BRAKE CHECK ---
              if health.should_emergency_stop():
                  print(f"\n!!! EMERGENCY STOP !!!")
                  print(f"Model collapsed {health.current_consecutive_collapses} times in a row.")
                  print("Terminating training to save resources.")
                  training_successful = False # Mark as failed
                  break # Break the epoch loop

              health.log_epoch(current_epoch_num)
              continue

          # --- NEW: RESET COUNTER ON SUCCESS ---
          # If we got here, v_results is valid. The model is healthy.
          health.reset_collapse_counter()

          # 2. Unpack the robust metrics
          val_auprc = v_results['val_auprc']
          val_auroc = v_results['val_auroc']
          val_mcc_star = v_results['val_mcc_star']

          # DECISION RULE:
          # Use AUPRC as the score to maximize.
          # (Since AUPRC is highly correlated with segmentation quality in imbalance)
          score = val_auprc
      except Exception as e:
          print(f"\nVal Err E{current_epoch_num}: {e}")
          traceback.print_exc()
          training_successful = False
          break

      # --- Logging ---
      epoch_dur=time.time()-epoch_start
      mins,secs=divmod(epoch_dur,60)
      print(
        f"\nE{current_epoch_num}/{NUM_EPOCHS} [{int(mins):02d}m{int(secs):02d}s] "
        f"Tr L:{t_loss:.4f}|"
        f"Val AUPRC:{val_auprc:.4f} AUROC:{val_auroc:.4f} MCC*:{val_mcc_star:.4f}"
        )

      if run:
          try:
              run.track(t_loss, 'loss', epoch=current_epoch_num, context={"subset": "train"})
              run.track(val_auprc,  'pixel_auprc', epoch=current_epoch_num, context={"subset": "val"})
              run.track(val_auroc,  'pixel_auroc', epoch=current_epoch_num, context={"subset": "val"})
              run.track(val_mcc_star,   'pixel_mcc',  epoch=current_epoch_num, context={"subset": "val"})
          except Exception as e:
              print(f"Aim Log Err: {e}")

      try:
          # We use AUPRC as the primary score for Early Stopping
          improvement_detected = early_stopping(score, model, optimizer, current_epoch_num, v_loss, val_auprc, val_mcc_star, val_auroc)

          if improvement_detected:
              # If EarlyStopping saved a NEW best, then metadata_best_path should reflect this fixed path.
              # early_stopping._current_best_checkpoint_on_disk_path will already be updated inside ES.
              best_val_score_for_architecture_auprc = val_auprc
              best_val_score_for_architecture_mcc = val_mcc_star
              best_val_score_for_architecture_auroc = val_auroc
              best_val_score_for_architecture_loss = v_loss
              metadata_best_path = early_stopping.output_best_model_path

              print(f"  >>> New Best Model! (AUPRC: {val_auprc:.4f})")
      except Exception as e:
          print(f"ES/Save Err: {e}")

      if not UNLEASHED and early_stopping.early_stop:
        print(f"Early stopping E{current_epoch_num}.")
        break

      print(f"Epoch {epoch} completed.")
      health.log_epoch(current_epoch_num)


# --- Post-Training for Fold ---
  if not training_successful:
      print(f"Train loop stopped early for {architecture} (Emergency Stop).")
      print("Skipping threshold tuning for this architecture due to instability.")
      # Clear GPU and move to the next architecture in the list
      del model, optimizer, loss_fn
      clear_gpu()
      continue  # <--- SKIPS REST OF LOOP, GOES TO NEXT ARCHITECTURE

  print(f"\nTrain loop finished successfully for {architecture}.")
  clear_gpu()

  # --- Load Best Model (based on Val Loss) ---
  print("Loading best model for threshold tuning...")

  # The EarlyStopping instance now holds the definitive path to the best checkpoint on disk.
  final_best_checkpoint_path = early_stopping._current_best_checkpoint_on_disk_path

  if final_best_checkpoint_path and os.path.exists(final_best_checkpoint_path):
      best_model_path = final_best_checkpoint_path
  else:
      best_model_path = full_resume_checkpoint_path

  try:
    if os.path.exists(best_model_path):
        chkpt=torch.load(best_model_path,map_location=device)
        (best_val_score_for_architecture_auprc,
         best_val_score_for_architecture_mcc,
         best_val_score_for_architecture_auroc,
         best_val_score_for_architecture_loss) = get_previous_metrics(chkpt,
                         best_val_score_for_architecture_auprc,
                         best_val_score_for_architecture_mcc,
                         best_val_score_for_architecture_auroc,
                         best_val_score_for_architecture_loss)

        save_metadata(
            best_val_score=best_val_score_for_architecture_auprc,
            checkpoint=chkpt,
            encoder=encoder,
            architecture=architecture,
            metadata_best_path=best_model_path,
            val_loss=best_val_score_for_architecture_loss,
            val_mcc=best_val_score_for_architecture_mcc,
            val_auroc=best_val_score_for_architecture_auroc)


        print("Emailing...");
        subject = f"Training {architecture} finished"
        body=create_email_body(metadata_best_path, encoder, architecture);
        send_email(f"Finished: {experiment_name}",body,sender,recipients,password)
    else:
      print(f"Best model not found: {best_model_path}. Skip test.")

  except Exception as e:
    print(f"Test/Visu Err: {e}")
    traceback.print_exc()

  if run:
    run.close()
    print("Aim run closed.")

  print(f"====== METRIC STABILITY FOR {architecture} ======")
  health.log_run()

  clear_gpu();
  print(f"\n{'='*20} Finished Fold {architecture} {'='*20}")
  time.sleep(3)


# --- Final Cleanup ---
print("\nAll folds processed.")
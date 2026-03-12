import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import atexit
import random
import shutil
import h5py # <--- Added
import zipfile
import warnings
import json
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import pandas as pd
from datetime import datetime
import gc
import time

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")

import contextlib # Add this to imports

from torch_lr_finder import LRFinder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    DEFAULT_SUBSET_RATIO = 1.0 #changed for test
    USE_SUBSET = False #changed for test

elif EXECUTION_MODE == "PAPER":
    print("🛡️ RUNNING IN PAPER MODE: Strict Determinism Enforced")
    # Reproducibility settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Defaults for paper
    DEFAULT_SUBSET_RATIO = 1.0
    USE_SUBSET = False # Force full dataset unless explicitly overridden

# Global AMP setting
AMP_PRECISION = "fp16" # or "bf16" or "fp32"

# -----------------------------
# 1. Config & Experiment Plan
# -----------------------------
HDF5_DRIVE_DIR = "IA_MEDICA_SAMPLES/CAMELYON16"
LOCAL_DATA_DIR = "/content/dataset"

# Resolve paths
output_dir = '/content/reports/LR_FINDER_REPORTS'

# Experiment Settings
seed = 24
batch_size = 32
workers = os.cpu_count()

# DATASET STRATEGY:
# "Full Dataset" implies subset_ratio = 1.0.
# Do not use subset_ratio < 1.0 if you want the full data.
subset_ratio = DEFAULT_SUBSET_RATIO
use_subset = USE_SUBSET

# LR Finder Settings
n_lhs = 12
end_lr = 1e-1
num_iter = 100

# AMP LOGIC (Fixing the conflict)
# We trust the Global AMP_PRECISION set at the top of the script.
# If it is anything other than 'fp32', we enable AMP.
USE_AMP = (AMP_PRECISION.lower() != "fp32")

N_REPEATS = 3

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

def autocast_ctx(x: torch.Tensor, amp_dtype):
    # Only autocast if we are NOT in float32
    use_autocast = (amp_dtype is not torch.float32) and x.is_cuda
    return torch.amp.autocast("cuda", dtype=amp_dtype) if use_autocast else contextlib.nullcontext()

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

# --- Seeding & Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    # Ensure DataLoader workers have distinct seeds
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # 2. CRITICAL: Disable OpenCV threading in workers
    # We want the worker to use 1 core fully, not spawn sub-threads.
    # This reduces contention/jitter significantly.
    cv2.setNumThreads(0)

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
        if target.ndim == 4:
            target = target.squeeze(1)

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
    """Copies HDF5 files from Drive to local VM for speed."""
    print(f"\n{'='*25} Setting up HDF5 Data {'='*25}")
    if os.path.exists(local_dir): shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)


    # We only need TRAIN for LR Finder
    resolved_drive_dir = change_paths(drive_dir)
    src = get_filename(resolved_drive_dir)
    filename = os.path.basename(src)

    dst = os.path.join(local_dir, filename)

    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing {src}")

    print(f"Copying {filename}...")
    shutil.copy2(src, dst)
    return dst

# --- AUGMENTATION PIPELINE (Strictly Updated for Albumentations 1.4+ & nnU-Net) ---
def get_transforms(mode="train", img_size=224):
    if mode == "train":
        return A.Compose([
            # 2. GEOMETRIC INVARIANCE (Keep on CPU)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-45, 45),
                fill=0,
                fill_mask=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            # 3. COLOR & INTENSITY (Keep on CPU)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.8),

            # 4. ARTIFACT & QUALITY SIMULATION (CPU Side)
            # Original: OneOf([Blur, Noise, Downscale], p=0.2) -> ~6.6% each
            # New: OneOf([Blur, Noise], p=0.14) -> 0.14 * 0.5 = ~7% each (Close enough)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1.0),
            ], p=0.14),

            # 5. REGULARIZATION
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, int(img_size/10)),
                hole_width_range=(1, int(img_size/10)),
                fill=0,
                fill_mask=0,
                p=0.2
            ),

            # CRITICAL: Use ToFloat to ensure data arrives at GPU as 0.0-1.0 float,
            # or keep ToTensorV2 (uint8) and let GPUNormalizer handle it.
            # Your GPUNormalizer handles uint8, so ToTensorV2 is fine.
            ToTensorV2(),
        ])
    else:
        return A.Compose([ToTensorV2()])

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

def create_stratified_subset_within_patients(full_dataset, ratio, seed=24):
    """
    HDF5-Compatible Stratification.
    """
    print(f"  Creating a {ratio:.0%} stratified subsample...")

    indices_by_patient_and_class = defaultdict(lambda: defaultdict(list))

    # Access metadata directly from the HDF5 wrapper
    pids = full_dataset.get_patient_ids()
    lbls = full_dataset.get_labels()

    for i in range(len(full_dataset)):
        patient_id = pids[i]
        label = lbls[i]
        indices_by_patient_and_class[patient_id][label].append(i)

    subset_indices = []
    generator = torch.Generator().manual_seed(seed)

    for patient_id, class_groups in indices_by_patient_and_class.items():
        for label, indices in class_groups.items():
            num_to_sample = int(np.ceil(len(indices) * ratio))
            shuffled_indices = torch.randperm(len(indices), generator=generator).tolist()
            subset_indices.extend([indices[i] for i in shuffled_indices[:num_to_sample]])

    return subset_indices # Returns list of indices, not a Subset object yet

def collate_fn(batch):
  batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
  return torch.utils.data.dataloader.default_collate(batch) if batch else None

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

# Updated Search Space for BCEDiceHybridLossPaper
@dataclass(frozen=True)
class BCEDiceSearchSpace:
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 0.5
    gamma_min: float = 0.1
    gamma_max: float = 0.5
    gamma_log: bool = False  # Assuming uniform distribution for gamma

def latin_hypercube(n_samples: int, n_dims: int, seed: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_dims), dtype=np.float64)
    for d in range(n_dims):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.random(n_samples)
        X[:, d] = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(X[:, d])
    return X

def sample_bcedice_params(n: int, space: BCEDiceSearchSpace, seed: int) -> List[Dict[str, float]]:
    H = latin_hypercube(n, 3, seed)
    out = []
    for i in range(n):
        a_u, b_u, g_u = H[i]
        alpha = space.alpha_min + a_u * (space.alpha_max - space.alpha_min)
        beta  = space.beta_min  + b_u * (space.beta_max  - space.beta_min)
        if space.gamma_log:
            lo = math.log(space.gamma_min)
            hi = math.log(space.gamma_max)
            gamma = math.exp(lo + g_u * (hi - lo))
        else:
            gamma = space.gamma_min + g_u * (space.gamma_max - space.gamma_min)
        out.append({"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)})
    return out

# -----------------------------
# Curve statistics
# -----------------------------
@dataclass
class CurveStats:
    min_loss: float
    # We keep min_loss_lr internally just in case you check CSVs,
    # but we won't aggregate it for the PDF report.
    min_loss_lr: float

def _moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x.copy()
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def compute_curve_stats(lrs: np.ndarray, losses: np.ndarray, skip_start: int = 10, skip_end: int = 5) -> CurveStats:
    """
    Simplified Statistics: Strictly finds the Minimum Loss.
    Removes unused gradient/divergence/smoothness heuristics.
    """
    lrs = np.asarray(lrs, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)

    # 1. Trim Data
    n = len(lrs)
    lo = skip_start
    hi = max(lo + 1, n - skip_end)

    if lo >= hi:
        return CurveStats(float('inf'), 0.0)

    lrs_c = lrs[lo:hi]
    loss_c = losses[lo:hi]

    # Filter NaNs
    finite_mask = np.isfinite(loss_c) & np.isfinite(lrs_c) & (lrs_c > 0)
    lrs_c = lrs_c[finite_mask]
    loss_c = loss_c[finite_mask]

    if len(lrs_c) < 5:
        return CurveStats(float('inf'), 0.0)

    # 2. Smooth the Loss (Crucial to find the true valley, not a noise spike)
    loss_smooth = _moving_average(loss_c, k=5)

    # 3. Find Absolute Minimum
    min_idx = np.argmin(loss_smooth)
    min_loss_val = float(loss_smooth[min_idx])
    min_loss_lr = float(lrs_c[min_idx])

    return CurveStats(
        min_loss=min_loss_val,
        min_loss_lr=min_loss_lr
    )

# -----------------------------
# Utilities
# -----------------------------
def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if x >= 1e-2 and x < 1e3:
        return f"{x:.4f}"
    return f"{x:.2e}"

def latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("_", "\\_")
             .replace("%", "\\%")
             .replace("&", "\\&")
             .replace("#", "\\#")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("^", "\\^{}")
             .replace("~", "\\~{}"))

@dataclass
class RunRecord:
    architecture: str
    encoder: str
    alpha: float
    beta: float
    gamma: float

    # The only metric that matters now
    median_min_loss: float

    # Files
    plot_path: str
    csv_path: str

def run_lr_finder_once(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    end_lr: float,
    num_iter: int,
    architecture: str,
    amp_enabled: bool,
) -> Dict[str, np.ndarray]:

    # 1. Setup Precision
    effective_amp_precision = "fp32" if not amp_enabled else AMP_PRECISION
    amp_dtype, scaler, _ = setup_precision(architecture, amp_precision=effective_amp_precision)

    # Initialize LRFinder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # 2. Patch the inner training loop
    def _train_batch_patched(self, train_iter, accumulation_steps, non_blocking_transfer):
            self.model.train()

            # Fast fetch
            try:
                batch_data = next(train_iter)
            except StopIteration:
                return float("nan")

            images, masks = batch_data

            # Move to GPU - Non Blocking is crucial
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

            # Normalize on GPU
            images = gpu_normalizer(images)

            images = gpu_downscale(images)

            self.optimizer.zero_grad(set_to_none=True) # set_to_none is faster

            try:
                # AMP Context
                with autocast_ctx(images, amp_dtype):
                    outputs = self.model(images)
                    # BCEDiceHybridLossPaper now handles LongTensor target directly
                    loss = self.criterion(outputs, masks)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                return loss.item()

            except Exception:
                return float("nan")

    # Apply Patch
    lr_finder._train_batch = _train_batch_patched.__get__(lr_finder, LRFinder)

    history = None
    try:
        # Run
        lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
        history = lr_finder.history
    except Exception as e:
        print(f"[LRFinder] Execution Error: {e}")
    finally:
        # --- CRITICAL MEMORY FIX ---
        # Explicitly break references to model and optimizer so GC can reclaim VRAM immediately
        if hasattr(lr_finder, 'model'):
            lr_finder.model = None
        if hasattr(lr_finder, 'optimizer'):
            lr_finder.optimizer = None
        if hasattr(lr_finder, 'criterion'):
            lr_finder.criterion = None

        del lr_finder

    if history is None or ("lr" not in history) or ("loss" not in history):
        # Return empty arrays to signal failure without crashing
        return {
            "lr": np.array([], dtype=np.float64),
            "loss": np.array([], dtype=np.float64),
        }

    return {
        "lr": np.array(history["lr"], dtype=np.float64),
        "loss": np.array(history["loss"], dtype=np.float64),
    }

def plot_stability_curves(
    all_lrs: List[np.ndarray],
    all_losses: List[np.ndarray],
    title: str,
    out_png: Path,
    skip_start: int = 10,
    skip_end: int = 5
):
    """
    Plots multiple LR vs Loss curves on the same figure to visualize stability.
    """
    plt.figure(figsize=(10, 6))

    for i, (lrs, losses) in enumerate(zip(all_lrs, all_losses)):
        n = len(lrs)
        # Apply skips to remove warmup/divergence noise at edges
        lo = skip_start
        hi = max(lo + 1, n - skip_end)

        if lo >= hi:
            continue # Skip runs that are too short

        lrs_c = lrs[lo:hi]
        loss_c = losses[lo:hi]

        # Plot with transparency to see overlaps
        plt.plot(np.log10(lrs_c), loss_c, alpha=0.6, linewidth=1.5, label=f"Run {i+1}")

    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def build_latex_report(
    records: List[RunRecord],
    out_dir: Path,
    meta: Dict[str, str],
    pdf_name: str = "report.pdf",
) -> Tuple[Path, Path]:
    """
    Creates report.tex and compiles to report.pdf.
    Features: 4-decimal precision for params, only shows Min Loss, Top 3 Plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    by_arch: Dict[str, List[RunRecord]] = {}
    for r in records:
        by_arch.setdefault(r.architecture, []).append(r)

    tex_lines = []
    tex_lines.append(r"\documentclass[11pt]{article}")
    tex_lines.append(r"\usepackage[a4paper,margin=1.0cm]{geometry}")
    tex_lines.append(r"\usepackage{graphicx}")
    tex_lines.append(r"\usepackage{float}")
    tex_lines.append(r"\usepackage{booktabs}")
    tex_lines.append(r"\usepackage{longtable}")
    tex_lines.append(r"\usepackage{hyperref}")
    tex_lines.append(r"\usepackage{caption}")
    tex_lines.append(r"\captionsetup{font=small,labelfont=bf}")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"\title{LR Finder Screening (N=3 Stability)}")
    tex_lines.append(rf"\date{{{latex_escape(meta.get('timestamp',''))}}}")
    tex_lines.append(r"\maketitle")

    for arch, recs in by_arch.items():
        tex_lines.append(rf"\section{{{latex_escape(arch)}}}")
        enc_name = latex_escape(recs[0].encoder) if recs else "NA"
        tex_lines.append(rf"\noindent\textbf{{Encoder:}} {enc_name}\\")
        tex_lines.append(r"\medskip")

        # --- TABLE (MINIMALIST) ---
        tex_lines.append(r"\begin{longtable}{lll|r}")
        tex_lines.append(r"\caption{Configs sorted by Minimum Loss (Best first)}\\")
        tex_lines.append(r"\toprule")
        # Columns: Params (4 decimals) | Min Loss
        tex_lines.append(r"$\alpha$ & $\beta$ & $\gamma$ & \textbf{Min Loss} \\")
        tex_lines.append(r"\midrule")
        tex_lines.append(r"\endfirsthead")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"$\alpha$ & $\beta$ & $\gamma$ & \textbf{Min Loss} \\")
        tex_lines.append(r"\midrule")
        tex_lines.append(r"\endhead")

        # SORTING: Median Min Loss (Lower is better)
        recs_sorted = sorted(recs, key=lambda r: r.median_min_loss)

        for r in recs_sorted:
          # Bold the row if it's the top 1
          prefix = r"\textbf{" if r == recs_sorted[0] else ""
          suffix = "}" if r == recs_sorted[0] else ""

          tex_lines.append(
              rf"{prefix}{r.alpha:.4f}{suffix} & "  # 4 Decimals
              rf"{prefix}{r.beta:.4f}{suffix} & "   # 4 Decimals
              rf"{prefix}{r.gamma:.4f}{suffix} & "  # 4 Decimals
              rf"{prefix}{fmt(r.median_min_loss)}{suffix} \\"
              )

        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{longtable}")
        tex_lines.append(r"\clearpage")

        tex_lines.append(r"\subsection*{Top Configuration Curves}")

        for i, r in enumerate(recs_sorted):
            tex_lines.append(r"\begin{figure}[H]")
            tex_lines.append(r"\centering")
            try:
                rel = Path(r.plot_path).relative_to(out_dir)
                tex_lines.append(rf"\includegraphics[width=0.75\linewidth]{{{latex_escape(str(rel))}}}")
            except ValueError:
                tex_lines.append(r"Image path error")

            cap = (
                f"Rank {i+1}: BCE(a={r.alpha:.4f}, b={r.beta:.4f}, g={r.gamma:.4f}). "
                f"Min Loss={fmt(r.median_min_loss)}."
            )
            tex_lines.append(rf"\caption{{{latex_escape(cap)}}}")
            tex_lines.append(r"\end{figure}")

        tex_lines.append(r"\clearpage")

    tex_lines.append(r"\end{document}")

    tex_path = out_dir / "report.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    import subprocess
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_path.name)]
    for _ in range(2):
        subprocess.run(cmd, cwd=str(out_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    pdf_path = out_dir / "report.pdf"
    return tex_path, pdf_path

# -----------------------------
# Helpers for run IDs / names
# -----------------------------
def make_run_tag(arch, enc, w, d, g, idx):
    # idx is 1-based, fixed-width for sorting
    return f"{arch}_a{w:.3f}_b{d:.3f}_g{g:.3f}__{idx:03d}"

def print_arch_banner(arch, enc):
    print("\n" + "="*80)
    print(f"[ARCH] {arch} | [ENC] {enc}")
    print("="*80)

HDF5_DRIVE_DIR = change_paths(HDF5_DRIVE_DIR)
output_dir_drive = change_paths('LR_FINDER_REPORTS/CAMELYON16')

out_dir = Path(output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Transfer Data
train_h5_path = setup_local_hdf5(HDF5_DRIVE_DIR, LOCAL_DATA_DIR)

# Architectures / encoders (as requested)
LIST_ARCH = ['SWIN','DEEPLABV3PLUS','UNET++','FPN','SEGFORMER','MANET','DPT','UPERNET']
LIST_ENCODER = [
    'tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k',
    'tu-resnest101e',
    'efficientnet-b7',
    'senet154',
    'mit_b5',
    'resnet152',
    'tu-vit_large_patch16_224.augreg_in21k_ft_in1k',
    'tu-hiera_large_224'
]

# -----------------------------
# Experiment plan (ONE source of truth)
# -----------------------------
assert len(LIST_ARCH) == len(LIST_ENCODER), "LIST_ARCH and LIST_ENCODER must match length"

# Define the search space
space = BCEDiceSearchSpace(
    alpha_min=0.1, alpha_max=1.0,
    beta_min=0.1, beta_max=0.5,
    gamma_min=0.1, gamma_max=0.5,
    gamma_log=False
)

lhs_samples = sample_bcedice_params(n=n_lhs, space=space, seed=seed)

(out_dir / "LHS_SAMPLES.json").write_text(json.dumps(lhs_samples, indent=2), encoding="utf-8")

N_ARCH = len(LIST_ARCH)
N_LHS  = len(lhs_samples)
TOTAL_RUNS = N_ARCH * N_LHS

print(f"\n[INFO] Total planned runs = {N_ARCH} architectures × {N_LHS} LHS samples = {TOTAL_RUNS}\n")

# Global progress bar
global_pbar = tqdm(total=TOTAL_RUNS * N_REPEATS, desc="GLOBAL LR-FINDER", unit="trial", dynamic_ncols=True)

records: List[RunRecord] = []
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

completed_ok = 0
failed_runs = 0

# Instantiate globally
gpu_normalizer = GPUNormalizer(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    device=device
)

gpu_downscale = GPUDownscale(p=0.07).to(device)

# ==================================================================================
# DATASET SETUP (Perform ONCE to prevent OOM)
# ==================================================================================
print("\n[SETUP] Initializing Dataset and caching to RAM if possible...")

# 1. Create Stratification Indices (Lightweight Metadata Read)
# We use a temporary instance just to read metadata and calculate indices.
# We don't load images here.
temp_meta_ds = HybridProstateDataset(train_h5_path, mode="train")
if use_subset and subset_ratio < 1.0:
    sub_indices_base = create_stratified_subset_within_patients(temp_meta_ds, subset_ratio, seed=seed)
else:
    sub_indices_base = None

# Cleanup temp metadata object
del temp_meta_ds
gc.collect()

# 2. Load the ACTUAL Persistent Dataset
# This will allocate the ~6GB RAM once. We will reuse this object.
persistent_dataset = HybridProstateDataset(
    train_h5_path,
    mode="train",
    subset_indices=sub_indices_base
)

# Pre-calculate weights for sampling (these rely on labels, which are static)
# We calculate the raw weights once. The randomization happens in the Sampler class.
dataset_labels = persistent_dataset.get_labels()
class_counts = np.bincount(dataset_labels)
class_counts[class_counts == 0] = 1
class_weights = 1. / class_counts
sample_weights_base = class_weights[dataset_labels]
sample_weights_tensor = torch.from_numpy(sample_weights_base).float()

print(f"[SETUP] Dataset loaded. Starting LR Finder loops...\n")

# ==================================================================================
# MAIN EXECUTION LOOP
# ==================================================================================
for arch, enc in zip(LIST_ARCH, LIST_ENCODER):
    print_arch_banner(arch, enc)
    arch_dir = out_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    for j, p in enumerate(lhs_samples, start=1):
        alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
        tag_base = f"Loss_alpha_{alpha:.3f}_beta_{beta:.3f}_gamma_{gamma:.3f}"

        rep_lrs = []
        rep_losses = []
        rep_stats = []

        print(f"\n--- Testing Config {j}/{len(lhs_samples)} ({N_REPEATS} repeats) ---")

        for rep in range(N_REPEATS):
            # 1. Deterministic Seeding (Affects Weights AND Data Shuffling)
            current_seed = seed + (j * 100) + rep
            seed_everything(current_seed)

            # 2. Initialize Model & Optimizer
            model = get_model(arch, enc).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8, weight_decay=1e-4)
            criterion = BCEDiceHybridLossPaper(alpha=alpha, beta=beta, gamma=gamma)

            # 3. Create Sampler & DataLoader (Lightweight)
            # CRITICAL: We pass the EXISTING 'persistent_dataset'. We do NOT reload 6GB.
            # The WeightedRandomSampler uses the global torch generator (seeded above),
            # so the shuffling WILL be different and deterministic for this repeat.

            sampler_iter = WeightedRandomSampler(
                weights=sample_weights_tensor,
                num_samples=len(sample_weights_tensor),
                replacement=True
            )

            train_loader_iter = DataLoader(
                persistent_dataset, # Reuse the RAM-loaded dataset
                batch_size=batch_size,
                shuffle=False,      # Sampler handles shuffling
                sampler=sampler_iter,
                num_workers=workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn,
                persistent_workers=(workers > 0),
                prefetch_factor=4 if workers > 0 else None
            )

            try:
                hist = run_lr_finder_once(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=train_loader_iter, # Use the local loader
                    device=device,
                    end_lr=end_lr,
                    num_iter=num_iter,
                    architecture=arch,
                    amp_enabled=USE_AMP, # Use the derived boolean
                )

                lrs = hist["lr"]
                losses = hist["loss"]
                stats = compute_curve_stats(lrs, losses, skip_start=10, skip_end=5)

                rep_lrs.append(lrs)
                rep_losses.append(losses)
                rep_stats.append(stats)
                completed_ok += 1

            except Exception as e:
                print(f"[FAIL] Rep {rep+1}: {e}")
                failed_runs += 1

            finally:
                # Cleanup
                if 'model' in locals(): del model
                if 'optimizer' in locals(): del optimizer
                if 'criterion' in locals(): del criterion
                if 'train_loader_iter' in locals(): del train_loader_iter # Break reference to loader

                gc.collect()
                clear_gpu()
                global_pbar.update(1)

        # --- AGGREGATION & REPORTING ---
        if len(rep_stats) > 0:
            # 1. Plotting
            png_path = arch_dir / f"{j:03d}_{tag_base}.png"
            plot_title = f"{arch} | Stability (N={N_REPEATS})\nα={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}"

            plot_stability_curves(
                all_lrs=rep_lrs,
                all_losses=rep_losses,
                title=plot_title,
                out_png=png_path,
                skip_start=10,
                skip_end=5
            )

            # 2. Statistics Aggregation (Cleaned)
            # We only care about which config produced the lowest loss
            min_losses = [s.min_loss for s in rep_stats if not np.isnan(s.min_loss)]
            median_min_loss = float(np.median(min_losses)) if min_losses else float('inf')

            # Save Record (Cleaned)
            records.append(RunRecord(
                architecture=arch,
                encoder=enc,
                alpha=alpha,
                beta=beta,
                gamma=gamma,

                median_min_loss=median_min_loss,

                plot_path=str(png_path),
                csv_path="aggregated"
            ))

            # Simple, clean log
            print(f"[RESULT] {arch} Config {j}: Loss = {median_min_loss:.4f}")

    # Save summary per architecture
    arch_df = pd.DataFrame([asdict(r) for r in records if r.architecture == arch])
    if len(arch_df) > 0:
        arch_df.to_csv(arch_dir / f"SUMMARY_{arch}_STABILITY.csv", index=False)

global_pbar.close()

# global summary
all_df = pd.DataFrame([asdict(r) for r in records])
all_df.to_csv(out_dir / "SUMMARY_ALL.csv", index=False)

meta = {
    "timestamp": timestamp,
    "seed": str(seed),
    "device": str(device),
    "batch_size": str(batch_size),
    "workers": str(workers),
    "subset_ratio": str(subset_ratio) if use_subset else "1.0",
    "lhs_n": str(n_lhs),
    "BCE_alpha_range": f"[{space.alpha_min}, {space.alpha_max}]",
    "BCE_beta_range": f"[{space.beta_min}, {space.beta_max}]",
    "BCE_gamma_range": f"[{space.gamma_min}, {space.gamma_max}] " + ("(log-uniform)" if space.gamma_log else "(uniform)"),
    "optimizer": "torch.optim.AdamW",
    "lr_finder": f"torch_lr_finder.LRFinder | end_lr={end_lr} | num_iter={num_iter} | step_mode=exp",
    "amp": str(USE_AMP), # FIXED: Changed from 'amp' to 'USE_AMP'
}

print("\nBuilding LaTeX report...")
tex_path, pdf_path = build_latex_report(records, out_dir, meta=meta)
print("Wrote:", tex_path)
print("PDF :", pdf_path)

from pathlib import Path
import shutil

def copy_out_dir_to_final(out_dir: str | Path, final_report_path: str | Path, overwrite: bool = True) -> Path:
    """
    Copy the entire contents of `out_dir` into `final_report_path`.

    - If `overwrite=True`, it will delete `final_report_path` first (if it exists),
      then copy everything fresh (recommended for reproducibility).
    - If `overwrite=False`, it will merge contents (may overwrite files with same names).

    Returns:
        Path to the final_report_path.
    """
    src = Path(out_dir)
    dst = Path(final_report_path)

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"out_dir does not exist or is not a directory: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            # Merge-copy (Python >= 3.8 supports dirs_exist_ok)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return dst

    # Fresh copy
    shutil.copytree(src, dst)
    return dst

# --- Helper Functions ---
def get_formatted_datetime_string():
  now = datetime.now()
  return now.strftime("%d_%m_%Y_%H_%M_%S")

os.makedirs(output_dir, exist_ok=True)
dst = os.path.join(output_dir_drive, get_formatted_datetime_string())
shutil.copytree(output_dir, dst)
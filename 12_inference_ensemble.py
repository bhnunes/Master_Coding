import atexit
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime

import albumentations as A
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import auc as sklearn_auc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 5. Configuration & Setup ---
print("Configuring environment...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

ENSEMBLE_META_PATH = "RESULTS_REPORT_ENSEMBLE/CAMELYON16/ENSEMBLE_TWO_STREAM_04_03_2026_04_49_46.json" # <--- UPDATE THIS PATH

BATCH_SIZE = 32
WORKERS = 2
SEED = 24

# =============================================================================
# 3) Dataset locations
# =============================================================================
# Google Drive Folder containing TEST.h5
HDF5_DRIVE_DIR = 'IA_MEDICA_SAMPLES/CAMELYON16'

# Local temporary directory on Colab (fast NVMe I/O)
LOCAL_DATA_DIR = '/content/dataset'

DECODER_DROPOUT = 0

WHERE_WAS_CREATED = '/content/drive/MyDrive/Personal_Drive_Bruno/'
CURRENT_ENV = '/content/drive/MyDrive/'

DISABLE_AMP = {}

def change_path(path, possible_paths=None):
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

def change_paths(path):
  if not os.path.exists(path):
    new_path = str(path).replace(WHERE_WAS_CREATED, CURRENT_ENV)
    if os.path.exists(new_path):
      path = new_path
      print(f"Updated path to {path}")
    else:
      print(f"Path not found: {new_path}")
  return path

# --- Seeding ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

print("Configuration complete.")

# --- Helper Functions ---
def get_formatted_datetime_string():
  now = datetime.now()
  return now.strftime("%d_%m_%Y_%H_%M_%S")

def _mask_to_binary_indices(masks: torch.Tensor) -> torch.Tensor:
    """
    Accepts either:
      - one-hot masks: [B,2,H,W] (legacy)
      - binary masks:  [B,H,W]   (new fast path)
    Returns:
      - binary indices: [B,H,W] uint8 in {0,1}
    """
    if masks.ndim == 4 and masks.shape[1] == 2:
        return torch.argmax(masks, dim=1).to(torch.uint8)
    if masks.ndim == 3:
        return masks.to(torch.uint8)
    raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")

def postprocess_binary_mask_np(pred_mask_uint8, min_area, open_ksize):
    """
    Applies morphological opening and area filtering to a binary mask (H, W).
    Input: pred_mask_uint8 (0 or 1), numpy array.
    """
    # 1. Morphological Opening
    if open_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        pred_mask_uint8 = cv2.morphologyEx(pred_mask_uint8, cv2.MORPH_OPEN, kernel)

    # 2. Area Filtering
    if min_area > 0:
        # cv2.connectedComponentsWithStats takes uint8 image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_uint8, connectivity=8)

        # stats: [x, y, width, height, area]
        # Label 0 is background
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                pred_mask_uint8[labels == i] = 0

    return pred_mask_uint8

def get_model(architecture, encoder,validation=False):

  if validation:
    aux_params=None
  else:
    aux_params=dict(dropout=DECODER_DROPOUT, classes=2)

  encoder_weights = None if validation else "imagenet"

  if architecture=="SWIN":
    model = smp.Unet(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=2,
    activation=None,
    decoder_attention_type=None,
    aux_params=aux_params)
  elif architecture=="DEEPLABV3PLUS":
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="INCEPTIONRESNETV2":
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="DPT":
    model = smp.DPT(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        decoder_readout='ignore',
        aux_params=aux_params)
  elif architecture=="UNET++":
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="FPN":
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="SEGFORMER":
    model = smp.Segformer(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="MANET":
    model = smp.MAnet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  elif architecture=="UPERNET":
    model = smp.UPerNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_attention_type=None,
        aux_params=aux_params)
  else:
    raise ValueError(f"Unknown architecture: {architecture}")

  return model

@torch.inference_mode()
def predict_with_tta_batched(model, images, use_amp: bool = True):
    """
    TTA = {orig, hflip, vflip} executed SEQUENTIALLY to save memory.

    Previous version: [3*B, C, H, W] -> Peak Memory: High
    This version:     [B, C, H, W] x 3 -> Peak Memory: Low (1/3rd)
    """
    # 1. Forward Pass (Original)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        out = model(images)
        if isinstance(out, (tuple, list)): out = out[0]

        # Convert to probabilities
        if out.shape[1] == 1:
            probs = torch.sigmoid(out).squeeze(1)
        else:
            probs = torch.softmax(out, dim=1)[:, 1, :, :]

    # 2. Horizontal Flip (Input Flip -> Forward -> Output Flip)
    images_h = torch.flip(images, dims=[3])
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        out_h = model(images_h)
        if isinstance(out_h, (tuple, list)): out_h = out_h[0]

        if out_h.shape[1] == 1:
            p_h = torch.sigmoid(out_h).squeeze(1)
        else:
            p_h = torch.softmax(out_h, dim=1)[:, 1, :, :]

    # Accumulate in-place (Un-flip Width axis 2)
    probs.add_(torch.flip(p_h, dims=[2]))
    # Free memory immediately
    del images_h, out_h, p_h

    # 3. Vertical Flip
    images_v = torch.flip(images, dims=[2])
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        out_v = model(images_v)
        if isinstance(out_v, (tuple, list)): out_v = out_v[0]

        if out_v.shape[1] == 1:
            p_v = torch.sigmoid(out_v).squeeze(1)
        else:
            p_v = torch.softmax(out_v, dim=1)[:, 1, :, :]

    # Accumulate in-place (Un-flip Height axis 1)
    probs.add_(torch.flip(p_v, dims=[1]))
    # Free memory
    del images_v, out_v, p_v

    # Average
    probs.div_(3.0)
    return probs

def clear_gpu():
    print("Clearing GPU cache...")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            # Optional: only print if something really went wrong
            print(f"[clear_gpu] Warning: {e}")
    print("GPU cache cleared.")

def calculate_metrics(tp, fp, fn, tn):
    """
    Single-class metrics for the tumor (positive) class.

    Dice/IoU follow the standard empty-ground-truth convention:
      - If GT is empty and prediction is empty: Dice=1, IoU=1 (perfect rejection)
      - If GT is empty and prediction is not empty: Dice=0, IoU=0 (false alarm)
      - Else: standard Dice/IoU
    """
    tp = float(tp); fp = float(fp); fn = float(fn); tn = float(tn)

    denom_tpr  = tp + fn
    denom_tnr  = tn + fp
    denom_prec = tp + fp
    denom_acc  = tp + tn + fp + fn

    gt_empty   = (tp + fn) == 0.0
    pred_empty = (tp + fp) == 0.0

    if gt_empty:
        if pred_empty:
            dice = 1.0
            iou  = 1.0
        else:
            dice = 0.0
            iou  = 0.0
    else:
        denom_dice = (2 * tp + fp + fn)
        denom_iou  = (tp + fp + fn)
        dice = (2 * tp) / denom_dice if denom_dice > 0 else 0.0
        iou  = tp / denom_iou        if denom_iou  > 0 else 0.0

    return {
        'dice': dice,
        'iou': iou,
        'accuracy': (tp + tn) / denom_acc if denom_acc > 0 else np.nan,
        'tpr': tp / denom_tpr if denom_tpr > 0 else np.nan,     # sensitivity
        'tnr': tn / denom_tnr if denom_tnr > 0 else np.nan,     # specificity
        'precision': tp / denom_prec if denom_prec > 0 else np.nan,
        'fpr': fp / denom_tnr if denom_tnr > 0 else np.nan,
        'fnr': fn / denom_tpr if denom_tpr > 0 else np.nan
    }

# Helper for printing
def print_metric_line(metric, pe, ci):
    ci_str = f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])" if not np.isnan(ci[0]) else "(CI not calculated; n_patients < 20)"
    print(f"  {metric.replace('_',' ').title():<12}: {pe:.4f}  {ci_str}")

@torch.inference_mode()
def analyze_ensemble_metrics(models_list, constituent_models_info,
                            test_loader, device, optimal_threshold, train_mean, train_std, ensemble_recipe=None, num_auc_steps=101):
    """
    Refactored for 'Two-Stream Spatial Gating' inference.
    1. Enforces strict strategy check.
    2. Splits accumulation into Semantic (Scout) and Spatial (Sniper) streams.
    3. Generates ROI via Low-Pass filter simulation.
    4. Gates Spatial predictions.
    5. Calculates metrics on the final Gated output.
    """

    # --- 1. STRICT STRATEGY VALIDATION ---
    strategy = ensemble_recipe.get("ensemble_strategy")
    if strategy != "two_stream_spatial_gating":
        raise ValueError(
            f"Invalid strategy '{strategy}'. This pipeline ONLY supports 'two_stream_spatial_gating'. "
            "Please check your JSON configuration."
        )

    # Get ROI Params from JSON
    roi_config = ensemble_recipe.get("roi_config", {})
    roi_scale = int(roi_config.get("scale", 4)) # Default from optimization

    # Pre-calculate AMP settings
    for model in models_list:
        model.eval()

    # --- 2. DATA COLLECTION LOOP ---
    stats_by_patient = defaultdict(list)

    # === OPTIMIZATION 1: GPU Histograms ===
    AUC_BINS = 4096
    auc_pos_hist = torch.zeros(AUC_BINS, dtype=torch.int64, device=device)
    auc_neg_hist = torch.zeros(AUC_BINS, dtype=torch.int64, device=device)

    print(f"Running Two-Stream Inference. Thr: {optimal_threshold:.4f}, ROI Scale: {roi_scale}")

    with torch.inference_mode():
        pbar = tqdm(test_loader, desc="Inference", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None or batch_data[0] is None:
                continue

            # Unpack
            if len(batch_data) == 3: images, masks, patient_ids = batch_data
            elif len(batch_data) == 4: images, _, masks, patient_ids = batch_data
            else: continue

            # Move to GPU
            images = images.to(device, non_blocking=True)
            images = gpu_normalizer(images)

            # Move GT to GPU
            masks_gpu = masks.to(device, non_blocking=True)
            true_gpu = _mask_to_binary_indices(masks_gpu)

            # --- STREAM ACCUMULATORS ---
            sem_accum = None
            spa_accum = None

            # --- Iterate Models ---
            for m_idx, (model, meta) in enumerate(zip(models_list, constituent_models_info)):

                # Extract Declarative Info
                weight = float(meta.get('weight', meta.get('ensemble_weight', 0.0)))
                role = meta.get('stream_role', 'none') # 'semantic' or 'spatial'

                if weight <= 1e-8: continue

                # Predict
                probs = predict_with_tta_batched(model, images, use_amp=True)

                # Cleanup & Clamp
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs.clamp_(0.0, 1.0)

                # Direct to correct stream
                if role == 'semantic':
                    if sem_accum is None:
                        sem_accum = probs * weight
                    else:
                        sem_accum.add_(probs, alpha=weight)

                elif role == 'spatial':
                    if spa_accum is None:
                        spa_accum = probs * weight
                    else:
                        spa_accum.add_(probs, alpha=weight)

                # Note: models with role='none' or weight=0 are strictly ignored

            # --- APPLY TWO-STREAM LOGIC ---

            # 1. Handle Empty Accumulators (Edge case safety)
            # If no spatial models fired, prediction is 0. If no semantic, mask is 0.
            if spa_accum is None:
                final_probs = torch.zeros_like(true_gpu, dtype=torch.float32)
            else:
                # We have spatial predictions, now check semantic gating
                if sem_accum is None:
                    # No semantic models -> Mask is 0 -> Final is 0
                    final_probs = torch.zeros_like(true_gpu, dtype=torch.float32)
                else:
                    # 2. Generate ROI Mask (Low-Pass Simulation)
                    # Downsample -> Threshold -> Upsample
                    B, H, W = images.shape[0], images.shape[2], images.shape[3]
                    small_h, small_w = H // roi_scale, W // roi_scale

                    # Downsample (Bilinear)
                    sem_small = torch.nn.functional.interpolate(
                        sem_accum.unsqueeze(1),
                        size=(small_h, small_w),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Threshold (Operating Point)
                    roi_small = (sem_small > optimal_threshold).float()

                    # Upsample (Nearest) to keep binary edges
                    roi_mask = torch.nn.functional.interpolate(
                        roi_small,
                        size=(H, W),
                        mode='nearest'
                    ).squeeze(1)

                    # 3. Apply Gate
                    final_probs = spa_accum * roi_mask

            # Final Clamp
            final_probs.clamp_(0.0, 1.0)

            # === METRICS ACCUMULATION (Same as before but using final_probs) ===

            # 1. AUC Histograms
            bin_idx = (final_probs * (AUC_BINS - 1)).long()
            bin_idx.clamp_(0, AUC_BINS - 1)

            flat_bins = bin_idx.view(-1)
            flat_true = true_gpu.view(-1).bool()

            pos_bins = flat_bins[flat_true]
            neg_bins = flat_bins[~flat_true]

            auc_pos_hist.add_(torch.bincount(pos_bins, minlength=AUC_BINS))
            auc_neg_hist.add_(torch.bincount(neg_bins, minlength=AUC_BINS))

            # 2. Confusion Matrix (Fast Path)
            # For Gated approach, the 'optimal_threshold' was used to generate the mask.
            # Any non-zero value in final_probs means (Spatial > 0 AND ROI == 1).
            # To be consistent with standard metrics, we threshold at > 0 (or a tiny epsilon).
            pred_gpu = (final_probs > 1e-4).to(torch.uint8)

            conf_vec = pred_gpu.mul(2).add_(true_gpu).view(pred_gpu.size(0), -1)

            for j, pid in enumerate(patient_ids):
                cnts = torch.bincount(conf_vec[j], minlength=4).cpu().tolist()
                stats_by_patient[pid].append({
                    'tn': cnts[0], 'fn': cnts[1], 'fp': cnts[2], 'tp': cnts[3]
                })

            # Cleanup
            del sem_accum, spa_accum, final_probs, bin_idx, flat_bins, pos_bins, neg_bins, masks_gpu, true_gpu
            if "roi_mask" in locals(): del roi_mask
            if "images" in locals(): del images

    # --- 3. POST-LOOP METRICS ---
    # (Remains identical to your existing robust metric calculation)

    if not stats_by_patient:
        print("Error: No patients processed.")
        return None

    unique_patient_ids = list(stats_by_patient.keys())
    n_patients = len(unique_patient_ids)
    run_bootstrap = n_patients >= 20

    if not run_bootstrap:
        print(f"\nWARNING: n_patients={n_patients} < 20. Skipping Bootstrap CIs.")

    n_bootstrap_samples = 10000
    metric_keys = ['dice', 'iou', 'accuracy', 'tpr', 'tnr', 'precision', 'fpr', 'fnr']

    # Initialize results
    final_micro_metrics = {'point_estimate': {}, 'ci': {key: [np.nan, np.nan] for key in metric_keys}}
    final_macro_metrics = {'point_estimate': {}, 'ci': {key: [np.nan, np.nan] for key in metric_keys}}

    # --- 4. POINT ESTIMATES ---
    print("\nCalculating Point Estimates...")
    all_patches_stats = [stat for pat_stats in stats_by_patient.values() for stat in pat_stats]
    tp_micro, fp_micro, fn_micro, tn_micro = np.sum([list(s.values()) for s in all_patches_stats], axis=0)
    final_micro_metrics['point_estimate'] = calculate_metrics(tp_micro, fp_micro, fn_micro, tn_micro)

    per_patient_scores = {key: np.zeros(n_patients) for key in metric_keys}
    for i, pid in enumerate(unique_patient_ids):
        tp_pat, fp_pat, fn_pat, tn_pat = np.sum([list(s.values()) for s in stats_by_patient[pid]], axis=0)
        p_metrics = calculate_metrics(tp_pat, fp_pat, fn_pat, tn_pat)
        for key in metric_keys:
            per_patient_scores[key][i] = p_metrics[key]

    # --- NEW (Rule 6) Logic ---
    dice_pos_only = np.full(n_patients, np.nan, dtype=np.float64)
    neg_clean = np.full(n_patients, np.nan, dtype=np.float64)
    n_pos_patients = 0
    n_neg_patients = 0

    for i, pid in enumerate(unique_patient_ids):
        tp_pat, fp_pat, fn_pat, tn_pat = np.sum([list(s.values()) for s in stats_by_patient[pid]], axis=0)
        p_metrics = calculate_metrics(tp_pat, fp_pat, fn_pat, tn_pat)

        has_tumor = (tp_pat + fn_pat) > 0
        if has_tumor:
            dice_pos_only[i] = p_metrics['dice']
            n_pos_patients += 1
        else:
            neg_clean[i] = 1.0 if (fp_pat == 0) else 0.0
            n_neg_patients += 1

    for metric in metric_keys:
        final_macro_metrics['point_estimate'][metric] = np.nanmean(per_patient_scores[metric])

    macro_dice_pos_only_pe = float(np.nanmean(dice_pos_only)) if n_pos_patients > 0 else float("nan")
    macro_neg_clean_rate_pe = float(np.nanmean(neg_clean)) if n_neg_patients > 0 else float("nan")

    # --- 5. BOOTSTRAP ---
    if run_bootstrap:
        print(f"Calculating Bootstrap CIs ({n_bootstrap_samples} samples)...")
        np.random.seed(SEED)

        # Micro
        boot_micro = {key: np.zeros(n_bootstrap_samples) for key in metric_keys}
        for i in range(n_bootstrap_samples):
            resampled_pids = np.random.choice(unique_patient_ids, size=n_patients, replace=True)
            # Note: For strict correctness, we map back to indices to speed up sums
            # (Simplified here for readability, your existing implementation is fine)
            resampled_stats = [stats_by_patient[pid] for pid in resampled_pids]
            flat_stats = [item for sublist in resampled_stats for item in sublist]

            totals = {'tp':0, 'fp':0, 'fn':0, 'tn':0}
            for s in flat_stats:
                totals['tp']+=s['tp']; totals['fp']+=s['fp']; totals['fn']+=s['fn']; totals['tn']+=s['tn']

            m = calculate_metrics(totals['tp'], totals['fp'], totals['fn'], totals['tn'])
            for k in metric_keys: boot_micro[k][i] = m[k]

        for k in metric_keys:
            final_micro_metrics['ci'][k] = np.percentile(boot_micro[k], [2.5, 97.5])

        # Macro
        boot_macro = {key: np.zeros(n_bootstrap_samples) for key in metric_keys}
        boot_dice_pos_only = np.zeros(n_bootstrap_samples, dtype=np.float64)
        boot_neg_clean = np.zeros(n_bootstrap_samples, dtype=np.float64)

        for i in range(n_bootstrap_samples):
            idxs = np.random.choice(range(n_patients), size=n_patients, replace=True)
            for k in metric_keys:
                boot_macro[k][i] = np.nanmean(per_patient_scores[k][idxs])

            boot_dice_pos_only[i] = np.nanmean(dice_pos_only[idxs])
            boot_neg_clean[i] = np.nanmean(neg_clean[idxs])

        for k in metric_keys:
            final_macro_metrics['ci'][k] = np.percentile(boot_macro[k], [2.5, 97.5])

        macro_dice_pos_only_ci = np.percentile(boot_dice_pos_only, [2.5, 97.5]) if n_pos_patients > 0 else [np.nan, np.nan]
        macro_neg_clean_rate_ci = np.percentile(boot_neg_clean, [2.5, 97.5]) if n_neg_patients > 0 else [np.nan, np.nan]
    else:
        macro_dice_pos_only_ci = [np.nan, np.nan]
        macro_neg_clean_rate_ci = [np.nan, np.nan]

    # --- 6. AUC ---
    print("\nCalculating Final AUC...")
    try:
        auc_pos_np = auc_pos_hist.cpu().numpy()
        auc_neg_np = auc_neg_hist.cpu().numpy()
        total_pos = int(auc_pos_np.sum())
        total_neg = int(auc_neg_np.sum())

        if total_pos > 0 and total_neg > 0:
            tp = np.cumsum(auc_pos_np[::-1]).astype(np.float64)
            fp = np.cumsum(auc_neg_np[::-1]).astype(np.float64)
            tpr = tp / total_pos
            fpr = fp / total_neg
            tpr = np.concatenate(([0.0], tpr))
            fpr = np.concatenate(([0.0], fpr))
            auc_score = sklearn_auc(fpr, tpr)
        else:
            auc_score = float("nan")
    except Exception as e:
        print(f"AUC Error: {e}")
        auc_score = float("nan")

    # --- 7. RETURN ---
    return {
        "micro_averaged_metrics": final_micro_metrics,
        "macro_averaged_metrics": final_macro_metrics,
        "auc": float(auc_score),
        "bootstrap": {
            "ran": bool(run_bootstrap),
            "n_patients": int(n_patients),
            "n_bootstrap_samples": int(n_bootstrap_samples) if run_bootstrap else 0,
            "seed": int(SEED),
        },
        "confusion_matrix": {
            "tp": int(tp_micro), "fp": int(fp_micro), "fn": int(fn_micro), "tn": int(tn_micro),
        },
        "normalization": {
            "mean": [float(x) for x in train_mean],
            "std":  [float(x) for x in train_std],
        },
        "ensemble": {
            "method": str(strategy),
            "threshold": float(optimal_threshold),
            "weights": None, # Weights are embedded in role logic now
        },
        "macro_dice_rule6_split": {
          "dice_pos_only": {
              "point_estimate": float(macro_dice_pos_only_pe),
              "ci": [float(macro_dice_pos_only_ci[0]), float(macro_dice_pos_only_ci[1])],
          },
          "neg_clean_rate": {
              "point_estimate": float(macro_neg_clean_rate_pe),
              "ci": [float(macro_neg_clean_rate_ci[0]), float(macro_neg_clean_rate_ci[1])],
          },
          "n_pos_patients": int(n_pos_patients),
          "n_neg_patients": int(n_neg_patients),
        },
    }

def visualize_ensemble_predictions(
    models_list,
    dataloader,
    device,
    threshold,
    num_samples,
    train_mean,
    train_std,
    constituent_models_info
):

    num_models = len(models_list)
    if num_models == 0:
        print("No models provided.")
        return

    if constituent_models_info is None or len(constituent_models_info) != num_models:
        raise ValueError("constituent_models_info must be provided and aligned with models_list.")

    # Strict recipe-driven weights
    w = torch.tensor(
        [m["weight"] for m in constituent_models_info],
        device=device,
        dtype=torch.float32,
    )

    w = w / (w.sum() + 1e-12)
    w_cpu = w.detach().cpu().numpy().tolist()

    for model in models_list:
        model.eval()

    print(f"Visualizing {num_samples} ensemble samples | thr={threshold:.4f}")

    with torch.no_grad():
        try:
            batch_data = next(iter(dataloader))
        except StopIteration:
            print("DataLoader empty.")
            return

        if batch_data is None:
            print("Cannot load batch (collate_fn returned None).")
            return

        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            images, masks, patient_ids = batch_data
            images_ctx = None
        elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 4:
            images, images_ctx, masks, patient_ids = batch_data
        else:
            raise ValueError(f"Unexpected batch tuple size: {len(batch_data)}")

        if images is None or images.shape[0] == 0:
            print("Empty image batch.")
            return

        actual_batch_size = images.shape[0]
        num_samples = min(num_samples, actual_batch_size)
        sample_indices = random.sample(range(actual_batch_size), num_samples)

        images_vis = images[sample_indices].to(device, non_blocking=True)
        images_vis = gpu_normalizer(images_vis) # Becomes Float32 here

        masks_vis = masks[sample_indices]  # keep on CPU
        true_classes = _mask_to_binary_indices(masks_vis).cpu().numpy().astype(np.uint8)

        # --- Compute per-model probabilities on these samples ---
        model_probs = []
        for m_idx, model in enumerate(models_list):
            meta = (constituent_models_info[m_idx]
                    if constituent_models_info is not None and m_idx < len(constituent_models_info)
                    else {})
            arch = meta.get("architecture", "")
            use_amp = arch not in DISABLE_AMP

            probs_cancer = predict_with_tta_batched(model, images_vis, use_amp=use_amp)  # [N,H,W]
            model_probs.append(probs_cancer)

        # Stack to [M,N,H,W]

        cleaned_model_probs = [torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0) for p in model_probs]
        stack_probs = torch.stack(cleaned_model_probs, dim=0)

        # Always compute a continuous ensemble probability score (useful for heatmaps)
        ensemble_probs_for_auc = (stack_probs * w[:, None, None, None]).sum(dim=0).clamp(0, 1)  # [N,H,W]

        ensemble_probs_cancer = ensemble_probs_for_auc

        ensemble_preds = (
            ensemble_probs_cancer >= float(threshold)
        ).to(torch.uint8).detach().cpu().numpy()

        # --- Prepare images for plotting (de-normalize) ---
        images_np = images_vis.detach().cpu().numpy()  # [N,C,H,W]
        mean = np.array(train_mean, dtype=np.float32)
        std = np.array(train_std, dtype=np.float32)

        for j in range(num_samples):
            img = images_np[j].transpose(1, 2, 0)  # HWC
            img = std * img + mean
            img = np.clip(img, 0, 1)

            pred_mask = ensemble_preds[j]
            true_mask = true_classes[j]

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")

            # Pred overlay
            axes[1].imshow(img)
            axes[1].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="jet", alpha=0.5)
            axes[1].set_title(f"Ensemble Pred (method='Transformer-CNN Gating')")
            axes[1].axis("off")

            # True overlay
            axes[2].imshow(img)
            axes[2].imshow(np.ma.masked_where(true_mask == 0, true_mask), cmap="jet", alpha=0.5)
            axes[2].set_title("True Mask")
            axes[2].axis("off")

            # Probability / score heatmap
            prob_map = ensemble_probs_cancer[j].detach().cpu().numpy()
            axes[3].imshow(prob_map, vmin=0, vmax=1)
            axes[3].set_title("Ensemble Score (0..1)")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()

def print_scientific_analysis_report(metrics_results):
    """
    Prints a detailed, objective, and scientific guide for interpreting the model's
    performance, now including explanations for patient-level bootstrapping and CIs.
    """
    if not metrics_results:
        print("\nMetrics object is empty. Cannot generate report.")
        return

    # --- Unpack all the necessary data ---
    micro = metrics_results.get('micro_averaged_metrics', {})
    macro = metrics_results.get('macro_averaged_metrics', {})
    auc = metrics_results.get('auc')

    if not micro or not macro or auc is None:
        print("\nMetrics object is missing required data. Cannot generate full report.")
        return

    micro_pe = micro.get('point_estimate', {})
    micro_ci = micro.get('ci', {})
    macro_pe = macro.get('point_estimate', {})
    macro_ci = macro.get('ci', {})
    split = metrics_results.get("macro_dice_rule6_split", {})
    dice_pos = split.get("dice_pos_only", {})
    neg_clean = split.get("neg_clean_rate", {})

    # --- Helper to format the CI string dynamically ---
    def get_ci_string(ci_data):
        if ci_data is not None and not np.isnan(ci_data[0]):
            return f"(95% CI: [{ci_data[0]:.4f}, {ci_data[1]:.4f}])"
        else:
            return "(CI not calculated; n_patients < 20)"

    print("\n" + "="*30 + " Scientific Performance Analysis " + "="*30)
    print("\nThis report provides a scientific context for the model's performance on the hold-out test set.")
    print("It emphasizes patient-level generalization and statistical uncertainty.")

    print("\n\n" + "-"*25 + " Analysis of Key Metrics " + "-"*25)

    # --- 1. Segmentation Quality (Dice & IoU) ---
    print("\n[--- Segmentation Quality: Dice and IoU ---]")
    print("  - Definition: Measures the spatial overlap between predicted and true masks (Ideal = 1.0).")
    print("  - Micro-Average: Aggregates all pixels from all patients. This reflects overall pixel-level accuracy")
    print("    but can be dominated by patients who contributed a large number of patches.")
    print(f"    - Micro-Average Dice: {micro_pe.get('dice', 0):.4f} {get_ci_string(micro_ci.get('dice'))}")

    print("\n  - Macro-Average: Calculates the metric for each patient first, then averages these scores. This is the")
    print("    primary metric for clinical generalization, as it treats each patient equally.")
    print(f"    - Macro-Average Dice (Per-Patient Mean): {macro_pe.get('dice', 0):.4f} {get_ci_string(macro_ci.get('dice'))}")

    print("\n  - Interpretation of the 95% Confidence Interval (CI): The CI provides a plausible range for the true")
    print("    performance metric. A narrow CI suggests that the model's performance is stable and consistent")
    print("    across different patients in the test set.")

    # --- 2. Clinical Reliability: Sensitivity (TPR) and Miss Rate (FNR) ---
    print("\n[--- Clinical Reliability: Sensitivity / Miss Rate ---]")
    print("  - Definition (FNR): The False Negative Rate, or 'Miss Rate' (Ideal = 0.0). It is the proportion of")
    print("    cancerous regions/patients that the model failed to detect.")
    print("  - Interpretation: This metric is critical for clinical safety. A low FNR is essential for a screening tool.")
    print(f"  - The model's Macro-Average FNR is {macro_pe.get('fnr', 0):.4f} {get_ci_string(macro_ci.get('fnr'))}. This suggests that, on average,")
    print(f"    the model is expected to miss approximately {macro_pe.get('fnr', 0):.2%} of cancerous patients/regions.")

    # --- 3. Clinical Reliability: Specificity (TNR) and False Alarm Rate (FPR) ---
    print("\n[--- Clinical Reliability: Specificity / False Alarm Rate ---]")
    print("  - Definition (FPR): The False Positive Rate, or 'False Alarm Rate' (Ideal = 0.0). It is the proportion")
    print("    of healthy regions/patients that were incorrectly flagged as cancerous.")
    print("  - Interpretation: This metric is important for clinical efficiency, reducing unnecessary reviews.")
    print(f"  - The model's Macro-Average FPR is {macro_pe.get('fpr', 0):.4f} {get_ci_string(macro_ci.get('fpr'))}. This suggests that, on average,")
    print(f"    an estimated {macro_pe.get('fpr', 0):.2%} of non-cancerous patients/regions would trigger a false alarm.")

    # --- 4. Overall Discriminative Power (AUC) ---
    print("\n[--- Overall Discriminative Power: AUC ---]")
    print("  - Definition: The Area Under the ROC Curve measures the model's ability to distinguish between")
    print("    positive and negative pixels across all possible thresholds (Ideal = 1.0).")
    print(f"  - The model's pixel-level AUC is {auc:.4f}. As a benchmark, values above 0.95 typically reflect")
    print("    excellent discriminative power between the classes.")

    print("\n\n" + "="*25 + " How to Form a Conclusion " + "="*25)
    print("A robust and generalizable model demonstrates a combination of strengths:")
    print("  1. High Technical Skill: Indicated by a high Micro-Average Dice and a high AUC.")
    print("  2. High Generalization to New Patients: Indicated by a strong Macro-Average (per-patient) Dice score.")
    print("  3. High Safety & Sensitivity: Indicated by a low Macro-Average FNR.")
    print("  4. High Efficiency & Specificity: Indicated by a low Macro-Average FPR.")
    print("  5. High Confidence: Indicated by narrow 95% Confidence Intervals on the key macro-average metrics.")
    print("\nEvaluate these metrics based on the intended clinical application. For a screening tool, a low")
    print("FNR and its upper CI bound are paramount. For a confirmatory tool, a low FPR may be more critical.")
    print("="*80)
    print(f"    - Macro-Average Dice (All Patients): {macro_pe.get('dice', 0):.4f} {get_ci_string(macro_ci.get('dice'))}")
    print(f"    - Macro Dice (Pos-only, TP+FN>0): {dice_pos.get('point_estimate', float('nan')):.4f} {get_ci_string(dice_pos.get('ci', [np.nan, np.nan]))}")
    print(f"    - Neg Clean Rate (GT empty, FP==0): {neg_clean.get('point_estimate', float('nan')):.4f} {get_ci_string(neg_clean.get('ci', [np.nan, np.nan]))}")
    print(f"    - N pos patients: {split.get('n_pos_patients', 0)}, N neg patients: {split.get('n_neg_patients', 0)}")

# You can add this function definition after the `print_scientific_analysis_report` function

def export_results_to_csv(ensemble_recipe, metrics_results, output_dir):
    """
    Exports the complete configuration and final results of the ensemble evaluation
    to a comprehensive and machine-readable CSV file.
    """
    if not metrics_results or not ensemble_recipe:
        print("Cannot export results to CSV: Missing metrics or ensemble recipe.")
        return

    timestamp = get_formatted_datetime_string()
    filename = f"FINAL_EVALUATION_REPORT_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    print(f"\n--- Exporting final report to CSV: {filepath} ---")

    # --- 1. Prepare Data for Export ---

    # Section 1: Experiment Configuration
    config_data = {
        'Parameter': [
            'Experiment Datetime',
            'Ensemble Recipe Path',
            'Dataset ZIP Directory',
            'Evaluation Set',
            'Random Seed',
            'Batch Size',
            'Optimal Ensemble Threshold'
        ],
        'Value': [
            timestamp,
            ENSEMBLE_META_PATH, # Global variable
            HDF5_DRIVE_DIR,    # Global variable
            'TEST',
            SEED,               # Global variable
            BATCH_SIZE,         # Global variable
            (ensemble_recipe.get('roi_config')).get('threshold', "FAILED")
        ]
    }
    config_df = pd.DataFrame(config_data)

    # Section 2: Ensemble Composition
    comp_models = ensemble_recipe.get('model_registry', [])
    composition_data = {
        'Model Index': [f"Model {i+1}" for i in range(len(comp_models))],
        'Architecture': [m.get('architecture') for m in comp_models],
        'Encoder': [m.get('encoder') for m in comp_models],
        'Ensemble Weight': [m.get('weight') for m in comp_models],
        'Source Checkpoint': [os.path.basename(m.get('checkpoint_path', '')) for m in comp_models],
        'Stream Role': [m.get('stream_role', '') for m in comp_models]
    }
    composition_df = pd.DataFrame(composition_data)

    # Section 3: Performance Metrics
    micro = metrics_results.get('micro_averaged_metrics', {})
    macro = metrics_results.get('macro_averaged_metrics', {})
    auc = metrics_results.get('auc')

    metric_keys = sorted(micro.get('point_estimate', {}).keys())

    metrics_data = {
        'Metric Type': [],
        'Metric': [],
        'Point Estimate': [],
        'CI Lower (2.5%)': [],
        'CI Upper (97.5%)': []
    }

    # Add Micro-Averages
    for key in metric_keys:
        metrics_data['Metric Type'].append('Micro-Average (Pixel Level)')
        metrics_data['Metric'].append(key.title())
        metrics_data['Point Estimate'].append(micro.get('point_estimate', {}).get(key))
        ci = micro.get('ci', {}).get(key, [np.nan, np.nan])
        metrics_data['CI Lower (2.5%)'].append(ci[0])
        metrics_data['CI Upper (97.5%)'].append(ci[1])

    # Add AUC
    metrics_data['Metric Type'].append('Micro-Average (Pixel Level)')
    metrics_data['Metric'].append('AUC')
    metrics_data['Point Estimate'].append(auc)
    metrics_data['CI Lower (2.5%)'].append(np.nan)
    metrics_data['CI Upper (97.5%)'].append(np.nan)

    # Add Macro-Averages
    for key in metric_keys:
        metrics_data['Metric Type'].append('Macro-Average (Patient Level)')
        metrics_data['Metric'].append(key.title())
        metrics_data['Point Estimate'].append(macro.get('point_estimate', {}).get(key))
        ci = macro.get('ci', {}).get(key, [np.nan, np.nan])
        metrics_data['CI Lower (2.5%)'].append(ci[0])
        metrics_data['CI Upper (97.5%)'].append(ci[1])

    metrics_df = pd.DataFrame(metrics_data)

    # --- 2. Write to CSV File ---
    try:
        with open(filepath, 'w', newline='') as f:
            f.write("--- Experiment Configuration ---\n")
            config_df.to_csv(f, index=False)

            f.write("\n--- Ensemble Composition ---\n")
            composition_df.to_csv(f, index=False)

            f.write("\n--- Final Performance Metrics ---\n")
            metrics_df.to_csv(f, index=False)

        print(f"Successfully exported detailed results to {filepath}")
    except Exception as e:
        print(f"Failed to export results to CSV: {e}")

def parse_ensemble_recipe(recipe: dict):
    """
    Parses the declarative JSON.
    Returns:
        roi_thr (float): Threshold for the Semantic stream to generate ROI.
        model_configs (list): List of dicts with model paths and metadata.
        strategy (str): 'weighted_average' or 'two_stream_spatial_gating'.
    """
    # 1. Detect Strategy
    strategy = recipe.get("ensemble_strategy")
    print(f"Ensemble Strategy detected: {strategy.upper()}")

    # 2. Get ROI Threshold
    roi_thr = 0.5 # default
    if "roi_config" in recipe:
        roi_thr = float(recipe["roi_config"].get("threshold", 0.5))
    else:
        raise KeyError("JSON missing 'roi_config'.")

    # 3. Parse Models from Registry
    if "model_registry" in recipe:
        models_data = recipe["model_registry"]
    else:
        raise KeyError("JSON missing 'model_registry'")

    return roi_thr, models_data, strategy

def load_checkpoint_strict_without_aux(model, chkpt_path, device):
    chkpt = torch.load(chkpt_path, map_location="cpu")
    state_dict = chkpt.get("model_state_dict", chkpt)

    # Drop aux head keys if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classification_head.")}

    # Handle torch.compile prefix mismatch
    model_keys = list(model.state_dict().keys())
    sd_keys = list(state_dict.keys())
    if model_keys and sd_keys:
        model_has = model_keys[0].startswith("_orig_mod.")
        sd_has = sd_keys[0].startswith("_orig_mod.")
        if model_has and not sd_has:
            state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        elif (not model_has) and sd_has:
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    if chkpt.get("is_compiled", False):
        print("[INFO] Re-compiling model (checkpoint was compiled)")
        try:
          model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    return model

def save_confusion_matrix_png(tp, fp, fn, tn, out_path_png):
    """
    Saves row-normalized confusion matrix as a PNG suitable for papers.
    """
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=['Pred NoCancer', 'Pred Cancer'],
        yticklabels=['True NoCancer', 'True Cancer']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensemble Confusion Matrix (Row-Normalized %)')
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path_png

def _tex_escape(s: str) -> str:
    # minimal escape for paths / underscores
    return (s.replace('\\', r'\textbackslash ')
             .replace('_', r'\_')
             .replace('%', r'\%')
             .replace('&', r'\&')
             .replace('#', r'\#')
             .replace('{', r'\{')
             .replace('}', r'\}'))

def fmt_ci(ci_pair, ran_bootstrap: bool):
    """
    Return CI string: [low, high] if bootstrap ran and CI is valid, else NA.
    """
    if not ran_bootstrap:
        return "NA"
    if ci_pair is None:
        return "NA"
    lo, hi = ci_pair
    if (lo is None) or (hi is None) or np.isnan(lo) or np.isnan(hi):
        return "NA"
    return f"[{lo:.4f}, {hi:.4f}]"

def fmt_float(x):
    if x is None:
        return "NA"
    try:
        if np.isnan(x):
            return "NA"
    except Exception:
        pass
    return f"{float(x):.4f}"

def write_ensemble_report_latex(
    ensemble_recipe: dict,
    ensemble_metrics: dict,
    train_mean, train_std,
    stats_sample_size: int,
    cm_png_path: str,
    output_dir: str,
    report_name_prefix: str = "FINAL_ENSEMBLE_REPORT"
):
    """
    Writes report.tex and compiles to report.pdf in output_dir.
    Includes:
      - micro & macro metrics with point estimate + CI/NA
      - normalization mean/std
      - confusion matrix figure
      - ensemble composition table
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    tex_path = os.path.join(output_dir, f"{report_name_prefix}_{ts}.tex")
    pdf_path = os.path.join(output_dir, f"{report_name_prefix}_{ts}.pdf")

    # --- Pull bootstrap info safely ---
    bootstrap = ensemble_metrics.get("bootstrap", {})
    ran_bootstrap = bool(bootstrap.get("ran", False))
    n_patients = bootstrap.get("n_patients", None)
    n_boot = bootstrap.get("n_bootstrap_samples", None)
    boot_seed = bootstrap.get("seed", None)

    micro = ensemble_metrics.get("micro_averaged_metrics", {})
    macro = ensemble_metrics.get("macro_averaged_metrics", {})
    auc = ensemble_metrics.get("auc", None)

    micro_pe = micro.get("point_estimate", {})
    micro_ci = micro.get("ci", {})
    macro_pe = macro.get("point_estimate", {})
    macro_ci = macro.get("ci", {})

    # --- NEW: Rule-6 Dice Reporting (Seghier 2024) ---
    rule6 = ensemble_metrics.get("macro_dice_rule6_split", {})
    dice_pos = rule6.get("dice_pos_only", {})
    neg_clean = rule6.get("neg_clean_rate", {})

    # Metrics order (paper-friendly)
    metric_order = ["dice", "iou", "tpr", "tnr", "precision", "accuracy", "fpr", "fnr"]

    # Ensemble composition
    comp_models = ensemble_recipe.get("model_registry", [])
    # Try threshold key variants
    thr = (ensemble_recipe.get("roi_config")).get("threshold", None)

    # --- LaTeX content ---
    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{hyperref}}

\title{{Ensemble Final Evaluation Report}}
\author{{Automated Pipeline}}
\date{{{ts}}}

\begin{{document}}
\maketitle

\section*{{Experiment Summary}}
\begin{{itemize}}
  \item Ensemble Strategy (from recipe): {_tex_escape(str(ensemble_recipe.get("ensemble_strategy", "NA")))}
  \item Threshold (recipe): {fmt_float(thr)}
  \item Bootstrap ran: {"Yes" if ran_bootstrap else "No"} \\
        (n\_patients={_tex_escape(str(n_patients))}, n\_bootstrap={_tex_escape(str(n_boot))}, seed={_tex_escape(str(boot_seed))})
  \item AUC (pixel-level ROC): {fmt_float(auc)}
\end{{itemize}}

\section*{{Normalization Statistics}}
Computed on TRAIN set (post-stain-normalization), then applied to TEST.
\begin{{itemize}}
  \item Mean (RGB): [{train_mean[0]:.6f}, {train_mean[1]:.6f}, {train_mean[2]:.6f}]
  \item Std (RGB): [{train_std[0]:.6f}, {train_std[1]:.6f}, {train_std[2]:.6f}]
\end{{itemize}}

\section*{{Ensemble Composition}}
\begin{{table}}[H]
\centering
\caption{{Constituent models and weights (from recipe).}}
\begin{{tabular}}{{r l l r}}
\toprule
\# & Architecture & Encoder & Weight \\
\midrule
"""
    # rows
    for i, m in enumerate(comp_models, start=1):
        arch = _tex_escape(str(m.get("architecture", "NA")))
        enc  = _tex_escape(str(m.get("encoder", "NA")))
        w    = m.get("weight", None)
        tex += rf"{i} & {arch} & {enc} & {fmt_float(w)} \\ " + "\n"

    tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Micro table
    tex += r"""
\section*{Final Metrics (Micro / Pixel-Level)}
\begin{table}[H]
\centering
\caption{Micro-averaged metrics with 95\% CI (NA if bootstrap did not run).}
\begin{tabular}{l r l}
\toprule
Metric & Point Estimate & 95\% CI \\
\midrule
"""
    for k in metric_order:
        pe = micro_pe.get(k, None)
        ci = micro_ci.get(k, None)
        tex += rf"{_tex_escape(k.upper())} & {fmt_float(pe)} & {fmt_ci(ci, ran_bootstrap)} \\ " + "\n"
    tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Macro table
    tex += r"""
\section*{Final Metrics (Macro / Patient-Level)}
\begin{table}[H]
\centering
\caption{Macro-averaged (per-patient) metrics with 95\% CI (NA if bootstrap did not run).}
\begin{tabular}{l r l}
\toprule
Metric & Point Estimate & 95\% CI \\
\midrule
"""
    for k in metric_order:
        pe = macro_pe.get(k, None)
        ci = macro_ci.get(k, None)
        tex += rf"{_tex_escape(k.upper())} & {fmt_float(pe)} & {fmt_ci(ci, ran_bootstrap)} \\ " + "\n"
    tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    #ADD
    tex += rf"""
\subsection*{{Dice Index (Rule 6: Split Reporting)}}
Following Seghier (2024), Dice is reported separately for patients with and without positive ground truth.

\begin{{tabular}}{{l c}}
\toprule
Metric & Value \\
\midrule
Dice (Positive GT only) &
{dice_pos.get('point_estimate', float('nan')):.4f}
[{dice_pos.get('ci', [float('nan'), float('nan')])[0]:.4f},
 {dice_pos.get('ci', [float('nan'), float('nan')])[1]:.4f}] \\

Negative Clean Rate (FP=0 | GT empty) &
{neg_clean.get('point_estimate', float('nan')):.4f}
[{neg_clean.get('ci', [float('nan'), float('nan')])[0]:.4f},
 {neg_clean.get('ci', [float('nan'), float('nan')])[1]:.4f}] \\
\bottomrule
\end{{tabular}}

\noindent
Number of patients with lesions: {rule6.get('n_pos_patients', 0)} \\
Number of patients without lesions: {rule6.get('n_neg_patients', 0)}
"""
    # Confusion matrix figure
    cm_png_rel = os.path.basename(cm_png_path)
    tex += rf"""
\section*{{Confusion Matrix}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.75\linewidth]{{{_tex_escape(cm_png_rel)}}}
\caption{{Row-normalized confusion matrix (percent).}}
\end{{figure}}
"""

    tex += r"""
\end{document}
"""

    # --- Write .tex and copy cm png to same folder (so LaTeX finds it) ---
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    # Ensure figure is in same dir as tex
    cm_target = os.path.join(output_dir, os.path.basename(cm_png_path))
    if os.path.abspath(cm_png_path) != os.path.abspath(cm_target):
        import shutil
        shutil.copy2(cm_png_path, cm_target)

    # --- Compile to PDF (Colab has pdflatex usually; if not, you'll see the error) ---
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)],
            cwd=output_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # pdflatex outputs PDF with same basename as tex
        built_pdf = os.path.join(output_dir, os.path.splitext(os.path.basename(tex_path))[0] + ".pdf")
        if os.path.exists(built_pdf):
            os.rename(built_pdf, pdf_path)
        print(f"[OK] PDF generated: {pdf_path}")
    except Exception as e:
        print("[WARN] pdflatex failed. The .tex was still created so you can compile manually.")
        print(f"       TeX file: {tex_path}")
        print(f"       Error: {e}")

    return tex_path, pdf_path

def setup_local_hdf5(drive_dir, local_dir):
    """Copies HDF5 files from Drive to local VM for speed."""
    print(f"\n{'='*25} Setting up HDF5 Data {'='*25}")
    if os.path.exists(local_dir): shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # We strictly need TEST.h5 for final evaluation
    filename = "TEST.h5"
    resolved_drive_dir = change_path(drive_dir)

    src = os.path.join(resolved_drive_dir, filename)
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
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                # GaussNoise (Strict 1.4 API: std_range as fraction of 255)
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1.0),

                # REPLACEMENT: Downscale (Strict 1.4 API + nnU-Net Logic)
                # nnU-Net: downsample factor U(1, 2) -> scale 0.5 to 1.0
                # nnU-Net: Down=Nearest, Up=Cubic
                A.Downscale(
                    scale_range=(0.5, 0.9),
                    interpolation_pair={
                        "downscale": cv2.INTER_NEAREST,
                        "upscale": cv2.INTER_CUBIC
                    },
                    p=1.0
                ),
            ], p=0.2),

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

class ProstateCancerDatasetHDF5(Dataset):
    def __init__(
        self,
        hdf5_path,
        mode="validation",
        return_context=False, # Required for Stage 2 Gating
        context_scale=1       # Required for LowPass simulation
    ):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.return_context = return_context
        self.context_scale = int(context_scale)

        self.transform = get_transforms(mode=mode, img_size=224)

        # --- METADATA LOADING ---
        with h5py.File(self.hdf5_path, 'r') as f:
            self.full_pids = f["patient_ids"][:]
            self.total_len = len(self.full_pids)

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
        return self.total_len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._open_file()

        # Map subset index -> real index in the HDF5
        real_idx = self.indices[idx] if hasattr(self, "indices") and self.indices is not None else idx

        # 1) Load raw data
        image = self.images_dset[real_idx]  # uint8 [H,W,3]
        mask  = self.masks_dset[real_idx]   # uint8 [H,W]
        if not hasattr(self, "full_pids"):
            raise KeyError(
                "[InferenceDataset] HDF5 is missing 'patient_ids' (full_pids). "
                "Patient-level metrics are invalid without it."
            )
        patient_id = str(self.full_pids[real_idx])

        # 2) Prepare SINGLE-channel mask (uint8 0/1). No one-hot anywhere.
        #    We keep it uint8 to reduce CPU work and memory traffic.
        mask01 = (mask != 0).astype(np.uint8)  # [H,W] values {0,1}

        try:
            # 3) Augmentations / ToTensorV2 (for test/val this is deterministic)
            augmented = self.transform(image=image, mask=mask01)

            final_image = augmented["image"]  # Tensor [3,H,W]
            final_mask  = augmented["mask"]   # Tensor [H,W] (or ndarray [H,W] depending on pipeline)

            # Ensure tensor + dtype contract: uint8 {0,1}
            if not torch.is_tensor(final_mask):
                final_mask = torch.from_numpy(final_mask)
            # final_mask may come as long/int; force uint8 for cheap transfer/ops
            final_mask = final_mask.to(torch.uint8)

            # 4) Optional context gating (low-pass simulation) — same contract as optimizer
            if getattr(self, "return_context", False):

                _, h, w = final_image.shape
                scale = max(1, int(getattr(self, "context_scale", 1)))
                small_h, small_w = max(1, h // scale), max(1, w // scale)

                img_unsqueezed = final_image.unsqueeze(0)
                ctx_small = F.interpolate(img_unsqueezed, size=(small_h, small_w),
                                          mode="bilinear", align_corners=False)
                ctx_full  = F.interpolate(ctx_small, size=(h, w),
                                          mode="bilinear", align_corners=False)
                final_context = ctx_full.squeeze(0)

                # Return 4 items: Image, Context, Mask(H,W), PID
                return final_image, final_context, final_mask, patient_id

            # Return 3 items: Image, Mask(H,W), PID
            return final_image, final_mask, patient_id

        except Exception as e:
            print(f"Error on index {idx} (real_idx={real_idx}): {e}")
            if getattr(self, "return_context", False):
                return None, None, None, None
            return None, None, None


            # 4) Optional context gating (low-pass simulation) — same contract as optimizer
            if getattr(self, "return_context", False):

                _, h, w = final_image.shape
                scale = max(1, int(getattr(self, "context_scale", 1)))
                small_h, small_w = max(1, h // scale), max(1, w // scale)

                img_unsqueezed = final_image.unsqueeze(0)
                ctx_small = F.interpolate(img_unsqueezed, size=(small_h, small_w),
                                          mode="bilinear", align_corners=False)
                ctx_full  = F.interpolate(ctx_small, size=(h, w),
                                          mode="bilinear", align_corners=False)
                final_context = ctx_full.squeeze(0)

                # Return 4 items: Image, Context, Mask, PID
                return final_image, final_context, final_mask, patient_id

            # Return 3 items: Image, Mask, PID
            return final_image, final_mask, patient_id

        except Exception as e:
            print(f"Error on index {idx} (real_idx={real_idx}): {e}")
            if getattr(self, "return_context", False):
                return None, None, None, None
            return None, None, None

    # Required for stratifying holdout sets
    def get_patient_ids(self):
        return [str(p) for p in self.full_pids]


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

# --- SHARED: Updated Collate Function ---
def collate_fn(batch):
    # Filter None
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None

    # Detect tuple size from first valid item
    sample_len = len(batch[0])

    if sample_len == 3:
        # Standard: (img, mask, pid)
        images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
        masks = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
        pids = [b[2] for b in batch]
        return images, masks, pids

    elif sample_len == 4:
        # Context: (img, img_ctx, mask, pid)
        images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
        images_ctx = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
        masks = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
        pids = [b[3] for b in batch]
        return images, images_ctx, masks, pids

    elif sample_len == 5:
        images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
        images_ctx = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
        masks = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
        pids = [b[3] for b in batch]
        pos = torch.as_tensor([b[4] for b in batch], dtype=torch.long)
        return images, images_ctx, masks, pids, pos

    else:
        raise ValueError(f"Unexpected batch item length: {sample_len}")

# --- INSTANTIATE GPU NORMALIZER GLOBALLY ---
gpu_normalizer = GPUNormalizer(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    device=device
)

ENSEMBLE_META_PATH = change_path(ENSEMBLE_META_PATH)
HDF5_DRIVE_DIR = change_path(HDF5_DRIVE_DIR)

# ### REFACTORED: Step 1 - Load the pre-computed ensemble recipe (NEW, IMPROVED VERSION) ###
print(f"\nLoading ensemble recipe from: {ENSEMBLE_META_PATH}")
if not os.path.exists(ENSEMBLE_META_PATH):
    raise FileNotFoundError(f"Ensemble metadata file not found! Path: {ENSEMBLE_META_PATH}")

with open(ENSEMBLE_META_PATH, 'r') as f:
    ensemble_recipe = json.load(f)

# ==============================================================================
# --- 13. Main Training Loop ---
# ==============================================================================
# 1. Setup Data
test_h5_path = setup_local_hdf5(HDF5_DRIVE_DIR, LOCAL_DATA_DIR)

# --- DataLoaders (with optional stratified subsampling) ---
print("\nCreating DataLoaders...")

try:
    # Instantiate the HDF5 Dataset
    full_test_ds = ProstateCancerDatasetHDF5(
    hdf5_path=test_h5_path,
    mode="test",
    return_context=False,
    context_scale=1,
)

    # For reporting, we use the stats the dataset actually adopted
    train_mean = [0.485, 0.456, 0.406]
    train_std  = [0.229, 0.224, 0.225]


    test_loader = DataLoader(
    full_test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    num_workers=WORKERS,
    prefetch_factor=8,
    collate_fn=collate_fn,  # your collate already supports 3/4-tuples :contentReference[oaicite:3]{index=3}
)

    print(f"Test dataset size: {len(full_test_ds)}")
    print("DataLoaders created successfully.")

except Exception as e:
  print(f"DataLoader Err: {e}")
  import traceback; traceback.print_exc()
  clear_gpu()

optimal_threshold, constituent_models_info, _ = parse_ensemble_recipe(ensemble_recipe)
n_models = len(constituent_models_info)

print(f"Using Optimal Threshold: {optimal_threshold:.6f}")
print("Using Optimal Weights:")
for model in constituent_models_info:
    print(f"  {model['architecture']} - {model['weight']}")

# ### REFACTORED: Step 2 - Load the EXACT models from the recipe ###
ensemble_models = []
print(f"\nLoading the {n_models} constituent models...")
for i, model_meta in enumerate(constituent_models_info):
    arch = model_meta.get('architecture')
    enc = model_meta.get('encoder')
    chkpt_path = model_meta.get('checkpoint_path')
    chkpt_path = change_paths(chkpt_path)

    print(f" Loading Model {i+1}: {arch} ({enc}) from {os.path.basename(chkpt_path)}...")
    try:
        model = get_model(architecture=arch, encoder=enc, validation=True)
        model = load_checkpoint_strict_without_aux(model, chkpt_path, device)

        ensemble_models.append(model)
    except Exception as e:
        print(f"  FATAL ERROR loading model {os.path.basename(chkpt_path)}: {e}")
        # In a final run, a failure to load a required model should be a fatal error.
        exit(1)

if len(ensemble_models) != n_models:
    print("\nError: Number of loaded models does not match the recipe. Exiting.")
    exit(1)

print(f"\nSuccessfully loaded all {len(ensemble_models)} models.")

# # ### REFACTORED: Step 3 - Directly evaluate on the Test Set ###
# The val_loader and find_optimal_ensemble_threshold call are REMOVED.
print("\n--- Evaluating Ensemble Performance on Hold-Out Test Set ---")
ensemble_metrics = None
try:
  ensemble_metrics = analyze_ensemble_metrics(
      models_list=ensemble_models,
      constituent_models_info=constituent_models_info,
      test_loader=test_loader,
      device=device,
      optimal_threshold=optimal_threshold,
      train_mean=train_mean,
      train_std=train_std,
      ensemble_recipe=ensemble_recipe
  )
except Exception as e:
    print(f"Error during final ensemble evaluation: {e}")
    import traceback; traceback.print_exc()

if ensemble_metrics is None:
  print("\nEnsemble evaluation failed. Exiting.")
  exit(1)
else:
  cm_out_dir = os.path.dirname(ENSEMBLE_META_PATH)
  cm_out_png = os.path.join(cm_out_dir, "confusion_matrix.png")

  tp = ensemble_metrics.get('confusion_matrix', {}).get('tp', 0)
  fp = ensemble_metrics.get('confusion_matrix', {}).get('fp', 0)
  fn = ensemble_metrics.get('confusion_matrix', {}).get('fn', 0)
  tn = ensemble_metrics.get('confusion_matrix', {}).get('tn', 0)
  save_confusion_matrix_png(tp, fp, fn, tn, cm_out_png)

# --- Visualization & Reporting (Adjusted for final return structure) ---
if ensemble_metrics:
  visualize_ensemble_predictions(
      models_list=ensemble_models,
      dataloader=test_loader,
      device=device,
      threshold=optimal_threshold,
      num_samples=5,
      train_mean=train_mean,
      train_std=train_std,
      constituent_models_info=constituent_models_info
  )

  print("\n" + "="*20 + " Final Summary From Returned Object " + "="*20)

  # Extract the main results dictionaries
  micro_results = ensemble_metrics.get('micro_averaged_metrics')
  macro_results = ensemble_metrics.get('macro_averaged_metrics')
  auc_score = ensemble_metrics.get('auc')

  if not micro_results or not macro_results:
      print("Evaluation produced no valid metrics.")
  else:
      # --- Print the Micro-Averages with their CIs ---
      print("\n--- Overall Pixel-Level Metrics (Micro-Averages) ---")
      micro_pes = micro_results.get('point_estimate', {})
      micro_cis = micro_results.get('ci', {})

      for metric in sorted(micro_pes.keys()):
          pe = micro_pes[metric]
          ci = micro_cis.get(metric, [0.0, 0.0])
          print(f"  {metric.replace('_',' ').title():<12}: {pe:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

      # Report the AUC score
      if auc_score is not None:
          print(f"  {'AUC':<12}: {auc_score:.4f}  (CI not calculated via bootstrap)")


      # --- Print the Macro-Averages with their CIs ---
      print("\n--- Mean Per-Image Metrics (Macro-Averages) ---")
      macro_pes = macro_results.get('point_estimate', {})
      macro_cis = macro_results.get('ci', {})

      for metric in sorted(macro_pes.keys()):
          pe = macro_pes[metric]
          ci = macro_cis.get(metric, [0.0, 0.0])
          print(f"  {metric.replace('_',' ').title():<12}: {pe:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

else:
    print("\nEnsemble evaluation failed.")

# --- ADD THE CUSTOMIZED REPORT CALL HERE ---
if ensemble_metrics:
    print_scientific_analysis_report(ensemble_metrics)
    # --- NEW: Call the CSV export function ---
    export_results_to_csv(
        ensemble_recipe=ensemble_recipe,
        metrics_results=ensemble_metrics,
        output_dir=os.path.dirname(ENSEMBLE_META_PATH) # Save CSV in the same folder as the recipe
    )

print("\n--- Final Evaluation Script Finished ---")

# 2) Write + compile LaTeX report
tex_path, pdf_path = write_ensemble_report_latex(
    ensemble_recipe=ensemble_recipe,
    ensemble_metrics=ensemble_metrics,
    train_mean=train_mean,
    train_std=train_std,
    stats_sample_size=0,  # <- put the same value you used in dataset stats
    cm_png_path=cm_out_png,
    output_dir=os.path.dirname(ENSEMBLE_META_PATH),
)
print("TeX:", tex_path)
print("PDF:", pdf_path)
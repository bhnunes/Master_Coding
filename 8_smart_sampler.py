import os
import sys
import json
import gc
import random
import time
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import shutil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

# Suppress minor warnings for cleaner scientific logs
warnings.filterwarnings("ignore", category=UserWarning)

# @title 2. Configuration & Determinism
# scientific_context: Single source of truth for hyperparameters and reproducibility.

class Config:
    # --- DRIVE PATHS (Permanent Storage) ---
    # Where your data lives now
    DRIVE_SOURCE_PATH = "/content/drive/MyDrive/IA_MEDICA_SAMPLES/CAMELYON16/TRAIN.h5"
    # Where you want the results saved
    DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/IA_MEDICA_SAMPLES/CAMELYON16"

    # --- LOCAL PATHS (Colab Ephemeral NVMe) ---
    # We will copy data here for fast processing
    LOCAL_WORK_DIR = "/content/workspace"
    TRAIN_H5_PATH = os.path.join(LOCAL_WORK_DIR, "TRAIN.h5")
    OUTPUT_DIR = os.path.join(LOCAL_WORK_DIR, "output")
    OUTPUT_FILENAME = "TRAIN_FILTERED.h5"

    # --- Encoder (SMP) ---
    ENCODER_NAME = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stability Loop (Adaptive N) ---
    N_START = 512
    N_MAX = 15000
    GROWTH_FACTOR = 2.0
    STABILITY_THRESHOLD = 0.85
    STABILITY_REPEATS = 3
    MAX_STEPS = 7
    INTERSECTION_RATIO_THRESHOLD = 0.2

    # --- Clustering & Selection ---
    K_MIN = 20
    K_MAX = 80
    M_MAX = 2000
    SELECTION_STRATEGY = "uniform"

    # --- System ---
    SEED = 42
    NUM_WORKERS = 2
    USE_CACHE = True
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

def seed_everything(seed: int):
    """Enforces strict determinism across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Global seed set to: {seed}")

# @title 3. Data Loading Architecture

class H5PatchDataset(Dataset):
    """
    Reads patches on-the-fly from HDF5.
    Only reads the specific indices requested to minimize IO overhead.
    """
    def __init__(self, h5_path: str, indices: np.ndarray, transform=None):
        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open file in every worker process (h5py is not fork-safe)
        with h5py.File(self.h5_path, 'r') as f:
            # Sort indices to minimize disk seek time if batching,
            # though DataLoader shuffles. Here we access single idx.
            global_idx = self.indices[idx]

            # Handle Layout A: (N, H, W, C) or (N, C, H, W)
            img_data = f['images'][global_idx]

            # Ensure format is (H, W, C) for PIL/Transforms
            if img_data.shape[0] <= 4: # Likely (C, H, W)
                img_data = np.transpose(img_data, (1, 2, 0))

        # Preprocessing: uint8 -> float32 [0,1] is handled by ToTensor()
        if self.transform:
            img_data = self.transform(img_data)

        return img_data

def get_preprocessing_transforms(input_size):
    """
    Returns standard ImageNet normalization + Resize.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(), # Converts to [0,1] float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

class H5MetadataIndex:
    """
    Scans HDF5 once to build a patient_id -> [global_indices] map.
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.patient_map = {}
        self.total_samples = 0
        self._build_index()

    def _build_index(self):
        print(f"[Data] Indexing {self.h5_path}...")
        with h5py.File(self.h5_path, 'r') as f:
            pids = f['patient_ids'][:]
            self.total_samples = len(pids)

            # vectorized grouping
            df = pd.DataFrame({'pid': pids, 'idx': np.arange(len(pids))})
            self.patient_map = df.groupby('pid')['idx'].apply(np.array).to_dict()

        print(f"[Data] Indexed {len(self.patient_map)} patients, {self.total_samples} total patches.")

# @title 4. Embedding Engine

class EmbeddingExtractor:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model = self._init_model()
        self.transform = get_preprocessing_transforms(config.INPUT_SIZE)

    def _init_model(self):
        # Instantiate SMP model just for the encoder
        # We use Unet as a wrapper, but we will only call encoder
        model = smp.Unet(
            encoder_name=self.config.ENCODER_NAME,
            encoder_weights=self.config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=1
        )
        model.encoder.to(self.device)
        model.encoder.eval()
        return model.encoder

    @torch.no_grad()
    def get_embeddings(self, h5_path, indices):
        """
        Extracts embeddings for a specific list of global indices.
        Returns: numpy array (N, D)
        """
        if len(indices) == 0:
            return np.empty((0, 0))

        dataset = H5PatchDataset(h5_path, indices, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )

        embeddings_list = []

        for batch in loader:
            batch = batch.to(self.device)
            # SMP encoders return a list of stages. We usually want the last one.
            features = self.model(batch)
            last_map = features[-1] # Shape (B, C, H, W)

            # Global Average Pooling
            # mean over spatial dims (2,3)
            gap = torch.mean(last_map, dim=[2, 3])
            embeddings_list.append(gap.cpu().numpy())

        return np.vstack(embeddings_list)

# @title 5. Stability & Selection Algorithm

def compute_k(n_samples, config):
    """Defined rule: k = clamp(sqrt(n), 20, 80)."""
    if n_samples == 0: return 1
    raw_k = int(np.sqrt(n_samples))
    k = max(config.K_MIN, min(raw_k, config.K_MAX))

    # Corner case: very small n
    if n_samples < config.K_MIN:
        k = max(2, int(n_samples / 10))
        if k < 2: k = n_samples # Trivial case
    return k

def calculate_stability_score(emb_s1, emb_s2, idx_s1, idx_s2, config):
    """
    Computes ARI stability.
    Handles disjoint sets via 'Probe Set' method as per Section 4.3.3.
    """
    n = len(emb_s1)
    k = compute_k(n, config)

    # 1. Fit models
    mbk1 = MiniBatchKMeans(n_clusters=k, batch_size=512, n_init=3,
                           random_state=config.SEED, max_iter=100).fit(emb_s1)
    mbk2 = MiniBatchKMeans(n_clusters=k, batch_size=512, n_init=3,
                           random_state=config.SEED+1, max_iter=100).fit(emb_s2)

    # 2. Check intersection
    intersect_mask_1 = np.isin(idx_s1, idx_s2)
    intersect_mask_2 = np.isin(idx_s2, idx_s1)
    intersection_size = np.sum(intersect_mask_1)

    # 3. Decision: Intersection ARI vs Probe Set ARI
    if intersection_size > (config.INTERSECTION_RATIO_THRESHOLD * n):
        # Use intersection
        # We need to align labels.
        # Actually, ARI is independent of permutation, so we just need
        # labels for the SAME points from both models.

        # Get the common indices
        common_indices = idx_s1[intersect_mask_1]

        # We need to map global indices back to local embedding rows
        # Simplest way: predict on the intersection embeddings
        # (Or track positions. Predict is safer/easier).
        emb_intersection = emb_s1[intersect_mask_1] # This is technically S1's view of intersection

        labels1 = mbk1.predict(emb_intersection)
        labels2 = mbk2.predict(emb_intersection)

        score = adjusted_rand_score(labels1, labels2)
    else:
        # Use Probe Set (Centroid assignment)
        # Create a random probe set P from available data (can be S1 or S2 or mixed)
        # For simplicity, use S1 as the probe for model 2 and S2 for model 1?
        # Better: Use a fixed small random noise or just predict on S1 using both.
        # Spec suggests: "Fixed probe set P"

        # Let's use S1 as the probe set (it's size n).
        labels1 = mbk1.labels_ # Labels of S1 on Model 1
        labels2 = mbk2.predict(emb_s1) # Labels of S1 on Model 2

        score = adjusted_rand_score(labels1, labels2)

    return score

def select_diverse_samples(embeddings, global_indices, m_target, config):
    """
    Selects m samples using MiniBatchKMeans + Uniform Sampling.
    """
    n = len(embeddings)
    if n <= m_target:
        return global_indices, n, 1 # Keep all

    k = compute_k(n, config)
    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3,
                                random_state=config.SEED).fit(embeddings)
    labels = clusterer.labels_

    selected_indices = []

    # Uniform Strategy
    df = pd.DataFrame({'idx': global_indices, 'label': labels})

    # Target per cluster
    quota = int(np.ceil(m_target / k))

    # First pass: take up to quota
    groups = df.groupby('label')
    pool_remain = []

    for _, group in groups:
        curr_n = len(group)
        if curr_n <= quota:
            selected_indices.extend(group['idx'].tolist())
        else:
            # Random sample (deterministic via seed)
            selected = group.sample(n=quota, random_state=config.SEED)
            selected_indices.extend(selected['idx'].tolist())

            # Add remainders to pool if we need to fill gap
            # (In uniform strategy, usually we just stop, but if m_target is strict
            # we might need to top up from large clusters)

    # If we undershot m_target (because some clusters were small),
    # we can do a second pass on the remaining samples from large clusters
    # Implementation: Simple truncation or padding not strictly required by prompt
    # unless exact m is critical. The prompt says "approx ceil(m/k)".
    # We will stick to the strict selection to avoid duplicates.

    return np.array(selected_indices), k, "uniform"

def setup_local_environment(config):
    """
    Copies data from Drive to Local Disk to prevent network I/O bottlenecks.
    """
    print(f"\n[Setup] Setting up local workspace at {config.LOCAL_WORK_DIR}...")

    # Clean previous runs
    if os.path.exists(config.LOCAL_WORK_DIR):
        shutil.rmtree(config.LOCAL_WORK_DIR)
    os.makedirs(config.LOCAL_WORK_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Copy Input File
    if not os.path.exists(config.DRIVE_SOURCE_PATH):
        raise FileNotFoundError(f"Source file not found on Drive: {config.DRIVE_SOURCE_PATH}")

    file_size_gb = os.path.getsize(config.DRIVE_SOURCE_PATH) / (1024**3)
    print(f"[Setup] Copying TRAIN.h5 ({file_size_gb:.2f} GB) from Drive to Local SSD...")
    print(f"        Source: {config.DRIVE_SOURCE_PATH}")
    print(f"        Dest:   {config.TRAIN_H5_PATH}")

    t0 = time.time()
    shutil.copyfile(config.DRIVE_SOURCE_PATH, config.TRAIN_H5_PATH)
    print(f"[Setup] Copy complete in {time.time() - t0:.1f} seconds.")

def export_artifacts_to_drive(config):
    """
    Copies all generated files from Local Disk back to Drive.
    """
    print(f"\n[Export] moving results to Drive: {config.DRIVE_OUTPUT_DIR}...")

    os.makedirs(config.DRIVE_OUTPUT_DIR, exist_ok=True)

    # Copy filtered H5
    local_h5 = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
    drive_h5 = os.path.join(config.DRIVE_OUTPUT_DIR, config.OUTPUT_FILENAME)

    if os.path.exists(local_h5):
        print(f"        Copying HDF5 ({os.path.getsize(local_h5)/(1024**3):.2f} GB)...")
        shutil.copyfile(local_h5, drive_h5)

    # Copy CSVs and JSONs
    for fname in os.listdir(config.OUTPUT_DIR):
        if fname.endswith('.csv') or fname.endswith('.json'):
            src = os.path.join(config.OUTPUT_DIR, fname)
            dst = os.path.join(config.DRIVE_OUTPUT_DIR, fname)
            print(f"        Copying artifact: {fname}")
            shutil.copyfile(src, dst)

    print("[Export] All files synced to Google Drive.")

def run_pipeline():
    # --- PHASE 2: PROCESSING (On Local SSD) ---
    extractor = EmbeddingExtractor(config)
    h5_indexer = H5MetadataIndex(config.TRAIN_H5_PATH)

    selection_manifest = []
    stats_log = []
    all_selected_indices = []

    patient_ids = sorted(h5_indexer.patient_map.keys())

    print(f"\n[Pipeline] Starting processing for {len(patient_ids)} patients...")

    for pid in tqdm(patient_ids, desc="Processing Patients"):
        t0 = time.time()
        all_patches = h5_indexer.patient_map[pid]
        N = len(all_patches)

        # --- A. Adaptive N_embed Loop ---
        n_curr = min(config.N_START, N)
        final_embeddings = None
        stability_history = []

        # Small patient optimization
        if N <= config.N_START * 1.5:
            n_curr = N
            final_embeddings = extractor.get_embeddings(config.TRAIN_H5_PATH, all_patches)
            stability_history.append((N, 1.0))
        else:
            step = 0
            while step < config.MAX_STEPS:
                if n_curr >= N:
                    n_curr = N
                    final_embeddings = extractor.get_embeddings(config.TRAIN_H5_PATH, all_patches)
                    break

                idx_s1 = np.random.choice(all_patches, n_curr, replace=False)
                idx_s2 = np.random.choice(all_patches, n_curr, replace=False)

                emb_s1 = extractor.get_embeddings(config.TRAIN_H5_PATH, idx_s1)
                emb_s2 = extractor.get_embeddings(config.TRAIN_H5_PATH, idx_s2)

                scores = []
                for _ in range(config.STABILITY_REPEATS):
                    s = calculate_stability_score(emb_s1, emb_s2, idx_s1, idx_s2, config)
                    scores.append(s)
                avg_score = np.median(scores)
                stability_history.append((n_curr, avg_score))

                if avg_score >= config.STABILITY_THRESHOLD:
                    final_embeddings = emb_s1
                    all_patches_pool = idx_s1
                    break

                new_n = int(n_curr * config.GROWTH_FACTOR)
                if new_n >= config.N_MAX or new_n >= N:
                    n_curr = min(N, config.N_MAX)
                    idx_final = np.random.choice(all_patches, n_curr, replace=False)
                    final_embeddings = extractor.get_embeddings(config.TRAIN_H5_PATH, idx_final)
                    all_patches_pool = idx_final
                    break
                n_curr = new_n
                step += 1

                if final_embeddings is None:
                     idx_final = np.random.choice(all_patches, n_curr, replace=False)
                     final_embeddings = extractor.get_embeddings(config.TRAIN_H5_PATH, idx_final)
                     all_patches_pool = idx_final

        if final_embeddings is None:
             n_curr = min(N, config.N_START)
             all_patches_pool = np.random.choice(all_patches, n_curr, replace=False)
             final_embeddings = extractor.get_embeddings(config.TRAIN_H5_PATH, all_patches_pool)

        # --- B. Clustering & Selection ---
        if 'all_patches_pool' not in locals():
            all_patches_pool = all_patches

        m_curr = min(config.M_MAX, N)
        selected_local_indices, k_used, method = select_diverse_samples(
            final_embeddings, all_patches_pool, m_curr, config
        )

        all_selected_indices.extend(selected_local_indices)

        stats_log.append({
            'patient_id': pid,
            'total_patches': N,
            'chosen_n_embed': len(final_embeddings),
            'k_clusters': k_used,
            'selected_m': len(selected_local_indices),
            'runtime_sec': round(time.time() - t0, 2),
            'stability_trace': str(stability_history)
        })

        for sel_idx in selected_local_indices:
            selection_manifest.append({
                'patient_id': pid,
                'global_index': sel_idx,
                'embedding_source_n_embed': len(final_embeddings),
                'k_clusters': k_used,
                'selection_method': method
            })

    # --- PHASE 3: WRITE ARTIFACTS (Locally) ---
    print(f"\n[Output] Writing Filtered HDF5 (Selected {len(all_selected_indices)} samples)...")

    all_selected_indices = sorted(list(set(all_selected_indices)))

    # Write HDF5 to local disk
    write_filtered_hdf5(config, all_selected_indices)

    # Write CSVs to local disk
    pd.DataFrame(selection_manifest).to_csv(
        os.path.join(config.OUTPUT_DIR, "train_filtered_selection.csv"), index=False)
    pd.DataFrame(stats_log).to_csv(
        os.path.join(config.OUTPUT_DIR, "patient_filter_stats.csv"), index=False)

    with open(os.path.join(config.OUTPUT_DIR, "filter_run_config.json"), 'w') as f:
        cfg_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
        json.dump(cfg_dict, f, indent=4, default=str)

    # --- PHASE 4: EXPORT TO DRIVE ---
    export_artifacts_to_drive(config)
    print("[Pipeline] Complete.")

def guardrail(f_src):
  required = ['images', 'masks', 'patient_ids', 'labels']
  missing = [k for k in required if k not in f_src]
  if missing:
      raise KeyError(f"Source HDF5 missing required keys: {missing}. Available: {list(f_src.keys())}")

def write_filtered_hdf5(config, selected_indices):
    """
    Creates the new HDF5 file and copies data in chunks.
    """
    src_path = config.TRAIN_H5_PATH
    dst_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)

    with h5py.File(src_path, 'r') as f_src, h5py.File(dst_path, 'w') as f_dst:
        total = len(selected_indices)
        guardrail(f_src)
        # Initialize Datasets based on Source
        # (Assuming Layout A per spec)
        for key in ['images', 'masks', 'patient_ids', 'labels', 'filename']:
            if key in f_src:
                shape = list(f_src[key].shape)
                shape[0] = total # Update N dimension

                # Compression for images/masks recommended
                kw = {}
                if key in ['images', 'masks']:
                    kw = {'compression': 'gzip', 'chunks': True}

                ds = f_dst.create_dataset(key, shape=tuple(shape), dtype=f_src[key].dtype, **kw)

        # Copy Data in Batches
        batch_size = 1000
        for i in tqdm(range(0, total, batch_size), desc="Writing HDF5"):
            # Get slice of global indices
            batch_indices = selected_indices[i : i + batch_size]

            # Read sorted list from source (h5py supports list indexing if sorted)
            # IMPORTANT: h5py selection must be strictly increasing.
            # batch_indices are sorted by definition in run_pipeline

            for key in ['images', 'masks', 'patient_ids', 'labels', 'filename']:
                if key in f_src:
                    data = f_src[key][batch_indices]
                    f_dst[key][i : i + len(batch_indices)] = data

# @title 6. Main Pipeline Execution
# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)

seed_everything(Config.SEED)

torch.cuda.empty_cache()
gc.collect()

config = Config()

# --- PHASE 1: PREPARE LOCAL DISK ---
setup_local_environment(config)

try:
    run_pipeline()
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
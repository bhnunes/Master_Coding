import logging
import os
import sys
import time
import traceback
from typing import Any, cast

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler

from helpers.logging_utils import LoggerSettings, LoggerWriter, configure_root_logger
from helpers.training.checkpointing import (
    EarlyStopping,
    OHEMCheckpointSettings,
    ResumeCheckpointRequest,
    TrainingProvenanceRequest,
    build_training_compatibility_signature,
    get_previous_metrics,
    load_checkpoint_for_resume,
    save_metadata,
)
from helpers.training.config import load_training_ensemble_config
from helpers.training.data import (
    collate_batch,
    prepare_training_data,
    verify_patient_separation,
)
from helpers.training.gpu import GPUDownscale, GPUNormalizer
from helpers.training.loop import train_epoch, validate_epoch
from helpers.training.losses import BCEDiceHybridLossConfig, BCEDiceHybridLossPaper
from helpers.training.metrics import TrainingHealthTracker
from helpers.training.models import (
    create_model,
    create_optimizer,
    get_learning_rate,
    get_loss_weights,
)
from helpers.training.pipeline import (
    EpochRunConfig,
    FinalizeArtifactsConfig,
    RunHParams,
    build_run_hparams,
    finalize_training_artifacts,
    run_training_epochs,
)
from helpers.training.reporting import (
    close_aim_run,
    create_aim_run,
    create_email_body,
    ensure_aim_repo,
    send_email,
    track_epoch_metrics,
)
from helpers.training.runtime import seed_everything, worker_init_fn
from helpers.training.utils import clear_gpu, get_formatted_datetime_string

load_dotenv(override=True)

# =============================================================================
# 1) Environment & Hardware
# =============================================================================
training_config = load_training_ensemble_config()
training_logger = configure_root_logger(
    training_config.log_path,
    settings=LoggerSettings(
        logger_level=logging.INFO,
        file_level=logging.INFO,
        console_level=logging.INFO,
        file_mode="a",
        file_pattern="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        console_pattern="%(message)s",
    ),
)
sys.stdout = LoggerWriter(training_logger, logging.INFO)
sys.stderr = LoggerWriter(training_logger, logging.ERROR)
print("Libraries imported.")
print("Configuring environment...")

# Device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# 2) Training Loop Control
# =============================================================================
num_epochs = training_config.num_epochs
patience = training_config.patience
unleashed = training_config.unleashed
seed = training_config.seed

# =============================================================================
# 3) Data Loading & Performance
# =============================================================================
batch_size = training_config.batch_size
workers = training_config.workers
accumulation_steps = training_config.accumulation_steps

# Val loader uses a larger batch size
val_batch_size = training_config.val_batch_size

optimizer_name = training_config.optimizer_name

# =============================================================================
# 5) Loss Function (from Khened et al., Scientific Reports 2021)
# =============================================================================
# =============================================================================
# 7) Dataset & Normalization
# =============================================================================
master_manifest_path = training_config.master_manifest_path
metadata_dir = str(training_config.metadata_dir)
identifier = training_config.identifier

# =============================================================================
# 9) Checkpoints, Tracking & Notifications
# =============================================================================
checkpoint_path = str(training_config.checkpoint_path)
aim_repo_path = str(training_config.aim_repo_path)

# Email notifications
email_sender = training_config.email_sender or ""
email_recipients = list(training_config.email_recipients)
email_password = training_config.email_password or ""

# =============================================================================
# 0) Execution Mode & Reproducibility Profiles
# =============================================================================
# Options: "FAST_DEV" (Speed prioritized) | "PAPER" (Strict Determinism)
execution_mode = training_config.execution_mode
smart_sampling = training_config.smart_sampling
use_artifact_aware_loss = training_config.use_artifact_aware_loss
run_ohem = training_config.run_ohem
ohem_start_epoch = training_config.ohem_start_epoch
ohem_ratio = training_config.ohem_ratio
ohem_min_kept = training_config.ohem_min_kept
effective_master_manifest_path = master_manifest_path if use_artifact_aware_loss else None

subset_ratio = 1.0
use_subset = False

if execution_mode == "FAST_DEV":
    print("⚠️ RUNNING IN FAST_DEV MODE: Benchmarking ON, Determinism OFF")
    # Speed optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    # Defaults for dev
    subset_ratio = 1.0
    use_subset = False

elif execution_mode == "PAPER":
    print("🛡️ RUNNING IN PAPER MODE: Strict Determinism Enforced")
    # Reproducibility settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Defaults for paper
    subset_ratio = 1.0
    use_subset = False  # Force full dataset unless explicitly overridden

amp_precision = training_config.amp_precision
amp_log: dict[str, str] = {}


seed_everything(seed)


print(f"Reproducibility configured. seed={seed}")
print("Setting up Aim repository...")
ensure_aim_repo(aim_repo_path)
print("Aim/Ngrok setup complete.")


# ==============================================================================
# --- 13. Main Training Loop ---
# ==============================================================================

# 1. Resolve manifest-backed training and validation inputs
prepared_training_data = prepare_training_data(
    master_manifest_path=master_manifest_path,
    local_data_dir=training_config.local_data_dir,
    smart_sampling=smart_sampling,
    use_subset=use_subset,
    subset_ratio=subset_ratio,
    seed=seed,
    use_artifact_aware_loss=use_artifact_aware_loss,
)
train_dataset_provenance = prepared_training_data.training_provenance
validation_dataset_provenance = prepared_training_data.validation_provenance

print("\nCreating DataLoaders...")
train_ds: Any = None
val_ds: Any = None
train_loader: Any = None
val_loader: Any = None
try:
    # A) Full HDF5 Wrappers
    # Note: Ensure your preprocessing script included "patient_ids" in the HDF5
    # for the leakage check below to function.
    full_train_ds_h5 = prepared_training_data.train_dataset
    full_val_ds_h5 = prepared_training_data.validation_dataset

    # B) Patient Leakage Check (Critical for scientific validity)
    print("Verifying data integrity...")
    verify_patient_separation(full_train_ds_h5, full_val_ds_h5)

    train_ds = full_train_ds_h5
    val_ds = full_val_ds_h5

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
    samples_weights = prepared_training_data.sample_weights.numpy()

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(samples_weights).double().tolist(),
        num_samples=len(samples_weights),
        replacement=True,
        generator=sampler_generator,
    )

    print("Sampler Ready.")

    # 4) Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
    )

    print("DataLoaders created successfully (Train: canonical rows, Val: canonical rows).")

except Exception as e:
    print(f"DataLoader Err: {e}")
    import traceback

    traceback.print_exc()
    clear_gpu()
    raise

selected_architecture = training_config.architecture
selected_encoder = training_config.encoder
selected_resume_checkpoint = str(training_config.resume_checkpoint or "")
selected_run = (selected_architecture, selected_encoder, selected_resume_checkpoint)

# Instantiate globally
gpu_normalizer = GPUNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)

gpu_downscale = GPUDownscale(p=0.07).to(device)

for architecture, encoder, resume_checkpoint_path in [selected_run]:
    health = TrainingHealthTracker(name=f"{architecture}_{encoder}")
    base_learning_rate, weight_decay = get_learning_rate(architecture)
    loss_weights = get_loss_weights(architecture)
    alpha_bce = loss_weights.alpha_bce
    beta_dice_bg = loss_weights.beta_dice_bg
    gamma_dice_fg = loss_weights.gamma_dice_fg

    # --- Init Aim Run ---
    experiment_name = f"{identifier}_{architecture}_{encoder}_{get_formatted_datetime_string()}"
    run = create_aim_run(
        experiment_name=experiment_name,
        repo_path=aim_repo_path,
        hparams=build_run_hparams(
            RunHParams(
                experiment_name=experiment_name,
                architecture=architecture,
                encoder=encoder,
                optimizer_name=optimizer_name,
                base_learning_rate=base_learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                num_epochs=num_epochs,
                workers=workers,
                seed=seed,
                patience=patience,
                train_len=len(train_ds),
                val_len=len(val_ds),
                alpha_bce=alpha_bce,
                beta_dice_bg=beta_dice_bg,
                gamma_dice_fg=gamma_dice_fg,
                run_ohem=run_ohem,
                ohem_start_epoch=ohem_start_epoch,
                ohem_ratio=ohem_ratio,
                ohem_min_kept=ohem_min_kept,
            )
        ),
    )

    print(f"Architecture: {architecture}")
    # --- Initialize Model, Optimizer, Early Stopping ---
    print("Initializing SMP Model, Optimizer, ES...")

    # --- Instantiate SMP Model ---
    try:
        model = create_model(architecture=architecture, encoder=encoder, validation=False)
        model.to(device)
    except Exception as e:
        print(f"Model init err: {e}")
        raise ValueError(f"Unknown architecture: {architecture}") from e

    # --- Compile Model ---
    try:
        model = cast(torch.nn.Module, torch.compile(model))
        print("Model compiled.")
    except Exception as e:
        print(f"Compile failed: {e}.")

    print(f"Model->{device}")

    optimizer = create_optimizer(model, optimizer_name, base_learning_rate, weight_decay)

    print(f"Optimizer initialized with {optimizer_name}")

    output_best_model_path_for_this_run = os.path.join(
        checkpoint_path, f"BEST_MODEL_{experiment_name}.pth"
    )

    # Initialize EarlyStopping with this fixed path
    early_stopping = EarlyStopping(
        patience=int(patience),
        verbose=True,
        delta=0.0001,
        output_best_model_path=output_best_model_path_for_this_run,
    )

    full_resume_checkpoint_path = (
        os.path.join(checkpoint_path, resume_checkpoint_path) if resume_checkpoint_path else None
    )
    expected_compatibility_signature = build_training_compatibility_signature(
        TrainingProvenanceRequest(
            dataset=train_dataset_provenance,
            validation_dataset=validation_dataset_provenance,
            master_manifest_path=effective_master_manifest_path,
            resume_checkpoint=None,
            ohem=OHEMCheckpointSettings(
                run_ohem=run_ohem,
                ohem_start_epoch=ohem_start_epoch,
                ohem_ratio=ohem_ratio,
                ohem_min_kept=ohem_min_kept,
            ),
        )
    )

    start_epoch = load_checkpoint_for_resume(
        ResumeCheckpointRequest(
            model=model,
            optimizer=optimizer,
            early_stopping=early_stopping,
            checkpoint_path=full_resume_checkpoint_path,
            device=device,
            expected_compatibility_signature=expected_compatibility_signature,
        )
    )

    if early_stopping._current_best_checkpoint_on_disk_path:
        # If we successfully loaded a resume checkpoint, that's our initial best.
        metadata_best_path = early_stopping._current_best_checkpoint_on_disk_path
    else:
        # If training from scratch, this will be the first best model saved.
        metadata_best_path = output_best_model_path_for_this_run

    print(f"Starting Training For {architecture} from epoch {start_epoch + 1}...")

    loss_fn = BCEDiceHybridLossPaper(
        BCEDiceHybridLossConfig(
            alpha=alpha_bce,
            beta=beta_dice_bg,
            gamma=gamma_dice_fg,
            run_ohem=run_ohem,
            ohem_start_epoch=ohem_start_epoch,
            ohem_ratio=ohem_ratio,
            ohem_min_kept=ohem_min_kept,
        )
    )
    print(
        f"Using BCE+Dice Hybrid Loss "
        f"(alpha={alpha_bce}, beta={beta_dice_bg}, gamma={gamma_dice_fg})"
    )
    if run_ohem:
        print(
            "OHEM enabled "
            f"(start_epoch={ohem_start_epoch}, ratio={ohem_ratio}, min_kept={ohem_min_kept})"
        )

    epoch_state = run_training_epochs(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=EpochRunConfig(
            health=health,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            architecture=architecture,
            unleashed=unleashed,
            train_epoch_fn=train_epoch,
            validate_epoch_fn=validate_epoch,
            early_stopping=early_stopping,
            track_epoch_metrics_fn=track_epoch_metrics,
            loss_fn=loss_fn,
            device=device,
            accumulation_steps=accumulation_steps,
            amp_precision=amp_precision,
            gpu_normalizer=gpu_normalizer,
            gpu_downscale=gpu_downscale,
            use_artifact_aware_loss=use_artifact_aware_loss,
            run=run,
        ),
    )
    training_successful = epoch_state.training_successful
    metadata_best_path = epoch_state.metadata_best_path
    amp_log = epoch_state.amp_log

    # --- Post-Training for Fold ---
    if not training_successful:
        print(f"Train loop stopped early for {architecture} (Emergency Stop).")
        print("Skipping threshold tuning for this architecture due to instability.")
        # Clear GPU and move to the next architecture in the list
        del model, optimizer, loss_fn
        clear_gpu()
        continue  # <--- SKIPS REST OF LOOP, GOES TO NEXT selected_architecture

    print(f"\nTrain loop finished successfully for {architecture}.")
    clear_gpu()

    print("Loading best model for threshold tuning...")
    final_best_checkpoint_path = early_stopping._current_best_checkpoint_on_disk_path

    try:
        finalize_training_artifacts(
            FinalizeArtifactsConfig(
                best=epoch_state.best,
                metadata_best_path=metadata_best_path,
                fallback_checkpoint_path=(
                    str(final_best_checkpoint_path)
                    if final_best_checkpoint_path is not None
                    else full_resume_checkpoint_path
                ),
                device=device,
                load_checkpoint_fn=torch.load,
                get_previous_metrics_fn=get_previous_metrics,
                save_metadata_fn=save_metadata,
                save_metadata_kwargs={
                    "encoder": encoder,
                    "architecture": architecture,
                    "metadata_dir": metadata_dir,
                    "amp_log": amp_log,
                    "base_learning_rate": base_learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "workers": workers,
                    "seed": seed,
                    "dataset": train_dataset_provenance,
                    "validation_dataset": validation_dataset_provenance,
                    "patience": patience,
                    "optimizer_name": optimizer_name,
                    "alpha_bce": alpha_bce,
                    "beta_dice_bg": beta_dice_bg,
                    "gamma_dice_fg": gamma_dice_fg,
                    "execution_mode": execution_mode,
                    "master_manifest_path": effective_master_manifest_path,
                    "use_artifact_aware_loss": use_artifact_aware_loss,
                    "run_ohem": run_ohem,
                    "ohem_start_epoch": ohem_start_epoch,
                    "ohem_ratio": ohem_ratio,
                    "ohem_min_kept": ohem_min_kept,
                    "resume_checkpoint": full_resume_checkpoint_path,
                },
                create_email_body_fn=create_email_body,
                send_email_fn=send_email,
                email_sender=email_sender,
                email_recipients=email_recipients,
                email_password=email_password,
                experiment_name=experiment_name,
            )
        )

    except Exception as e:
        print(f"Test/Visu Err: {e}")
        traceback.print_exc()
        raise

    close_aim_run(run)

    print(f"====== METRIC STABILITY FOR {architecture} ======")
    health.log_run()

    clear_gpu()
    print(f"\n{'=' * 20} Finished Fold {architecture} {'=' * 20}")
    time.sleep(3)


# --- Final Cleanup ---
print("\nAll folds processed.")

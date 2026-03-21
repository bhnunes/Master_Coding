from pathlib import Path

from helpers.ensemble_optimizer.metadata import SelectedModelMetadata
from helpers.ensemble_optimizer.reporting import build_recipe_metadata, write_recipe_metadata


def test_write_recipe_metadata_preserves_inference_contract(tmp_path: Path) -> None:
    selected_models = [
        SelectedModelMetadata(
            architecture="SWIN",
            encoder="enc-a",
            checkpoint_path="/tmp/a.ckpt",
            metadata_filename="a_meta.json",
            sort_metric_value=0.9,
            raw_metadata={"best_model_epoch": 3, "hyperparameters": {}},
        ),
        SelectedModelMetadata(
            architecture="FPN",
            encoder="enc-b",
            checkpoint_path="/tmp/b.ckpt",
            metadata_filename="b_meta.json",
            sort_metric_value=0.8,
            raw_metadata={"best_model_epoch": 5, "hyperparameters": {}},
        ),
    ]

    payload = build_recipe_metadata(
        selected_models=selected_models,
        semantic_indices=[0],
        spatial_indices=[1],
        semantic_weights=[1.0],
        spatial_weights=[1.0],
        roi_context_scale=4,
        roi_threshold=0.33,
        decision_threshold=0.57,
        spill_penalty_lambda=0.1,
        spatial_patient_policy="positive_only",
        calibration_metrics={"Calibration_best_mcc": 0.61},
        holdout_metrics={"Macro_AUPRC_in_ROI": 0.7},
        generated_at="2026-03-20_10_00_00",
        compatibility_signature="compat-a",
        validation_provenance={"dataset_sha256": "val-sha"},
        split_fingerprint="split-sha",
    )

    output_path = write_recipe_metadata(payload, tmp_path, "2026-03-20_10_00_00")

    assert output_path == tmp_path / "ENSEMBLE_TWO_STREAM_2026-03-20_10_00_00.json"
    assert payload["ensemble_strategy"] == "two_stream_spatial_gating"
    assert payload["roi_config"]["threshold"] == 0.33
    assert payload["decision_config"]["threshold"] == 0.57
    assert payload["model_registry"][0]["stream_role"] == "semantic"
    assert payload["model_registry"][1]["stream_role"] == "spatial"
    assert payload["calibration_metrics"] == {"Calibration_best_mcc": 0.61}
    assert "holdout_metrics" in payload
    assert payload["compatibility_signature"] == "compat-a"
    assert payload["provenance"]["validation"] == {"dataset_sha256": "val-sha"}
    assert payload["provenance"]["split_fingerprint"] == "split-sha"

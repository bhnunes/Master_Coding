from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from helpers.graph.contamination import GraphContaminationParameters


@dataclass(frozen=True)
class GraphCleaningParameterArtifact:
    """Portable Stage 4 parameter artifact shared by tuning and cleaning."""

    graph_params: GraphContaminationParameters
    tau: float
    best_cross_validated_f1: float
    total_labeled_pairs: int
    training_pairs: int
    test_pairs: int
    random_state: int
    generated_by: str


def save_graph_cleaning_parameter_artifact(
    output_path: Path,
    artifact: GraphCleaningParameterArtifact,
) -> None:
    """Save tuned graph parameters and tau to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")


def load_graph_cleaning_parameter_artifact(output_path: Path) -> GraphCleaningParameterArtifact:
    """Load tuned graph parameters and tau from JSON."""

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Graph cleaning parameter artifact not found: {output_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Graph cleaning parameter artifact is invalid JSON: {output_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("Graph cleaning parameter artifact must be a JSON object.")

    graph_params_payload = payload.get("graph_params")
    if not isinstance(graph_params_payload, dict):
        raise ValueError("Graph cleaning parameter artifact must contain 'graph_params'.")

    try:
        return GraphCleaningParameterArtifact(
            graph_params=GraphContaminationParameters(
                bg_intensity_thresh=int(graph_params_payload["bg_intensity_thresh"]),
                k=float(graph_params_payload["k"]),
                min_size=int(graph_params_payload["min_size"]),
                erosion_px=int(graph_params_payload["erosion_px"]),
            ),
            tau=float(payload["tau"]),
            best_cross_validated_f1=float(payload["best_cross_validated_f1"]),
            total_labeled_pairs=int(payload["total_labeled_pairs"]),
            training_pairs=int(payload["training_pairs"]),
            test_pairs=int(payload["test_pairs"]),
            random_state=int(payload["random_state"]),
            generated_by=str(payload["generated_by"]),
        )
    except KeyError as error:
        raise ValueError(
            f"Graph cleaning parameter artifact is missing required field: {error.args[0]}"
        ) from error

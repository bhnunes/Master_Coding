from __future__ import annotations

import numpy as np

from helpers.graph_contamination import (
    GraphContaminationParameters,
    coerce_graph_contamination_parameters,
)


def test_coerce_graph_contamination_parameters_casts_numpy_values() -> None:
    params = coerce_graph_contamination_parameters(
        {
            "bg_intensity_thresh": np.int64(198),
            "k": np.float64(386),
            "min_size": np.int64(200),
            "erosion_px": np.int64(0),
        }
    )

    assert params == GraphContaminationParameters(
        bg_intensity_thresh=198,
        k=386.0,
        min_size=200,
        erosion_px=0,
    )

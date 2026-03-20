from __future__ import annotations

import ast
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py
import numpy as np
from tqdm.auto import tqdm


def _load_smart_sampler_functions() -> dict[str, object]:
    source_path = Path("/workspace/8_smart_sampler.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected_nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"guardrail", "resolve_filename_key", "write_filtered_hdf5"}
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: dict[str, object] = {"h5py": h5py, "os": os, "tqdm": tqdm}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)
    return namespace


def test_write_filtered_hdf5_preserves_plural_filenames_dataset(tmp_path: Path) -> None:
    functions = _load_smart_sampler_functions()
    write_filtered_hdf5 = cast(
        Callable[[Any, np.ndarray[Any, Any]], None], functions["write_filtered_hdf5"]
    )

    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((2, 4, 4), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 2], dtype=np.int32))
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset(
            "filenames",
            data=np.array([b"PATIENT_1_a.png", b"PATIENT_2_b.png"], dtype="S32"),
        )

    config = SimpleNamespace(
        TRAIN_H5_PATH=str(source_path),
        OUTPUT_DIR=str(output_dir),
        OUTPUT_FILENAME="TRAIN_FILTERED.h5",
    )

    write_filtered_hdf5(config, np.array([1], dtype=np.int64))

    with h5py.File(output_dir / "TRAIN_FILTERED.h5", "r") as handle:
        assert "filenames" in handle
        assert "filename" not in handle
        assert handle["filenames"][0].decode("utf-8") == "PATIENT_2_b.png"

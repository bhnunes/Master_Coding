from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from helpers.extraction import patch_engine


def test_get_png_save_kwargs_uses_fast_defaults() -> None:
    image_kwargs = patch_engine.get_png_save_kwargs(kind="image")
    mask_kwargs = patch_engine.get_png_save_kwargs(kind="mask")

    assert image_kwargs == {"format": "PNG", "compress_level": 1, "optimize": False}
    assert mask_kwargs == {"format": "PNG", "compress_level": 1, "optimize": False}


def test_save_patch_outputs_passes_fast_png_kwargs(tmp_path: Path, monkeypatch: Any) -> None:
    patch_image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    final_mask = np.zeros((4, 4), dtype=np.uint8)
    captured: list[tuple[str, dict[str, Any]]] = []

    def fake_save(_self: Image.Image, fp: str, **kwargs: Any) -> None:
        captured.append((fp, kwargs))

    monkeypatch.setattr(Image.Image, "save", fake_save)

    patch_engine.save_patch_outputs(
        patch_pil=patch_image,
        final_mask=final_mask,
        image_output_path=tmp_path / "image.png",
        mask_output_path=tmp_path / "mask.png",
    )

    assert captured == [
        (
            str(tmp_path / "image.png"),
            {"format": "PNG", "compress_level": 1, "optimize": False},
        ),
        (
            str(tmp_path / "mask.png"),
            {"format": "PNG", "compress_level": 1, "optimize": False},
        ),
    ]

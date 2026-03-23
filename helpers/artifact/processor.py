from __future__ import annotations

import timeit
from pathlib import Path
from typing import Any, cast

from PIL import Image

from helpers.artifact.config import ArtifactDetectionConfig
from helpers.artifact.model_loader import ArtifactModelLoader
from helpers.runtime_platform import load_openslide_module

Image.MAX_IMAGE_PIXELS = 1_000_000_000


class ArtifactProcessor:
    """Process a single slide into a GeoJSON artifact map."""

    def __init__(self, config: ArtifactDetectionConfig, model_loader: ArtifactModelLoader) -> None:
        self.config = config
        self.model_loader = model_loader

    def process_slide(
        self, slide_path: Path, zip_member_name: str, geojson_output_path: Path
    ) -> None:
        """Run tissue detection and QC processing for one slide."""

        output_dir = slide_path.parent / "artifact_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        tissue_outputs = self._run_tissue_detection(slide_path, output_dir)
        self._run_qc_processing(slide_path, geojson_output_path, tissue_outputs)

    def _run_tissue_detection(self, slide_path: Path, output_dir: Path) -> dict[str, Path]:
        import cv2
        import numpy as np
        import torch

        from helpers.wsi.tis_detect_helper_fx import get_preprocessing, make_class_map

        models = self.model_loader.load()
        openslide_module = load_openslide_module()
        tissue_mask_dir = output_dir / "tis_det_mask"
        tissue_overlay_dir = output_dir / "tis_det_overlay"
        tissue_thumb_dir = output_dir / "tis_det_thumbnail"
        tissue_mask_col_dir = output_dir / "tis_det_mask_col"
        for directory in [
            tissue_mask_dir,
            tissue_overlay_dir,
            tissue_thumb_dir,
            tissue_mask_col_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        slide = openslide_module.OpenSlide(str(slide_path))
        width_l0, height_l0 = slide.level_dimensions[0]
        mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
        reduction_factor = self.config.mpp_model_td / mpp
        thumb_width = max(1, int(round(width_l0 / reduction_factor)))
        thumb_height = max(1, int(round(height_l0 / reduction_factor)))
        image_original = slide.get_thumbnail((thumb_width, thumb_height))

        slide_stem = slide_path.stem
        thumbnail_path = tissue_thumb_dir / f"{slide_stem}.jpg"
        tissue_mask_path = tissue_mask_dir / f"{slide_stem}_MASK.png"
        tissue_mask_colored_path = tissue_mask_col_dir / f"{slide_stem}_MASK_COL.png"
        tissue_overlay_path = tissue_overlay_dir / f"{slide_stem}_OVERLAY.jpg"
        image_original.save(thumbnail_path, quality=80)

        thumbnail_image = image_original.convert("RGB")

        width, height = thumbnail_image.size
        patch_size = self.config.model_patch_size
        tiles_x = width // patch_size
        tiles_y = height // patch_size
        overhang_x = width - tiles_x * patch_size
        overhang_y = height - tiles_y * patch_size
        tile_cols = tiles_x + (1 if overhang_x > 0 else 0)
        tile_rows = tiles_y + (1 if overhang_y > 0 else 0)
        colors = [[50, 50, 250], [128, 128, 128]]

        row_masks: list[Any] = []
        row_class_masks: list[Any] = []
        with torch.inference_mode():
            for tile_y in range(tile_rows):
                current_row_masks: list[Any] = []
                current_row_class_masks: list[Any] = []
                for tile_x in range(tile_cols):
                    image_work = _crop_tile(
                        thumbnail_image,
                        tile_x,
                        tile_y,
                        tiles_x,
                        tiles_y,
                        patch_size,
                        width,
                        height,
                    )
                    image_pre = get_preprocessing(  # type: ignore[no-untyped-call]
                        image_work, models.preprocessing_fn
                    )
                    x_tensor = torch.from_numpy(image_pre).to(self.config.device).unsqueeze(0)
                    predictions = cast(Any, models.tissue_model).predict(x_tensor)
                    predictions = predictions.squeeze().cpu().numpy()
                    mask = np.argmax(predictions, axis=0).astype("int8")
                    class_mask = make_class_map(mask, colors)  # type: ignore[no-untyped-call]
                    current_row_masks.append(mask)
                    current_row_class_masks.append(class_mask)
                stitched_row_mask, stitched_row_class_mask = _combine_horizontal_tiles(
                    current_row_masks,
                    current_row_class_masks,
                    patch_size=patch_size,
                    overhang_x=overhang_x,
                )
                row_masks.append(stitched_row_mask)
                row_class_masks.append(stitched_row_class_mask)

        end_image = None
        end_image_class_map = None
        if row_masks and row_class_masks:
            end_image, end_image_class_map = _combine_vertical_tiles(
                row_masks,
                row_class_masks,
                patch_size=patch_size,
                overhang_y=overhang_y,
            )

        if end_image is None or end_image_class_map is None:
            raise RuntimeError(f"No tissue detection output was created for '{slide_path.name}'.")

        Image.fromarray(end_image).save(tissue_mask_path)
        Image.fromarray(end_image_class_map).save(tissue_mask_colored_path)
        overlay = cv2.addWeighted(
            np.array(thumbnail_image),
            self.config.over_image,
            end_image_class_map,
            self.config.over_mask,
            0,
        )
        Image.fromarray(overlay).save(tissue_overlay_path)

        return {
            "tissue_mask_path": tissue_mask_path,
            "tissue_mask_colored_path": tissue_mask_colored_path,
            "tissue_overlay_path": tissue_overlay_path,
            "thumbnail_path": thumbnail_path,
        }

    def _run_qc_processing(
        self,
        slide_path: Path,
        geojson_output_path: Path,
        tissue_outputs: dict[str, Path],
    ) -> None:
        import cv2
        import numpy as np

        from helpers.wsi.colors import colors_QC7
        from helpers.wsi.maps import make_overlay
        from helpers.wsi.process import mask_to_geojson, slide_process_single
        from helpers.wsi.slide_info import slide_info

        start = timeit.default_timer()
        models = self.model_loader.load()
        openslide_module = load_openslide_module()
        slide = openslide_module.open_slide(str(slide_path))
        patch_size, patch_count_w, patch_count_h, mpp, width_l0, height_l0, _ = slide_info(
            slide,
            self.config.model_patch_size,
            self.config.mpp_model,
        )
        tissue_detection_map = Image.open(tissue_outputs["tissue_mask_path"])
        if tissue_detection_map.mode != "RGB":
            tissue_detection_map = tissue_detection_map.convert("RGB")  # type: ignore[assignment]
        target_width = max(1, int(width_l0 * mpp / self.config.mpp_model))
        target_height = max(1, int(height_l0 * mpp / self.config.mpp_model))
        tissue_detection_map_mpp = np.array(
            tissue_detection_map.resize((target_width, target_height), Image.Resampling.LANCZOS)
        )
        colors = colors_QC7
        map_image, full_mask = slide_process_single(  # type: ignore[no-untyped-call]
            models.qc_model,
            tissue_detection_map_mpp,
            slide,
            patch_count_w,
            patch_count_h,
            patch_size,
            self.config.model_patch_size,
            colors,
            self.config.encoder_model,
            self.config.encoder_weights,
            self.config.device,
            self.config.back_class,
            self.config.mpp_model,
            mpp,
            width_l0,
            height_l0,
        )

        maps_dir = slide_path.parent / "artifact_output" / "maps_qc"
        masks_dir = slide_path.parent / "artifact_output" / "mask_qc"
        overlays_dir = slide_path.parent / "artifact_output" / "overlays_qc"
        for directory in [maps_dir, masks_dir, overlays_dir, geojson_output_path.parent]:
            directory.mkdir(parents=True, exist_ok=True)

        map_path = maps_dir / f"{slide_path.stem}_map_QC.png"
        mask_path = masks_dir / f"{slide_path.stem}_mask.png"
        overlay_path = overlays_dir / f"{slide_path.stem}_overlay_QC.jpg"

        map_image.save(map_path)
        cv2.imwrite(str(mask_path), full_mask)
        factor = self.config.mpp_model / mpp
        mask_to_geojson(str(mask_path), str(geojson_output_path), factor)  # type: ignore[no-untyped-call]
        overlay = make_overlay(  # type: ignore[no-untyped-call]
            slide,
            map_image,
            patch_size,
            patch_count_w,
            patch_count_h,
            self.config.overlay_factor,
        )
        Image.fromarray(overlay).save(overlay_path)
        _ = start


def _crop_tile(
    image: Image.Image,
    tile_x: int,
    tile_y: int,
    tiles_x: int,
    tiles_y: int,
    patch_size: int,
    width: int,
    height: int,
) -> Image.Image:
    if tile_x != tiles_x and tile_y != tiles_y:
        return image.crop(
            (
                tile_x * patch_size,
                tile_y * patch_size,
                (tile_x + 1) * patch_size,
                (tile_y + 1) * patch_size,
            )
        )
    if tile_x == tiles_x and tile_y != tiles_y:
        return image.crop(
            (width - patch_size, tile_y * patch_size, width, (tile_y + 1) * patch_size)
        )
    if tile_x != tiles_x and tile_y == tiles_y:
        return image.crop(
            (tile_x * patch_size, height - patch_size, (tile_x + 1) * patch_size, height)
        )
    return image.crop((width - patch_size, height - patch_size, width, height))


def _combine_horizontal_tiles(
    row_masks: list[Any],
    row_class_masks: list[Any],
    *,
    patch_size: int,
    overhang_x: int,
) -> tuple[Any, Any]:
    import numpy as np

    if not row_masks or not row_class_masks or len(row_masks) != len(row_class_masks):
        raise RuntimeError("Horizontal tile rows must contain matching mask and class-mask tiles.")

    tiles = list(row_masks)
    class_tiles = list(row_class_masks)
    if len(tiles) > 1 and overhang_x > 0:
        tiles[-1] = tiles[-1][:, patch_size - overhang_x : patch_size]
        class_tiles[-1] = class_tiles[-1][:, patch_size - overhang_x : patch_size, :]
    return np.concatenate(tiles, axis=1), np.concatenate(class_tiles, axis=1)


def _combine_vertical_tiles(
    image_rows: list[Any],
    class_rows: list[Any],
    *,
    patch_size: int,
    overhang_y: int,
) -> tuple[Any, Any]:
    import numpy as np

    if not image_rows or not class_rows or len(image_rows) != len(class_rows):
        raise RuntimeError("Vertical tile rows must contain matching image and class-map rows.")

    tiles = list(image_rows)
    class_tiles = list(class_rows)
    if len(tiles) > 1 and overhang_y > 0:
        tiles[-1] = tiles[-1][patch_size - overhang_y : patch_size, :]
        class_tiles[-1] = class_tiles[-1][patch_size - overhang_y : patch_size, :, :]
    return np.concatenate(tiles, axis=0), np.concatenate(class_tiles, axis=0)


def _append_horizontal_tile(
    temp_image: Any,
    temp_image_class_map: Any,
    mask: Any,
    class_mask: Any,
    tile_x: int,
    tiles_x: int,
    patch_size: int,
    overhang_x: int,
) -> tuple[Any, Any]:
    import numpy as np

    if tile_x == 0:
        return mask, class_mask
    if temp_image is None or temp_image_class_map is None:
        raise RuntimeError("Horizontal stitching state was not initialized.")
    if tile_x == tiles_x:
        mask_clip = mask[:, patch_size - overhang_x : patch_size] if overhang_x > 0 else mask[:, :0]
        class_mask_clip = (
            class_mask[:, patch_size - overhang_x : patch_size, :]
            if overhang_x > 0
            else class_mask[:, :0, :]
        )
        return (
            np.concatenate((temp_image, mask_clip), axis=1),
            np.concatenate((temp_image_class_map, class_mask_clip), axis=1),
        )
    return (
        np.concatenate((temp_image, mask), axis=1),
        np.concatenate((temp_image_class_map, class_mask), axis=1),
    )


def _append_vertical_tile(
    end_image: Any,
    end_image_class_map: Any,
    temp_image: Any,
    temp_image_class_map: Any,
    tile_y: int,
    tiles_y: int,
    patch_size: int,
    overhang_y: int,
) -> tuple[Any, Any]:
    import numpy as np

    if tile_y == 0:
        return temp_image, temp_image_class_map
    if end_image is None or end_image_class_map is None:
        raise RuntimeError("Vertical stitching state was not initialized.")
    temp_image_array = cast(Any, temp_image)
    temp_image_class_map_array = cast(Any, temp_image_class_map)
    if tile_y == tiles_y:
        temp_clip = (
            temp_image_array[patch_size - overhang_y : patch_size, :]
            if overhang_y > 0
            else temp_image_array[:0, :]
        )
        temp_class_clip = (
            temp_image_class_map_array[patch_size - overhang_y : patch_size, :, :]
            if overhang_y > 0
            else temp_image_class_map_array[:0, :, :]
        )
        return (
            np.concatenate((end_image, temp_clip), axis=0),
            np.concatenate((end_image_class_map, temp_class_clip), axis=0),
        )
    return (
        np.concatenate((end_image, temp_image_array), axis=0),
        np.concatenate((end_image_class_map, temp_image_class_map_array), axis=0),
    )

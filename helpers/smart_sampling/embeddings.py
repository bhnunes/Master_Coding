from __future__ import annotations

import atexit
import os
from collections.abc import Sequence
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from helpers.smart_sampling.config import SmartSamplerConfig

IMAGE_TENSOR_NDIM = 3
CHANNEL_FIRST_IMAGE_MAX = 4


class H5PatchDataset(Dataset[Any]):
    def __init__(
        self,
        h5_path: str,
        indices: npt.NDArray[np.int64],
    ) -> None:
        self.h5_path = h5_path
        self.indices = indices
        self.h5_file: h5py.File | None = None
        self.images_dset: Any = None
        self._opened_pid: int | None = None
        self._atexit_registered = False

    def __len__(self) -> int:
        return len(self.indices)

    def _open_file(self) -> None:
        pid = os.getpid()
        if self.h5_file is not None and self._opened_pid != pid:
            self.close()

        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path,
                "r",
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024,
            )
            self.images_dset = cast(Any, self.h5_file["images"])
            self._opened_pid = pid
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def __getitem__(self, idx: int) -> npt.NDArray[np.uint8]:
        if self.h5_file is None:
            self._open_file()
        global_idx = int(self.indices[idx])
        image_data = np.asarray(cast(Any, self.images_dset)[global_idx])
        if image_data.ndim != IMAGE_TENSOR_NDIM:
            raise ValueError(f"Expected 3D image tensor for Stage 7, got shape {image_data.shape}.")
        if image_data.shape[0] <= CHANNEL_FIRST_IMAGE_MAX:
            image_data = np.transpose(image_data, (1, 2, 0))
        return np.asarray(image_data, dtype=np.uint8)

    def close(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
            self.images_dset = None
            self._opened_pid = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["images_dset"] = None
        state["_opened_pid"] = None
        state["_atexit_registered"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.h5_file = None
        self.images_dset = None
        self._opened_pid = None
        self._atexit_registered = False


def _collate_images(batch: Sequence[npt.NDArray[np.uint8]]) -> list[npt.NDArray[np.uint8]]:
    return [np.asarray(image, dtype=np.uint8) for image in batch]


class EmbeddingExtractor:
    def __init__(self, config: SmartSamplerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.processor = self._init_processor()
        self.model = self._init_model()

    def _init_processor(self) -> Any:
        try:
            processor_factory = cast(Any, AutoImageProcessor)
            return processor_factory.from_pretrained(self.config.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize Stage 7 image processor '{self.config.model_name}'."
            ) from error

    def _init_model(self) -> torch.nn.Module:
        try:
            model_factory = cast(Any, AutoModel)
            model = model_factory.from_pretrained(self.config.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize Stage 7 model '{self.config.model_name}'."
            ) from error
        model = cast(torch.nn.Module, model).to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def get_embeddings(
        self, h5_path: str, indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        if len(indices) == 0:
            return np.empty((0, 0), dtype=np.float32)

        dataset = H5PatchDataset(h5_path, indices)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=4 if self.config.num_workers > 0 else None,
            collate_fn=_collate_images,
        )

        embeddings_list: list[npt.NDArray[np.float32]] = []
        try:
            for batch in loader:
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {
                    name: tensor.to(self.device, non_blocking=self.device.type == "cuda")
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                    for name, tensor in inputs.items()
                }
                outputs = self.model(**inputs)
                cls_features = outputs.last_hidden_state[:, 0, :]
                embeddings_list.append(cls_features.cpu().numpy().astype(np.float32, copy=False))
        finally:
            dataset.close()

        return np.vstack(embeddings_list).astype(np.float32, copy=False)

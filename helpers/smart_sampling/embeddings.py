from __future__ import annotations

import atexit
import os
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from helpers.smart_sampling.config import SmartSamplerConfig


class H5PatchDataset(Dataset[Any]):
    def __init__(
        self,
        h5_path: str,
        indices: npt.NDArray[np.int64],
        transform: transforms.Compose,
    ) -> None:
        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.h5_file is None:
            self._open_file()
        global_idx = int(self.indices[idx])
        image_data = np.asarray(cast(Any, self.images_dset)[global_idx])
        if image_data.shape[0] <= 4:
            image_data = np.transpose(image_data, (1, 2, 0))
        transformed = self.transform(image_data)
        return torch.as_tensor(transformed)

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


def get_preprocessing_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class EmbeddingExtractor:
    def __init__(self, config: SmartSamplerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._init_model()
        self.transform = get_preprocessing_transforms(config.input_size)

    def _init_model(self) -> torch.nn.Module:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=3,
            classes=1,
        )
        encoder = model.encoder.to(self.device)
        encoder.eval()
        return cast(torch.nn.Module, encoder)

    @torch.no_grad()
    def get_embeddings(
        self, h5_path: str, indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        if len(indices) == 0:
            return np.empty((0, 0), dtype=np.float32)

        dataset = H5PatchDataset(h5_path, indices, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=4 if self.config.num_workers > 0 else None,
        )

        embeddings_list: list[npt.NDArray[np.float32]] = []
        try:
            for batch in loader:
                features = self.model(batch.to(self.device, non_blocking=True))
                last_map = features[-1]
                gap = torch.mean(last_map, dim=[2, 3])
                embeddings_list.append(gap.cpu().numpy().astype(np.float32))
        finally:
            dataset.close()

        return np.vstack(embeddings_list).astype(np.float32, copy=False)

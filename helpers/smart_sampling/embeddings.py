from __future__ import annotations

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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5_path, "r") as handle:
            global_idx = int(self.indices[idx])
            image_data = np.asarray(handle["images"][global_idx])
            if image_data.shape[0] <= 4:
                image_data = np.transpose(image_data, (1, 2, 0))
        transformed = self.transform(image_data)
        return torch.as_tensor(transformed)


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
        )

        embeddings_list: list[npt.NDArray[np.float32]] = []
        for batch in loader:
            features = self.model(batch.to(self.device))
            last_map = features[-1]
            gap = torch.mean(last_map, dim=[2, 3])
            embeddings_list.append(gap.cpu().numpy().astype(np.float32))

        return np.vstack(embeddings_list).astype(np.float32)

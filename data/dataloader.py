from pathlib import Path

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader

from data.utils import LowLightDataset


class LowLightDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        bench_dir: str,
        infer_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dir: Path = Path(train_dir)
        self.valid_dir: Path = Path(valid_dir)
        self.bench_dir: Path = Path(bench_dir)
        self.infer_dir: Path = Path(infer_dir)

        self.image_size: int = image_size
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        # ✅ 단일 데이터셋만 가정
        self.train_dataset: LowLightDataset | None = None
        self.valid_dataset: LowLightDataset | None = None
        self.bench_dataset: LowLightDataset | None = None
        self.infer_dataset: LowLightDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._set_dataset(
                data_dir=self.train_dir, augment=True, crop=True
            )

        if stage in (None, "fit", "validate"):
            self.valid_dataset = self._set_dataset(
                data_dir=self.valid_dir, augment=False, crop=False
            )

        if stage in (None, "test"):
            self.bench_dataset = self._set_dataset(
                data_dir=self.bench_dir, augment=False, crop=False
            )

        if stage in (None, "predict"):
            self.infer_dataset = self._set_dataset(
                data_dir=self.infer_dir, augment=False, crop=False
            )

    def _set_dataset(
        self,
        data_dir: Path,
        augment: bool,
        crop: bool,
    ) -> LowLightDataset | None:
        if not data_dir.exists():
            return None

        return LowLightDataset(
            path=data_dir,
            image_size=self.image_size,
            augment=augment,
            crop=crop,
        )

    def _set_dataloader(
        self,
        dataset: LowLightDataset | None,
        shuffle: bool = True,
        batch_size: int | None = None,
    ) -> DataLoader[tuple[Tensor, Tensor]] | None:
        if dataset is None:
            return None

        batch_size = batch_size if batch_size is not None else self.batch_size

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": True,
        }

        return DataLoader(dataset=dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]] | None:
        return self._set_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]] | None:
        return self._set_dataloader(
            dataset=self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]] | None:
        return self._set_dataloader(
            dataset=self.bench_dataset,
            shuffle=False,
            batch_size=1,
        )

    def predict_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]] | None:
        return self._set_dataloader(
            dataset=self.infer_dataset,
            shuffle=False,
            batch_size=1,
        )

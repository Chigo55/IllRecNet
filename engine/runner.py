from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor

from data.dataloader import LowLightDataModule
from utils.utils import save_images


class _BaseRunner(ABC):
    def __init__(
        self,
        model: LightningModule,
        params: dict[str, dict[str, Any]],
    ) -> None:
        self.model: LightningModule = model
        self.runner_params: dict[str, Any] = params["runner"]
        self.logger_params: dict[str, Any] = self.runner_params["logger"]
        self.callbacks_params: dict[str, Any] = self.runner_params["callbacks"]
        self.datamodule_params: dict[str, Any] = self.runner_params["datamodule"]
        self.trainer_params: dict[str, Any] = self.runner_params["trainer"]

        self.set_values()

        self.trainer: Trainer = self._build_trainer()
        self.datamodule: LightningDataModule = self._build_datamodule()

    def set_values(self) -> None:
        # logger
        self.save_dir = self.logger_params.get("save_dir", "runs/")
        self.experiment = self.logger_params.get("experiment", "test/")
        self.inference = self.logger_params.get("inference", "inference/")
        # callbacks
        self.monitor = self.callbacks_params.get("monitor", "valid/3_total")
        self.patience = self.callbacks_params.get("patience", 25)
        # datamodule
        self.train_dir = self.datamodule_params.get("train_dir", "data/1_train")
        self.valid_dir = self.datamodule_params.get("valid_dir", "data/2_valid")
        self.bench_dir = self.datamodule_params.get("bench_dir", "data/3_bench")
        self.infer_dir = self.datamodule_params.get("infer_dir", "data/4_infer")
        self.image_size = self.datamodule_params.get("image_size", 256)
        self.batch_size = self.datamodule_params.get("batch_size", 16)
        self.num_workers = self.datamodule_params.get("num_workers", 10)
        # trainer
        self.accelerator = self.trainer_params.get("accelerator", "auto")
        self.devices = self.trainer_params.get("devices", 1)
        self.precision = self.trainer_params.get("precision", "32-true")
        self.max_epochs = self.trainer_params.get("max_epochs", 100)
        self.log_every_n_steps = self.trainer_params.get("log_every_n_steps", 5)
        self.logger = self._build_logger()
        self.callbacks = self._build_callbacks()

    def _build_logger(self) -> TensorBoardLogger:
        return TensorBoardLogger(
            save_dir=self.save_dir,
            name=self.experiment,
        )

    def _build_callbacks(self) -> list[Callback]:
        return [
            ModelCheckpoint(
                monitor=self.monitor,
                save_top_k=1,
                filename="best",
                save_last=True,
            ),
            ModelCheckpoint(
                every_n_epochs=5,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def _build_datamodule(self) -> LowLightDataModule:
        return LowLightDataModule(
            train_dir=self.train_dir,
            valid_dir=self.valid_dir,
            bench_dir=self.bench_dir,
            infer_dir=self.infer_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _build_trainer(self) -> Trainer:
        return Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            logger=self._build_logger(),
            callbacks=self._build_callbacks(),
            max_epochs=self.max_epochs,
            log_every_n_steps=self.log_every_n_steps,
            deterministic=True,
        )

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class LightningTrainer(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Training...")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Training Completed.")


class LightningValidater(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Validating...")
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Validation Completed.")


class LightningBenchmarker(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Benchmarking...")
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Benchmark Completed.")


class LightningInferencer(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Inferencing...")
        output = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
        )
        if output is None:
            print("[WARN] Prediction returned None. Skipping save.")
            return

        else:
            output = cast(list[list[Tensor]], output)
            out_dir = Path().joinpath(
                self.save_dir,
                self.experiment,
                f"version_{self.logger.version}",
                self.inference,
            )

            save_images(batch_list=output, out_dir=out_dir)
            print("[INFO] Inference Completed.")

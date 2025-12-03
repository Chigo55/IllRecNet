from typing import Any, Literal, Sequence

import lightning as L
import torch
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers import get_cosine_schedule_with_warmup

from model.blocks.enhancer import Enhancer
from model.loss import MeanAbsoluteError, MeanSquaredError
from utils.metrics import ImageQualityMetrics


class LowLightEnhancerLightning(L.LightningModule):
    def __init__(
        self,
        params: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.model_params: dict[str, Any] = params["model"]
        self.hyper_params: dict[str, Any] = self.model_params["hyper"]
        self.optimizer_params: dict[str, Any] = self.model_params["optimizer"]

        self.save_hyperparameters(params)
        self.set_values()

        self.model: Enhancer = Enhancer(
            channels=self.channels,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_resolution=self.num_resolution,
            dropout_ratio=self.dropout_ratio,
        )

        self.mae_loss: MeanAbsoluteError = MeanAbsoluteError().eval()
        self.mse_loss: MeanSquaredError = MeanSquaredError().eval()

        self.metric: ImageQualityMetrics = ImageQualityMetrics().eval()

    def forward(
        self,
        low: Tensor,
    ) -> Tensor:
        return self.model(low)

    def set_values(self) -> None:
        # hyper
        self.channels = self.hyper_params.get("channels", 3)
        self.kernel_size = self.hyper_params.get("kernel_size", 15)
        self.sigma = self.hyper_params.get("sigma", 5)
        self.embed_dim = self.hyper_params.get("embed_dim", 32)
        self.num_heads = self.hyper_params.get("num_heads", 2)
        self.mlp_ratio = self.hyper_params.get("mlp_ratio", 2)
        self.num_resolution = self.hyper_params.get("num_resolution", 2)
        self.dropout_ratio = self.hyper_params.get("dropout_ratio", 0.2)
        # optimizer
        self.lr = self.optimizer_params.get("lr", 1e-4)
        self.betas = self.optimizer_params.get("betas", (0.9, 0.999))
        self.eps = self.optimizer_params.get("eps", 1e-8)
        self.weight_decay = self.optimizer_params.get("weight_decay", 0.1)
        self.warmup_ratio = self.optimizer_params.get("warmup_ratio", 0.1)

    def _calculate_loss(
        self,
        outputs: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        loss_mae: Tensor = self.mae_loss(outputs, target)
        loss_mse: Tensor = self.mse_loss(outputs, target)
        loss_total: Tensor = loss_mae + loss_mse

        loss_dict: dict[str, Tensor] = {
            "mae": loss_mae,
            "mse": loss_mse,
            "total": loss_total,
        }
        return loss_dict

    def _shared_step(
        self,
        batch: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        low_img, high_img = batch
        outputs = self.forward(low=low_img)
        loss_dict = self._calculate_loss(
            outputs=torch.clip(input=outputs, min=0, max=1),
            target=high_img,
        )
        return outputs, loss_dict

    def _logging(
        self,
        stage: Literal["train", "valid"],
        outputs: Tensor,
        loss_dict: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx % 25 != 0:
            return

        self.logger.experiment.add_images(f"{stage}/results", outputs, self.global_step)

        logs: dict[str, Tensor] = {}
        for i, (key, val) in enumerate(iterable=loss_dict.items()):
            logs[f"{stage}/{i + 1}_{key}"] = val

        self.log_dict(dictionary=logs, prog_bar=True)

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        outputs, loss_dict = self._shared_step(batch=batch)

        self._logging(
            stage="train",
            outputs=outputs,
            loss_dict=loss_dict,
            batch_idx=batch_idx,
        )

        return loss_dict["total"]

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        outputs, loss_dict = self._shared_step(batch=batch)

        self._logging(
            stage="valid",
            outputs=outputs,
            loss_dict=loss_dict,
            batch_idx=batch_idx,
        )

        return loss_dict["total"]

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        low_img, high_img = batch

        outputs = self.forward(low=low_img)
        metrics = self.metric.full(
            preds=torch.clip(input=outputs, min=0, max=1),
            targets=high_img,
        )

        self.log_dict(
            dictionary={
                "test/01_PSNR": metrics["PSNR"],
                "test/02_SSIM": metrics["SSIM"],
                "test/03_LPIPS": metrics["LPIPS"],
                "test/04_NIQE": metrics["NIQE"],
                "test/05_BRISQUE": metrics["BRISQUE"],
            },
            prog_bar=True,
        )

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        low_img, _ = batch

        results = self.forward(low=low_img)
        return torch.clip(input=results, min=0, max=1)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer: Optimizer = AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        total_training_steps = int(self.trainer.estimated_stepping_batches)
        num_warmup_steps = max(1, int(total_training_steps * self.warmup_ratio))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )

        sched_cfg: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [sched_cfg]

import os
from typing import Any, Final

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "precision": "16-mixed",
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": "train/",
        "inference": "inference/",
        "monitor_metric": "valid/3_total",
        "monitor_mode": "min",
        "patience": 20,
        "train_data_path": "../data/1_train",
        "valid_data_path": "../data/2_valid",
        "bench_data_path": "../data/3_bench",
        "infer_data_path": "../data/4_infer",
        "image_size": 256,
        "batch_size": 1,
        "num_workers": 10,
        "channels": 3,
        "kernel_size": 15,
        "sigma": 5,
        "embed_dim": 8,
        "num_heads": 2,
        "mlp_ratio": 1,
        "num_resolution": 2,
        "dropout_ratio": 0.2,
    }
    return hparams


DEFAULT_CHECKPOINT: Final[str] = "runs/train/version_2/checkpoints/best.ckpt"


def main() -> None:
    hparams: dict[str, Any] = get_hparams()

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        hparams=hparams,
        checkpoint_path=DEFAULT_CHECKPOINT,
    )

    engine.bench()


if __name__ == "__main__":
    main()

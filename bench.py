import os
from typing import Any

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_params() -> dict[str, Any]:
    params: dict[str, dict[str, Any]] = {
        "runner": {
            "logger": {
                "save_dir": "runs/",
                "experiment": "bench/",
                "inference": "inference/",
            },
            "callbacks": {
                "monitor": "bench/3_total",
                "patience": 25,
            },
            "datamodule": {
                "train_dir": "data/1_train",
                "valid_dir": "data/2_valid",
                "bench_dir": "data/3_bench",
                "infer_dir": "data/4_infer",
                "image_size": 256,
                "batch_size": 4,
                "num_workers": 10,
            },
            "trainer": {
                "accelerator": "auto",
                "devices": 1,
                "precision": "16-mixed",
                "max_epochs": 100,
                "log_every_n_steps": 5,
            },
        },
        "model": {
            "hyper": {
                "channels": 3,
                "kernel_size": 15,
                "sigma": 5,
                "embed_dim": 16,
                "num_heads": 2,
                "mlp_ratio": 1,
                "num_resolution": 2,
                "dropout_ratio": 0.2,
            },
            "optimizer": {
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.1,
                "warmup_ratio": 0.1,
            },
        },
    }
    return params


small_weights = "runs/train/version_0/checkpoints/best.ckpt"
base_weights = "runs/train/version_1/checkpoints/best.ckpt"
large_weights = "runs/train/version_2/checkpoints/best.ckpt"


def main() -> None:
    params: dict[str, Any] = get_params()

    params["model"]["hyper"]["embed_dim"] = 16
    params["model"]["hyper"]["num_heads"] = 2
    params["model"]["hyper"]["mlp_ratio"] = 1
    params["model"]["hyper"]["num_resolution"] = 2

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        checkpoint_path=None,
        params=params,
    )
    engine.bench()

    params["model"]["hyper"]["embed_dim"] = 32
    params["model"]["hyper"]["num_heads"] = 4
    params["model"]["hyper"]["mlp_ratio"] = 2
    params["model"]["hyper"]["num_resolution"] = 3

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        checkpoint_path=base_weights,
        params=params,
    )
    engine.bench()

    params["model"]["hyper"]["embed_dim"] = 64
    params["model"]["hyper"]["num_heads"] = 8
    params["model"]["hyper"]["mlp_ratio"] = 4
    params["model"]["hyper"]["num_resolution"] = 4

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        checkpoint_path=large_weights,
        params=params,
    )
    engine.bench()


if __name__ == "__main__":
    main()

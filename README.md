# IllRecNet

An experimental low-light image enhancement pipeline built with PyTorch and Lightning. This project uses an Encoder-Decoder architecture with self-attention and cross-attention mechanisms to restore and enhance images from low-light conditions. The core idea is to separate illumination and reflectance components, enhance them in a learned feature space, and reconstruct the final image.

## Repository Layout

```
.
├── data/
│   ├── dataloader.py
│   └── utils.py
├── engine/
│   ├── engine.py
│   └── runner.py
├── model/
│   ├── model.py
│   ├── loss.py
│   └── blocks/
│       ├── attention.py
│       ├── enhancer.py
│       ├── flatten.py
│       ├── sampling.py
│       └── separation.py
├── utils/
│   ├── metrics.py
│   └── utils.py
├── train.py
├── valid.py
├── bench.py
├── infer.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Environment Management

This project supports both `uv` and `pip` for environment management.

### Using `uv` (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Create and sync the environment from pyproject.toml
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Run a script
python train.py
```

### Using `pip`

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows
# .venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

The model is an `Enhancer` composed of an `Encoder` and a `Decoder`.

| Component | File | Description |
| --- | --- | --- |
| **Data Loading** | `data/dataloader.py` | `LowLightDataModule` creates data loaders for train, validation, benchmark, and inference stages. |
| **Separation Block** | `model/blocks/separation.py` | Approximates homomorphic filtering by using a Gaussian kernel to separate each color channel into illumination and reflectance maps. |
| **Encoder** | `model/blocks/enhancer.py` | Processes the separated illumination and reflectance maps using self-attention and cross-attention blocks to produce a contextual feature map. |
| **Decoder** | `model/blocks/enhancer.py` | A U-Net-like structure with cross-attention that takes the original image and the encoder's context map to reconstruct the enhanced image. It uses skip connections and attention at multiple resolutions. |
| **Attention Blocks**| `model/blocks/attention.py` | `SelfAttentionBlock` and `CrossAttentionBlock` are the core transformer-based components for feature processing. |
| **Lightning Wrapper** | `model/model.py` | `LowLightEnhancerLightning` wraps the `Enhancer` model. It defines the loss functions (MAE, MSE), the optimization logic (AdamW with cosine warmup), and the training/validation/testing steps. |
| **Engine** | `engine/engine.py` | The `LightningEngine` provides a clean interface to run training, validation, benchmarking, and inference by handling the PyTorch Lightning `Trainer` setup. |
| **Runners** | `train.py`, `valid.py`, `bench.py`, `infer.py` | Individual scripts for each pipeline stage. Each script configures its hyperparameters and invokes the `LightningEngine`. |
| **Metrics** | `utils/metrics.py` | Computes image quality metrics like PSNR, SSIM, LPIPS, NIQE, and BRISQUE. |

## Dataset Expectation

The pipeline expects a specific directory structure for the datasets. Each split (`1_train`, `2_valid`, etc.) can contain multiple sub-datasets (e.g., `LOLv1`, `LOLv2`). Inside each sub-dataset, you must provide matching `low/` and `high/` subfolders.

Example layout:
```
data/
├── 1_train/
│   └── LOLv1/
│       ├── low/
│       └── high/
├── 2_valid/
│   └── LOLv1/
│       ├── low/
│       └── high/
├── 3_bench/
│   └── LOLv1/
│       ├── low/
│       └── high/
└── 4_infer/
    └── LOLv1/
        ├── low/
        └── high/
```
File names inside corresponding `low/` and `high/` directories must be identical (e.g., `001.png`).

## Running the Pipeline

Activate your virtual environment and run the desired script. Each script is a standalone entry point for a specific task.

### Training

The `train.py` script starts the training process. By default, it runs two training sessions: one with the separation block frozen and one with it being trainable.

```bash
python train.py
```

### Validation

To evaluate the model on the validation set during or after training, use `valid.py`. You must provide a path to a model checkpoint.

```bash
python valid.py --checkpoint_path="path/to/your/checkpoint.ckpt"
```

### Benchmarking

To get quantitative metrics (PSNR, SSIM, etc.) on the benchmark dataset, use `bench.py`.

```bash
python bench.py --checkpoint_path="path/to/your/checkpoint.ckpt"
```

### Inference

To generate enhanced images from the inference set, use `infer.py`. The enhanced images will be saved in the `runs/<experiment_name>/inference` directory.

```bash
python infer.py --checkpoint_path="path/to/your/checkpoint.ckpt"
```
*Note: The runner scripts (`valid.py`, `bench.py`, `infer.py`) are not fully implemented to accept CLI arguments yet. You may need to modify the `get_hparams` function in each file to set the checkpoint path manually.*

## Hyperparameters

Hyperparameters for each stage are defined in a `get_hparams()` function within the corresponding script (`train.py`, `valid.py`, etc.). Key hyperparameters in `train.py` include:

- **Data**: `train_data_path`, `valid_data_path`, `image_size`, `batch_size`.
- **Training**: `seed`, `max_epochs`, `accelerator`, `precision`, `log_dir`, `experiment_name`, `patience` (for early stopping).
- **Model**: `embed_dim`, `num_heads`, `mlp_ratio`, `num_resolution`, `dropout_ratio`.
- **Optimizer**: Learning rate, betas, weight decay, and other optimizer-specific parameters are configured inside the `configure_optimizers` method in `model/model.py`.

## Metrics and Logging

- Training and validation losses are logged to TensorBoard.
- Image quality metrics (PSNR, SSIM, etc.) are calculated during the `test` step (used by `bench.py`).
- Logs and checkpoints are saved to `log_dir/experiment_name`. By default, this is `runs/train/`.
- The best checkpoint is saved based on the `valid/3_total` loss.
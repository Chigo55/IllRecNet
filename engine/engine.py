from typing import Any

from lightning import LightningModule
from engine.runner import (
    LightningBenchmarker,
    LightningInferencer,
    LightningTrainer,
    LightningValidater,
    _BaseRunner,
)


class LightningEngine:
    def __init__(
        self,
        model_class: type[LightningModule],
        checkpoint_path: str | None,
        params: dict[str, dict[str, Any]],
    ) -> None:
        self.checkpoint_path: str | None = checkpoint_path
        self.params: dict[str, dict[str, Any]] = params

        if checkpoint_path:
            print(f"Loaded model from: {self.checkpoint_path}")
            self.model = model_class.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                params=self.params,
            )
        else:
            print("Initialized model from scratch.")
            self.model = model_class(
                params=self.params,
            )

    def _create_and_run_runner(
        self,
        runner_class: type[_BaseRunner],
    ) -> None:
        runner: _BaseRunner = runner_class(
            model=self.model,
            params=self.params,
        )
        runner.run()

    def train(self) -> None:
        self._create_and_run_runner(runner_class=LightningTrainer)

    def valid(self) -> None:
        self._create_and_run_runner(runner_class=LightningValidater)

    def bench(self) -> None:
        self._create_and_run_runner(runner_class=LightningBenchmarker)

    def infer(self) -> None:
        self._create_and_run_runner(runner_class=LightningInferencer)

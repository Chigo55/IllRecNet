# engine-runner (Train/Valid/Test Runners)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#engine` `#runner` `#trainer`

## 개요
PyTorch Lightning의 `Trainer`를 래핑(Wrapping)하여 학습, 검증, 벤치마크, 추론 과정을 추상화한 `_BaseRunner` 및 하위 클래스들.

## 주요 로직

### `_BaseRunner` — 공통 기반 로직
- **초기화**: 주어진 하이퍼파라미터(`logger`, `callbacks`, `datamodule`, `trainer`)를 파싱하여 인스턴스 변수로 저장.
- **`_build_logger`**: `TensorBoardLogger` 생성.
- **`_build_callbacks`**: `ModelCheckpoint` (best 및 epoch 단위 저장), `EarlyStopping`, `LearningRateMonitor` 등 설정.
- **`_build_datamodule`**: [[.wiki/domain/data-dataloader]] 객체 초기화.
- **`_build_trainer`**: Lightning `Trainer` 객체 생성.
- **`run`**: 구체 클래스에서 구현해야 할 추상 메서드.

### `LightningTrainer`, `LightningValidater`, `LightningBenchmarker`
각각 `trainer.fit()`, `trainer.validate()`, `trainer.test()`를 호출.

### `LightningInferencer`
`trainer.predict()`를 호출하여 추론을 수행. 결과물(이미지 텐서 리스트)이 존재하면 `save_images` 유틸리티를 사용해 디스크에 저장.

## 연결
- [[.wiki/domain/engine-engine]] — 이 러너 클래스들을 인스턴스화하고 실행하는 엔진.
- [[.wiki/domain/data-dataloader]] — `_build_datamodule`에서 활용.
- [[.wiki/domain/utils-utils]] — `save_images` 기능 사용.

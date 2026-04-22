# Engine Module

> **관련 문서**: [[.wiki/index]] | [[.wiki/concept/data-module]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#engine` `#lightning` `#runner`

## 정의
`engine/engine.py`와 `engine/runner.py`에 구현된 모델 학습 및 실행 래퍼(wrapper) 모듈. PyTorch Lightning의 `Trainer`를 기반으로 학습(Train), 검증(Valid), 테스트(Bench), 추론(Infer) 파이프라인을 캡슐화한다.

## 상세

### LightningEngine (`engine/engine.py`)
주어진 모델 클래스, 체크포인트 경로, 파라미터(`params`)를 받아 모델을 초기화하거나 체크포인트를 로드하는 역할을 수행한다. 내부적으로 목적에 맞는 `_BaseRunner`를 생성하여 실행한다.
- `train()`: `LightningTrainer` 생성 및 실행
- `valid()`: `LightningValidater` 생성 및 실행
- `bench()`: `LightningBenchmarker` 생성 및 실행
- `infer()`: `LightningInferencer` 생성 및 실행

### Runner (`engine/runner.py`)
`_BaseRunner` 클래스가 템플릿(Base)으로 작동하며 PyTorch Lightning의 `Trainer`와 커스텀 콜백(ModelCheckpoint, EarlyStopping 등), 로거(TensorBoardLogger), 그리고 `LowLightDataModule`을 설정값 기반으로 동적 생성한다.
- **LightningTrainer**: `trainer.fit()` 실행 (학습)
- **LightningValidater**: `trainer.validate()` 실행 (검증)
- **LightningBenchmarker**: `trainer.test()` 실행 (벤치마크 테스트)
- **LightningInferencer**: `trainer.predict()` 실행 후 결과를 이미지 형태(`save_images`)로 저장 (추론)

## 연결
- [[.wiki/concept/data-module]] — Runner에서 사용하는 LowLightDataModule

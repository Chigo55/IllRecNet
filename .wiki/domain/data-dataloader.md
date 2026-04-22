# data-dataloader (DataModule)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#data` `#dataloader` `#lightning`

## 개요
PyTorch Lightning의 `LightningDataModule`을 상속받아, 학습/검증/벤치마크/추론에 사용될 데이터로더를 관리하는 모듈.

## 주요 로직

### `setup` — 데이터셋 준비
각 실행 단계(`fit`, `validate`, `test`, `predict`)에 맞춰 `LowLightDataset` 객체를 생성. 학습 시에만 데이터 증강(`augment`)과 크롭(`crop`)을 활성화.

### `_set_dataloader` — 데이터로더 생성 공통 로직
주어진 데이터셋에 대해 PyTorch `DataLoader`를 생성. `num_workers`, `pin_memory` 등의 병렬 처리 및 메모리 최적화 옵션을 설정.

### `train/val/test/predict_dataloader` — 단계별 로더 제공
각 단계에 맞는 데이터로더를 반환. 벤치마크(test)와 추론(predict) 시에는 배치 사이즈를 1로 고정하여 순차적 처리를 지원.

## 특이사항
- 단일 데이터셋(`LowLightDataset`)만 가정한 구조.
- `persistent_workers`를 통해 데이터 로딩 오버헤드 최소화.

## 연결
- [[.wiki/data-utils]] — `LowLightDataset` 데이터셋 클래스 의존.

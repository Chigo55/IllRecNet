# Data Module

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#data` `#lightning` `#dataset`

## 정의
`data/dataloader.py`와 `data/utils.py`에 구현된 파이프라인 모듈로, PyTorch Lightning의 `LightningDataModule`을 기반으로 한 `LowLightDataModule`과 PyTorch의 `Dataset`을 기반으로 한 `LowLightDataset`으로 구성된다.

## 상세

### LowLightDataset (`data/utils.py`)
- 하위 경로(`low`, `high`)에서 저조도 및 정상 조도 이미지 쌍을 로드한다.
- **데이터 증강(Augmentation)**: Random Horizontal Flip, Vertical Flip, 90도 단위 Rotate (각 50% 확률).
- **크롭(Crop)**: 설정한 `image_size`에 맞게 Random Crop 수행. 크기가 작을 경우 `reflect` 패딩을 적용한다.
- **크기 정규화**: 크롭을 사용하지 않을 경우 이미지 크기가 32의 배수가 되도록 `reflect` 패딩을 자동 적용한다.

### LowLightDataModule (`data/dataloader.py`)
- `train_dir`, `valid_dir`, `bench_dir`, `infer_dir` 4가지 데이터 경로를 받아 상황에 맞게 데이터셋 인스턴스를 초기화(setup)한다.
- 학습(Train) 단계에서는 `augment=True`, `crop=True`를 적용하고, 검증/테스트/추론 단계에서는 적용하지 않는다.
- 데이터 로더(DataLoader) 생성 로직을 캡슐화하여 PyTorch Lightning 학습 루프에 제공한다.

## 연결
- [[.wiki/index]] — 전체 카탈로그
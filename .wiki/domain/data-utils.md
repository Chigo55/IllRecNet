# data-utils (Dataset)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#data` `#dataset`

## 개요
저조도(Low-light) 이미지와 정상(High-light) 이미지 쌍을 로드하고 증강(Augmentation)하는 PyTorch `Dataset` 클래스(`LowLightDataset`) 정의.

## 화면 구조
UI 없음. 데이터 전처리를 위한 순수 파이썬 클래스.

## 주요 로직

### `__getitem__` — 데이터 반환
인덱스에 해당하는 low/high 이미지 쌍을 `PIL.Image`로 로드.
옵션에 따라 데이터 증강(`_pair_augment`), 랜덤 크롭(`_pair_random_crop`), 혹은 32배수 패딩(`_pad_to_multiple`)을 적용 후 `Tensor`로 변환하여 반환.

### `_pair_augment` — 데이터 증강
50% 확률로 수평 뒤집기, 50% 확률로 수직 뒤집기를 수행. 또한 0, 90, 180, 270도 중 무작위로 회전하여 데이터를 증강. 두 이미지(low, high)에 동일한 변환을 적용.

### `_pair_random_crop` — 무작위 자르기
지정된 `patch_size`보다 이미지가 작을 경우 `reflect` 패딩을 적용. 그 후 무작위 시작 좌표를 생성하여 두 이미지를 동일한 영역에서 잘라냄(Crop).

### `_pad_to_multiple` — 크기 맞춤
이미지 크기를 네트워크(특히 Downsampling) 처리에 적합하도록 특정 배수(기본 32)로 패딩. `reflect` 모드를 사용.

## 특이사항
- 데이터셋의 디렉토리 구조는 반드시 `low/` 및 `high/` 하위 폴더를 가져야 함.

## 연결
- [[.wiki/data-dataloader]] — 이 데이터셋을 활용하는 DataModule.

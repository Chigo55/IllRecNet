# utils-utils (Utility Functions)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#utils`

## 개요
디렉토리 관리, 이미지 시각화 및 저장, 모델 구조 요약, 가중치 초기화 등 잡다한 편의 기능을 제공하는 유틸리티 스크립트.

## 주요 로직

### `show_batch` — 시각화
텐서 형태의 이미지 배치(`B, C, H, W`)를 받아 Matplotlib을 사용해 그리드 형태로 시각화.

### `make_dirs` — 디렉토리 생성
주어진 경로에 대한 디렉토리가 없으면 부모 경로를 포함해 자동 생성(`mkdir(parents=True, exist_ok=True)`).

### `save_images` — 결과 저장
`torchvision.utils.save_image`를 사용해 이미지 텐서를 파일(PNG 등)로 저장. 출력 배치를 순회하며 디렉토리를 만들고 일관된 접두사(`prefix`)를 붙여 파일 이름 부여.

### `count_parameters` & `summarize_model` — 모델 분석
- `count_parameters`: `requires_grad=True`인 학습 가능 파라미터 총 개수를 반환.
- `summarize_model`: `torchinfo.summary`를 호출하여 모델 레이어별 구조와 파라미터 수를 상세 출력.

### `weights_init` — 가중치 초기화
Conv, Linear, BatchNorm 레이어에 대한 초기화 규칙(`xavier_normal_`, `constant_`)을 정의. `model.apply(weights_init)` 형식으로 사용.

## 연결
- [[.wiki/engine-runner]] — 추론 완료 후 `save_images`를 사용해 디스크에 결과 저장.

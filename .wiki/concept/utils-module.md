# Utils Module

> **관련 문서**: [[.wiki/index]] | [[.wiki/concept/model-module]] | [[.wiki/concept/engine-module]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#utils` `#metrics`

## 정의
`utils/` 디렉토리에 구현된 헬퍼 함수와 성능 평가용 지표(Metrics) 클래스 모음.

## 상세

### ImageQualityMetrics (`utils/metrics.py`)
`pyiqa` 패키지를 사용하여 이미지 품질을 정량적으로 평가하기 위한 클래스. PyTorch의 `nn.Module`을 상속한다.
- **Reference 기반 지표 (`forward`)**: 정답(target) 이미지가 필요한 지표로 `PSNR`, `SSIM`, `LPIPS`를 계산한다.
- **No-Reference 기반 지표 (`no_ref`)**: 정답 없이 평가 가능한 지표로 `BRISQUE`, `NIQE`를 계산한다.
- **Full Metrics (`full`)**: 두 가지 부류의 지표를 모두 계산하여 딕셔너리로 반환한다. (예: `test_step` 시 사용)

### 유틸리티 함수 (`utils/utils.py`)
- **`show_batch`**: matplotlib을 활용해 텐서 형태의 이미지 배치를 그리드로 시각화.
- **`make_dirs`**: 경로 생성 (존재 시 무시).
- **`print_metrics`**: 딕셔너리 형태의 지표를 콘솔에 출력.
- **`save_images`**: 추론(Inference) 결과를 디스크에 저장. (`torchvision.utils.save_image` 활용)
- **`count_parameters` / `summarize_model`**: 모델 파라미터 개수 파악 및 torchinfo를 통한 요약 출력.
- **`weights_init`**: 모델의 컨볼루션(Conv), 선형(Linear), 배치정규화(BatchNorm) 레이어 등에 대한 기본 가중치 초기화(Xavier Normal 등).

## 연결
- [[.wiki/concept/model-module]] — 이 모듈 내 `test_step`에서 ImageQualityMetrics 사용
- [[.wiki/concept/engine-module]] — Inference 단계에서 `save_images` 사용

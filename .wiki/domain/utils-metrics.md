# utils-metrics (Image Quality Metrics)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#utils` `#metrics` `#pyiqa`

## 개요
`pyiqa` 라이브러리를 활용하여 이미지 화질 평가 지표(Image Quality Metrics)를 계산하는 래퍼 모듈 `ImageQualityMetrics`.

## 주요 로직

### `__init__` — 측정 지표 초기화
다섯 가지 측정 지표를 `pyiqa.create_metric`으로 불러옴.
- **Reference 기반**: PSNR, SSIM, LPIPS
- **No-Reference 기반**: BRISQUE, NIQE

### `forward` — Reference 기반 측정
입력된 추론 이미지(`preds`)와 정답 이미지(`targets`) 간의 PSNR, SSIM, LPIPS 값을 계산하여 딕셔너리 형태로 반환. (배치 평균 `mean().item()` 사용)

### `no_ref` — No-Reference 기반 측정
정답 이미지가 필요 없는 지표인 BRISQUE, NIQE 값을 계산하여 반환.

### `full` — 전체 측정
Reference 기반과 No-Reference 기반 지표를 모두 계산하고 결과를 병합하여 하나의 딕셔너리로 반환. 벤치마크/테스트 시 주로 활용됨.

## 연결
- [[.wiki/domain/model-model]] — `LowLightEnhancerLightning` 내부에서 정량적 평가를 위해 사용.

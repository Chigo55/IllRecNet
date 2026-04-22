# bench (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#bench` `#script`

## 개요
학습된 모델 가중치를 기반으로 다양한 사이즈(small, base, large)의 모델에 대한 벤치마크 평가를 수행하는 스크립트.

## 화면 구조
파라미터 설정 함수(`get_params`)와 벤치마크 실행 로직(`main`)으로 구성.

## 주요 로직

### 모델별 벤치마크 수행
`main` 함수에서 `small_weights`, `base_weights`, `large_weights`의 3가지 체크포인트를 사용해 평가.
- Small 모델: `embed_dim=16`, `num_heads=2`, `num_resolution=2`
- Base 모델: `embed_dim=32`, `num_heads=4`, `num_resolution=3`
- Large 모델: `embed_dim=64`, `num_heads=8`, `num_resolution=4`
각 설정마다 `LightningEngine`을 재초기화하고 `.bench()`를 호출.

## 특이사항
- CUDA 장치 `1`번 고정.
- 모델 성능 측정을 목적으로 하므로 `train_dir`, `valid_dir` 등의 경로는 일반적인 `data/` 하위를 가리킴.

## 연결
- [[.wiki/engine-engine]] — 엔진의 벤치마크 기능 호출.
- [[.wiki/model-model]] — 모델 아키텍처.

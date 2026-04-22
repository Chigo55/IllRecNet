# infer (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#infer` `#script`

## 개요
다양한 모델 크기(small, base, large)에 대한 추론(Inference)을 수행하는 스크립트.

## 화면 구조
파라미터 설정과 추론 로직을 담고 있는 스크립트.

## 주요 로직

### `main` — 추론 흐름
세 가지 모델 설정(Small, Base, Large)에 맞춰 하이퍼파라미터를 변경하며 `LightningEngine` 인스턴스를 초기화한 후, `.infer()` 메서드를 통해 이미지 개선을 수행.
사용되는 체크포인트: `small_weights`, `base_weights`, `large_weights`

## 특이사항
- 벤치마크(`bench.py`)와 유사한 구조이나, 정량적 평가가 아닌 최종 결과물(이미지) 생성에 초점을 맞춤.
- CUDA 장치 `1`번 고정.

## 연결
- [[.wiki/engine-engine]] — 추론 기능 호출.
- [[.wiki/model-model]] — 추론할 모델.

# valid (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#valid` `#script`

## 개요
학습된 모델의 검증(Validation) 세트에 대한 평가를 진행하는 스크립트.

## 화면 구조
파라미터 설정과 검증 로직으로 구성됨.

## 주요 로직

### 모델 검증 수행
`main` 함수 내에서 모델의 크기(Small, Base, Large)에 따라 하이퍼파라미터를 다르게 설정하고, 미리 학습된 가중치(`best.ckpt`)를 불러와 `LightningEngine`의 `.valid()` 메서드를 호출.

## 특이사항
- 검증 셋을 사용하여 모델의 일반화 성능을 확인.
- CUDA 장치 `1`번 고정.

## 연결
- [[.wiki/domain/engine-engine]] — 검증 기능.
- [[.wiki/domain/model-model]] — 검증 모델.

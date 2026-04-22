# 02_LOLv2real_train (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#train` `#script` `#lolv2real`

## 개요
LOLv2-real 데이터셋을 사용하여 LowLightEnhancerLightning 모델을 학습시키는 진입점 스크립트.

## 화면 구조
스크립트 파일로서 `get_params()`를 통해 학습 환경 및 모델 하이퍼파라미터를 설정.

## 주요 로직

### `get_params` — 하이퍼파라미터 정의
러너(logger, callbacks, datamodule, trainer 설정)와 모델 파라미터를 반환.
- `experiment`: `02_LOLv2real/`
- `datamodule`: `data/02_LOLv2real/*` 경로 사용.

### `main` — 실행 흐름
`get_params`의 파라미터를 기반으로 `LightningEngine`을 초기화하고, 학습(`train`), 벤치마크(`bench`), 추론(`infer`) 과정을 실행.

## 특이사항
- 모델의 `embed_dim`, `num_heads` 등을 소규모 버전으로 재설정하여 실험.
- CUDA 장치 `1`번 고정.

## 연결
- [[.wiki/engine-engine]] — Lightning 엔진 호출을 위함.
- [[.wiki/model-model]] — 학습 대상 모델.

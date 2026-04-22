# 03_LOLv2synthetic_train (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#train` `#script` `#lolv2synthetic`

## 개요
LOLv2-synthetic 데이터셋을 사용하여 LowLightEnhancerLightning 모델을 학습시키는 진입점 스크립트.

## 화면 구조
설정값을 딕셔너리로 관리하는 스크립트 파일.

## 주요 로직

### `get_params` — 하이퍼파라미터 정의
학습기, 데이터 로더, 모델의 주요 하이퍼파라미터 설정.
- `experiment`: `03_LOLv2synthetic/`
- `datamodule`: `data/03_LOLv2synthetic/*` 경로 사용.

### `main` — 실행 흐름
설정 파라미터를 바탕으로 `LightningEngine` 인스턴스를 만들고, `.train()`, `.bench()`, `.infer()` 과정을 연달아 수행.

## 특이사항
- 다른 스크립트와 달리 `os.environ["CUDA_VISIBLE_DEVICES"]` 설정이 포함되어 있지 않음.

## 연결
- [[.wiki/engine-engine]] — Lightning 엔진 실행.
- [[.wiki/model-model]] — 모델 클래스 참조.

# 01_LOLv1_train (Script)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#train` `#script` `#lolv1`

## 개요
LOLv1 데이터셋을 사용하여 LowLightEnhancerLightning 모델을 학습시키는 진입점 스크립트.

## 화면 구조
- 스크립트 파일이므로 UI는 없으나, 하이퍼파라미터를 `get_params()` 딕셔너리로 관리.
- 학습, 평가(Bench), 추론(Infer) 과정을 순차적으로 수행.

## 주요 로직

### `get_params` — 하이퍼파라미터 정의
러너(logger, callbacks, datamodule, trainer 설정)와 모델(hyper, optimizer)의 파라미터를 딕셔너리로 정의.
- `experiment`: `01_LOLv1/`
- `datamodule`: `data/01_LOLv1/*` 경로 사용.

### `main` — 실행 흐름
모델의 크기(embed_dim, num_heads 등)를 조절하고 `LightningEngine` 인스턴스를 생성.
이후 `.train()`, `.bench()`, `.infer()` 메서드를 호출하여 전체 과정을 수행.

## 특이사항
- CUDA 장치를 `1`번으로 고정 (`os.environ["CUDA_VISIBLE_DEVICES"] = "1"`).

## 연결
- [[.wiki/domain/engine-engine]] — 학습 및 추론 엔진 구현.
- [[.wiki/domain/model-model]] — 모델 구조 및 Lightning 모듈 정의.

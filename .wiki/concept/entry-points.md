# Entry Points

> **관련 문서**: [[.wiki/index]] | [[.wiki/concept/engine-module]] | [[.wiki/concept/model-module]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#entry` `#train` `#infer` `#bench`

## 정의
루트 디렉토리에 존재하는 실행 스크립트 모음으로, 모델의 학습(Train), 검증(Valid), 평가(Bench), 추론(Infer) 파이프라인을 직접 실행(Trigger)하는 진입점 역할을 한다.

## 상세
각 실행 스크립트 내부에는 `get_params()` 함수가 정의되어 있어 파라미터(`runner`, `model` 등)를 하드코딩 방식으로 주입하며, 이를 통해 `LightningEngine`을 초기화하고 실행한다.

### 학습 실행 스크립트
특정 데이터셋이나 환경에 맞게 폴더 경로와 하이퍼파라미터를 지정하여 모델을 처음부터(scratch) 학습시키고 결과를 평가한다.
- `01_LOLv1_train.py`: LOLv1 데이터셋 대상 학습
- `02_LOLv2real_train.py`: LOLv2 real 데이터셋 대상 학습
- `03_LOLv2synthetic_train.py`: LOLv2 synthetic 데이터셋 대상 학습 (파일명 규칙으로 유추)

### 기타 실행 스크립트
미리 학습된 체크포인트(checkpoint) 파일들을 로드하거나 파라미터를 변경해 가면서, 모델 크기(Small, Base, Large)별로 실행을 수행한다.
- `bench.py`: `engine.bench()` 호출. 모델 테스트 데이터셋을 이용한 벤치마크.
- `infer.py`: `engine.infer()` 호출. 실제 이미지에 대한 추론 및 결과 파일 저장.
- `valid.py`: `engine.valid()` 호출. 검증 데이터셋에 대한 평가를 수행.

## 연결
- [[.wiki/concept/engine-module]] — 모든 스크립트에서 초기화하여 구동하는 핵심 클래스 (LightningEngine)
- [[.wiki/concept/model-module]] — 훈련 및 평가 대상이 되는 메인 모델 (LowLightEnhancerLightning)

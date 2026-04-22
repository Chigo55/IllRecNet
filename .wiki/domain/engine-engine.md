# engine-engine (Lightning Wrapper)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#engine` `#wrapper`

## 개요
PyTorch Lightning의 모듈 및 러너(Trainer, Validater 등) 초기화를 담당하는 상위 래퍼(Wrapper) 클래스 `LightningEngine`.

## 주요 로직

### `__init__` — 초기화 및 모델 로드
`checkpoint_path` 존재 여부에 따라 모델을 스크래치(scratch)부터 초기화하거나, 기존 가중치(`load_from_checkpoint`)를 불러옴.

### `_create_and_run_runner` — 러너 실행
특정 러너 클래스(`_BaseRunner`를 상속받은 구체 클래스)를 인스턴스화하고 `.run()`을 호출하여 실행 과정을 위임.

### `train`, `valid`, `bench`, `infer`
각각 학습, 검증, 벤치마크, 추론 단계에 대응하는 래퍼 메서드. 해당하는 Runner 클래스를 지정하여 `_create_and_run_runner`를 호출.

## 연결
- [[.wiki/domain/engine-runner]] — 실제 실행을 담당하는 러너 클래스들.
- [[.wiki/domain/model-model]] — 이 엔진을 통해 로드되고 훈련되는 LightningModule.

# Lightning Pipeline

> **관련 문서**: [[.wiki/concept/engine-module]] | [[.wiki/concept/data-module]] | [[.wiki/concept/model-module]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: pattern
**태그**: `#pytorch-lightning` `#pipeline`

## 문제
보일러플레이트(boilerplate) PyTorch 학습 루프를 매번 작성하는 것은 오류가 발생하기 쉽고 반복적입니다. 학습, 검증, 평가 등의 과정에서 중복 코드가 많아집니다.

## 해법
PyTorch Lightning(`LightningModule`, `LightningDataModule`)을 사용하고, `Trainer`를 `Runner` 래퍼 클래스 안에 캡슐화하여 학습(train)/검증(valid)/벤치마크(bench)/추론(infer) 파이프라인을 표준화합니다. 이를 통해 각 파이프라인에서의 공통 루틴을 캡슐화하고 코드를 모듈화합니다.

## 적용 사례
- 다양한 데이터셋에 대해 일관된 학습(train), 검증(valid) 및 추론(infer) 파이프라인을 구축할 때 사용 (`01_LOLv1_train.py` 등).

## 연결
- [[.wiki/concept/engine-module]] — Trainer 래퍼 등 엔진 구조를 통한 실행 단위 캡슐화
- [[.wiki/concept/data-module]] — LightningDataModule을 통한 데이터 파이프라인 표준화
- [[.wiki/concept/model-module]] — LightningModule을 통한 모델 및 훈련 스텝 표준화

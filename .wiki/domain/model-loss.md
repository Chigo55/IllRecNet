# model-loss (Loss Functions)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#model` `#loss`

## 개요
모델 학습 시 사용될 기본 손실 함수들을 래핑(Wrapping)하여 정의한 모듈.

## 주요 로직

### `MeanAbsoluteError`
PyTorch의 `nn.L1Loss`를 상속받는 클래스. 픽셀 간 절대 오차(MAE)를 계산함.

### `MeanSquaredError`
PyTorch의 `nn.MSELoss`를 상속받는 클래스. 픽셀 간 제곱 오차(MSE)를 계산함.

> **참고**: `LowLightEnhancerLightning`에서는 이 두 손실의 합을 최종 목적 함수로 사용.

## 연결
- [[.wiki/model-model]] — 이 손실 함수들을 결합하여 사용하는 Lightning 모듈.

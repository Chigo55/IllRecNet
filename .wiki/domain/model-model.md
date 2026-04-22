# model-model (LightningModule)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#model` `#lightning` `#enhancer`

## 개요
이미지 개선 모델인 `Enhancer`와 손실 함수, 평가 지표(Metrics)를 통합한 PyTorch Lightning 모듈(`LowLightEnhancerLightning`).

## 주요 로직

### 모델 초기화 (`__init__`, `set_values`)
하이퍼파라미터를 파싱하여 `Enhancer` 네트워크를 생성. 학습에 사용할 MAE, MSE 손실 함수와 평가를 위한 `ImageQualityMetrics`를 설정. (메트릭의 가중치는 업데이트되지 않도록 `requires_grad=False` 처리)

### `_shared_step` 및 `_calculate_loss`
순전파(Forward)를 수행한 후, 출력값을 `[0, 1]` 범위로 클리핑(clip).
클리핑된 출력값과 정답(Target) 이미지를 기반으로 MAE와 MSE의 합을 총 손실(total)로 계산.

### `training_step`, `validation_step`
`_shared_step`을 통해 손실을 계산하고, `_logging`을 호출하여 TensorBoard에 손실 지표와 결과 이미지를 로깅.

### `test_step`
정량적 벤치마크 수행. `ImageQualityMetrics`를 사용해 PSNR, SSIM, LPIPS, NIQE, BRISQUE 지표를 계산하고 로깅.

### `predict_step`
추론 모드. 주어진 Low-light 이미지를 네트워크에 통과시켜 개선된 이미지를 `[0, 1]`로 클리핑하여 반환.

### `configure_optimizers`
Adam 옵티마이저를 정의하고 반환. (lr, betas, weight_decay 등 하이퍼파라미터 적용)

## 연결
- [[.wiki/domain/model-blocks-enhancer]] — 코어 네트워크 `Enhancer` 정의.
- [[.wiki/domain/model-loss]] — 사용되는 MAE/MSE 손실.
- [[.wiki/domain/utils-metrics]] — 품질 평가 지표 모음.

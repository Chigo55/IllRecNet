# Model Module

> **관련 문서**: [[.wiki/index]] | [[.wiki/concept/engine-module]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#model` `#lightning` `#loss`

## 정의
`model/` 디렉토리에 구현된 핵심 신경망 아키텍처 및 손실 함수 모음. PyTorch Lightning 모듈로 래핑되어 있어 Engine Module과 직접 연동되며, `blocks/` 하위 모듈들을 조합하여 최종적인 저조도 이미지 개선(Low-Light Enhancement) 모델을 구성한다.

## 상세

### LowLightEnhancerLightning (`model/model.py`)
- PyTorch Lightning의 `LightningModule`을 상속받은 메인 래퍼 클래스.
- **Model**: 내부에 `model.blocks.enhancer.Enhancer` 네트워크를 초기화하여 사용한다.
- **Forward**: 저조도 이미지(`low`)를 입력받아 개선된 이미지를 출력한다.
- **Loss Calculation**: `mae`(L1)와 `mse`(L2) 손실을 더하여 최종 Total Loss를 계산한다 (`loss_dict` 반환).
- **Steps**:
  - `training_step`, `validation_step`: 손실을 계산하고 `_logging`을 통해 TensorBoard에 이미지와 지표를 로깅한다.
  - `test_step`: `utils.metrics.ImageQualityMetrics`를 사용해 PSNR, SSIM, LPIPS, NIQE, BRISQUE 등의 품질 평가 지표를 로깅한다.
  - `predict_step`: 0~1 사이로 클리핑(`torch.clip`)된 최종 이미지를 반환한다.
- **Optimizer**: 기본적으로 `Adam` 옵티마이저를 구성(`configure_optimizers`)하여 사용한다.

### Loss (`model/loss.py`)
- `MeanAbsoluteError`: PyTorch의 `nn.L1Loss` 래퍼.
- `MeanSquaredError`: PyTorch의 `nn.MSELoss` 래퍼.

### Blocks (`model/blocks/`)
- `enhancer.py`: 메인 Enhancer 모델.
- `attention.py`, `flatten.py`, `sampling.py`, `separation.py` 등 신경망을 구성하는 하위 아키텍처 블록들이 포함되어 있다.

## 연결
- [[.wiki/concept/engine-module]] — 이 LightningModule을 실행하는 런너

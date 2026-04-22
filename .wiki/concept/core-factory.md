# Core Factory

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: concept
**태그**: `#core` `#factory`

## 정의
`framework/core/factory.py`에 구현된 팩토리 클래스 모음으로, 설정(config) 딕셔너리와 레지스트리(Registry) 패턴을 활용하여 모델, 손실 함수, 옵티마이저, 스케줄러 인스턴스를 동적으로 생성한다.

## 상세
다음 4가지 팩토리를 제공한다:
- `ModelFactory`: `ModelRegistry`에 등록된 모델을 설정값 기반으로 생성한다.
- `LossFactory`: `LossRegistry`를 사용하며, 여러 손실 함수가 리스트로 주어지면 가중치(weight)가 적용된 `CompositeLoss`를 자동 생성한다.
- `OptimizerFactory`: `OptimizerRegistry`를 먼저 확인하고, 없으면 PyTorch의 기본 `torch.optim` 클래스들로 fallback하여 생성한다.
- `SchedulerFactory`: `SchedulerRegistry`를 확인 후, PyTorch의 `torch.optim.lr_scheduler`로 fallback하여 생성한다.

## 연결
- [[.wiki/index]] — 전체 카탈로그

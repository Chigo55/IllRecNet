# model-blocks-separation (Separation Block)

> **관련 문서**: [[.wiki/index]]

**생성**: 2025-02-18  **갱신**: 2025-02-18
**분류**: domain
**태그**: `#model` `#blocks` `#separation`

## 개요
Retinex 이론 등에 영감을 받아, 단일 채널(Grayscale) 이미지를 조명(Illumination, `il`) 성분과 반사(Reflectance, `re`) 성분으로 분리하는 모듈.

## 주요 로직

### `__init__`
가우시안 필터 생성을 위해 `sigma` 값을 이용해 1D 가우시안 커널(`kernel_1d`)을 생성.
이 1D 커널을 Y축용 컨볼루션(`conv_y`, `(kernel_size, 1)`)과 X축용 컨볼루션(`conv_x`, `(1, kernel_size)`)의 가중치로 각각 설정. 이들 가중치는 학습되지 않도록 고정(`requires_grad = False`).

### `forward` — 성분 분리 흐름
1. 입력 이미지를 `1e-6`으로 클리핑한 후 로그 공간(log space)으로 변환(`x_log`).
2. 로그 변환된 이미지에 가우시안 필터링(Y축 -> X축 순차 적용)을 수행하여 부드러운 저주파 조명 성분(`il_log`)을 획득. 패딩은 `ReflectionPad2d`를 사용.
3. 원본 로그 이미지에서 조명 성분을 빼서 고주파 반사 성분(`re_log = x_log - il_log`)을 도출.
4. `torch.exp`를 적용하여 다시 원래 스케일로 복원하고 `[0, 1]` 범위로 클리핑하여 최종 `il`과 `re`를 반환.

## 특이사항
- 파라미터가 없는 결정론적(Deterministic) 필터링 기법 사용. 컨볼루션을 가속화하기 위해 2D 가우시안 커널을 두 개의 1D 커널(분리 가능 필터)로 나누어 적용.

## 연결
- [[.wiki/model-blocks-enhancer]] — 인코더에서 초기 채널 분리 후 각 채널별 조명/반사 성분을 추출할 때 사용.

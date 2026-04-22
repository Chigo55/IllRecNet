# model-blocks-sampling (Resampling Blocks)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#model` `#blocks` `#sampling`

## 개요
피처맵의 공간 해상도(Spatial Resolution)를 축소하거나 확대하기 위한 다운샘플링 및 업샘플링 모듈.

## 주요 로직

### `Downsampling`
`kernel_size=2`, `stride=2`인 `nn.Conv2d`를 사용하여 특징맵의 공간 해상도를 절반으로 줄이면서 채널 수를 변경(`in_channels` -> `out_channels`).

### `Upsampling`
`kernel_size=2`, `stride=2`인 `nn.ConvTranspose2d`를 사용하여 특징맵의 공간 해상도를 두 배로 키우면서 채널 수를 변경(`in_channels` -> `out_channels`).

## 연결
- [[.wiki/domain/model-blocks-enhancer]] — 디코더(Decoder) 내부에서 스케일 변경을 위해 사용.

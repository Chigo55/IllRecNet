# model-blocks-flatten (Flatten Helpers)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#model` `#blocks` `#flatten`

## 개요
이미지 텐서(4D)를 어텐션 메커니즘 처리를 위한 시퀀스 텐서(3D)로 변환하거나, 그 반대로 복원하는 헬퍼 함수들을 정의.

## 주요 로직

### `Flatten`
입력 텐서 `(B, C, H, W)`의 공간 차원(`H`, `W`)을 병합하여 `(B, C, H*W)`로 만든 후, 시퀀스 길이를 뒤로 보내 `(B, H*W, C)` 형태의 3D 텐서로 변환. 이는 `MultiheadAttention`의 `batch_first=True` 입력에 맞추기 위함.

### `Unflatten`
입력 텐서 `(B, H*W, C)`를 받아 공간 차원(`H`, `W`)을 명시하여 원래의 2D 이미지 특징맵 형태인 `(B, C, H, W)`로 복원.

## 연결
- [[.wiki/domain/model-blocks-enhancer]] — 이 헬퍼 함수들을 사용하여 어텐션 연산 전후로 텐서 형태를 변환.

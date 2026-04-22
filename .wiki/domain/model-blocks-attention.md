# model-blocks-attention (Attention Blocks)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#model` `#blocks` `#attention`

## 개요
Enhancer 모델의 핵심을 구성하는 어텐션 및 MLP 관련 블록 정의.

## 주요 로직

### `MultiLayerPerceptron` (MLP)
LayerNorm, Linear, GELU, Dropout을 조합한 피드포워드 네트워크. 특징 차원을 `mlp_ratio`만큼 확장했다가 복구.

### `SelfAttention` 및 `SelfAttentionBlock`
PyTorch의 `nn.MultiheadAttention`을 `query=x`, `key=x`, `value=x` 형태로 사용하는 Self-Attention 구현.
`SelfAttentionBlock`은 잔차 연결(Residual Connection)과 함께 Attention과 MLP를 순차적으로 결합.

### `CrossAttention` 및 `CrossAttentionBlock`
`nn.MultiheadAttention`을 `query=x`, `key=c`, `value=c` 형태로 사용하여 조건(condition) 정보를 주입하는 Cross-Attention 구현.
`CrossAttentionBlock`은 잔차 연결을 포함하여 Attention 결과에 MLP를 적용.

## 특이사항
- 각 블록 진입 전 `LayerNorm`을 적용하는 Pre-Norm 방식을 채택.

## 연결
- [[.wiki/domain/model-blocks-enhancer]] — 이 어텐션 블록들을 활용하여 Encoder/Decoder를 구성.

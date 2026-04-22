# model-blocks-enhancer (Core Architecture)

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: domain
**태그**: `#model` `#blocks` `#enhancer`

## 개요
이미지 개선(Low-light Enhancement)을 위한 주 네트워크인 `Enhancer` 클래스와, 이를 구성하는 `Encoder`, `Decoder` 모듈.

## 주요 로직

### `Encoder`
입력 이미지를 채널별로 분리한 후 `SeparationBlock`을 통해 조명 성분(il)과 반사 성분(re)을 추출.
이후 각각에 Convolution을 거치고 Flatten하여 `SelfAttentionBlock`을 적용.
최종적으로 조명 성분과 반사 성분을 `CrossAttentionBlock`으로 결합(`cttn`)한 뒤, 원래의 공간 차원으로 복원(Unflatten)하고 Convolution하여 조건 피처맵(`c`)을 생성.

### `Decoder`
U-Net과 유사한 인코더-디코더 병목(Bottleneck) 구조를 가짐. 하지만 Convolution이 아닌 `CrossAttentionBlock`을 주 구성요소로 사용.
- **Down 단계**: `Downsampling`을 수행하며 특징 차원을 키우고 공간 해상도를 줄임. 동시에 스킵 커넥션을 위해 중간 출력을 리스트(`x_lst`, `c_lst`)에 저장.
- **Mid 단계**: 최하위 해상도에서 `CrossAttentionBlock`을 반복 수행.
- **Up 단계**: `Upsampling`으로 해상도를 키우고 Down 단계의 특징맵과 결합(`fusion`) 후 `CrossAttentionBlock` 적용.
- 최종 출력은 원본 입력 이미지와의 Global Residual Connection(`+ x_res * c_res`)을 통해 생성.

### `Enhancer`
`Encoder`와 `Decoder`를 감싸는 최상위 모듈. 입력 `x`를 `Encoder`에 통과시켜 조건 정보 `c`를 추출하고, 다시 `x`와 `c`를 `Decoder`에 전달하여 개선된 이미지를 출력.

## 특이사항
- 입력 데이터의 텐서 형태 변환(Flatten/Unflatten)이 어텐션 적용 전후에 수시로 발생함.

## 연결
- [[.wiki/domain/model-blocks-separation]] — 분리 블록 의존.
- [[.wiki/domain/model-blocks-attention]] — 어텐션 블록 의존.
- [[.wiki/domain/model-blocks-flatten]] — 텐서 차원 변형 의존.
- [[.wiki/domain/model-blocks-sampling]] — 업/다운 샘플링 블록 의존.

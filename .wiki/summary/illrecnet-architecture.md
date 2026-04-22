# IllRecNet Architecture

> **관련 문서**: [[.wiki/concept/model-module]] | [[.wiki/domain/model-blocks-separation]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: summary
**출처**: 프로젝트 아키텍처 및 요구사항 리뷰
**태그**: `#summary` `#architecture`

## 맥락
IllRecNet 프로젝트의 전체적인 모델 아키텍처 구조 및 흐름을 이해하기 위해 작성된 아키텍처 요약입니다.

## 핵심 내용
IllRecNet은 저조도 이미지 개선(low-light image enhancement)을 위한 딥러닝 네트워크입니다. 핵심 동작 원리는 다음과 같습니다:
- **분리 블록(Separation Block)**: 입력 이미지를 조명(illumination) 성분과 반사(reflection) 성분으로 분리합니다.
- **특징 처리(Feature Processing)**: 분리된 각 성분들을 멀티 헤드 어텐션(multi-head attention) 및 공간적 샘플링(spatial sampling) 기법을 사용해 고도화된 특징으로 처리합니다.
- **손실 함수(Loss Functions)**: 모델은 개선된 결과물과 고조도 타겟 이미지 간의 차이를 최소화하기 위해 MAE (L1 Loss) 및 MSE (L2 Loss)를 혼합하여 손실을 계산하고 학습합니다.

## 연결
- [[.wiki/concept/model-module]] — IllRecNet 메인 모델 모듈에 대한 전반적 설명
- [[.wiki/domain/model-blocks-separation]] — 조명과 반사 성분으로 분리하는 Separation 블록에 대한 세부 구조

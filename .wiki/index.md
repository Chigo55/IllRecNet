# Wiki Index

> 전체 위키 카탈로그. 카테고리별 분류, 한줄 요약 포함. [갱신: 2026-04-22]

## Concept
| 페이지 | 요약 | 갱신 |
|--------|------|------|
| [[.wiki/concept/core-factory]] | 레지스트리 기반 모델/Loss/옵티마이저 동적 생성 팩토리 | 2026-04-22 |
| [[.wiki/concept/data-module]] | PyTorch Lightning 기반 데이터 파이프라인 (Dataset, DataModule) | 2026-04-22 |
| [[.wiki/concept/engine-module]] | 모델 학습, 검증, 테스트 및 추론 실행을 캡슐화한 래퍼 (Trainer 래퍼) | 2026-04-22 |
| [[.wiki/concept/model-module]] | PyTorch Lightning 기반의 모델 래퍼, 손실 함수 및 하위 블록 구성 | 2026-04-22 |
| [[.wiki/concept/utils-module]] | pyiqa 기반의 이미지 품질 지표 측정 및 각종 헬퍼 함수 모음 | 2026-04-22 |
| [[.wiki/concept/entry-points]] | 프로젝트 루트의 학습, 추론, 평가 등을 실행하는 진입점 스크립트 | 2026-04-22 |

## Pattern
| 페이지 | 요약 | 갱신 |
|--------|------|------|
| [[.wiki/pattern/registry-factory]] | 구성 딕셔너리 기반 클래스 동적 생성을 위한 Registry 및 Factory 패턴 | 2026-04-22 |
| [[.wiki/pattern/lightning-pipeline]] | PyTorch Lightning과 Runner 래퍼를 통한 파이프라인 보일러플레이트 제거 | 2026-04-22 |

## Domain
| 페이지 | 요약 | 갱신 |
|--------|------|------|
| [[.wiki/domain/01-lolv1-train]] | LOLv1 데이터셋 모델 학습 스크립트 | 2025-02-18 |
| [[.wiki/domain/02-lolv2real-train]] | LOLv2-real 데이터셋 모델 학습 스크립트 | 2025-02-18 |
| [[.wiki/domain/03-lolv2synthetic-train]] | LOLv2-synthetic 데이터셋 모델 학습 스크립트 | 2025-02-18 |
| [[.wiki/domain/bench]] | 학습 모델의 성능 벤치마크 평가 스크립트 | 2025-02-18 |
| [[.wiki/domain/infer]] | 다양한 사이즈 모델을 이용한 이미지 추론 스크립트 | 2025-02-18 |
| [[.wiki/domain/valid]] | 모델의 검증 세트 성능 확인 스크립트 | 2025-02-18 |
| [[.wiki/domain/data-dataloader]] | PyTorch Lightning 기반 통합 데이터로더 모듈 | 2025-02-18 |
| [[.wiki/domain/data-utils]] | 이미지 로드 및 데이터 증강을 수행하는 Dataset 클래스 | 2025-02-18 |
| [[.wiki/domain/engine-engine]] | 학습/검증/추론 프로세스를 초기화하고 실행하는 엔진 래퍼 | 2025-02-18 |
| [[.wiki/domain/engine-runner]] | Trainer 기반의 구체적 실행 로직을 담당하는 Runner 클래스 모음 | 2025-02-18 |
| [[.wiki/domain/model-model]] | Enhancer 네트워크와 손실 함수, 평가 지표를 포함한 Lightning 모듈 | 2025-02-18 |
| [[.wiki/domain/model-loss]] | 모델 학습을 위한 L1, MSE 기반 래퍼 손실 함수들 | 2025-02-18 |
| [[.wiki/domain/model-blocks-attention]] | 모델 구성에 사용되는 Multihead 기반 어텐션 및 MLP 블록 | 2025-02-18 |
| [[.wiki/domain/model-blocks-enhancer]] | 개선 모델의 메인 아키텍처(Encoder, Decoder) 정의 블록 | 2025-02-18 |
| [[.wiki/domain/model-blocks-flatten]] | 4D 텐서를 3D 시퀀스 텐서로 변환, 복원하는 헬퍼 함수 | 2025-02-18 |
| [[.wiki/domain/model-blocks-sampling]] | 피처맵의 공간 해상도를 조절하는 업샘플링/다운샘플링 모듈 | 2025-02-18 |
| [[.wiki/domain/model-blocks-separation]] | 이미지를 조명(il)과 반사(re) 성분으로 분리하는 블록 | 2025-02-18 |
| [[.wiki/domain/utils-metrics]] | pyiqa를 활용한 다양한 이미지 화질 평가 지표 래퍼 | 2025-02-18 |
| [[.wiki/domain/utils-utils]] | 파라미터 계산, 이미지 저장, 모델 요약 등 다목적 유틸리티 | 2025-02-18 |

## MOC
| 페이지 | 요약 | 갱신 |
|--------|------|------|
| [[.wiki/moc/illrecnet-overview]] | The central hub for navigating the IllRecNet low-light image enhancement project | 2026-04-22 |

## Summary
| 페이지 | 요약 | 갱신 |
|--------|------|------|
| [[.wiki/summary/illrecnet-architecture]] | IllRecNet 아키텍처 개요 (분리 블록, 어텐션, 손실 함수 등) | 2026-04-22 |

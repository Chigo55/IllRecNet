# Registry Factory

> **관련 문서**: [[.wiki/concept/core-factory]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: pattern
**태그**: `#registry` `#factory`

## 문제
하드코딩된 클래스 인스턴스화(Hardcoding class instantiations)는 코드를 경직되게 만들고 다양한 모델이나 손실 함수(Losses)로 실험하기 어렵게 만듭니다.

## 해법
Registry 패턴을 사용하여 클래스(Models, Losses, Optimizers)를 등록하고, Factory 클래스(`ModelFactory`, `LossFactory`)를 사용하여 구성(Configuration) 딕셔너리를 통해 동적으로 인스턴스화합니다.

## 적용 사례
- 모델, 손실 함수, 옵티마이저 등 여러 컴포넌트를 구성 파일의 변경만으로 동적으로 갈아끼우며 실험해야 하는 머신러닝 파이프라인.

## 연결
- [[.wiki/concept/core-factory]] — 레지스트리 기반 모델/Loss/옵티마이저 동적 생성 팩토리 구현 상세

# IllRecNet Overview MOC

> **관련 문서**: [[.wiki/index]]

**생성**: 2026-04-22  **갱신**: 2026-04-22
**분류**: moc
**태그**: `#moc` `#illrecnet`

## 개요
The central hub for navigating the IllRecNet low-light image enhancement project. It organizes knowledge by architecture, core modules, and detailed model blocks.

## 페이지 목록

### Architecture & Patterns
- [[.wiki/summary/illrecnet-architecture]] — Overall project architecture and illumination/reflection separation strategy
- [[.wiki/pattern/lightning-pipeline]] — PyTorch Lightning and Trainer wrapper pipeline
- [[.wiki/pattern/registry-factory]] — Dynamic instantiation using Registry and Factory

### Core Modules (Concepts)
- [[.wiki/concept/entry-points]] — Training, inference, and benchmarking scripts
- [[.wiki/concept/core-factory]] — Registry-based dynamic generation factory
- [[.wiki/concept/data-module]] — Lightning DataModule and Dataset pipeline
- [[.wiki/concept/engine-module]] — Trainer wrapper for execution
- [[.wiki/concept/model-module]] — Main LightningModule and loss functions
- [[.wiki/concept/utils-module]] — Image quality metrics (pyiqa) and helpers

### Key Model Blocks (Domains)
- [[.wiki/domain/model-blocks-enhancer]] — Main Enhancer network architecture
- [[.wiki/domain/model-blocks-separation]] — Illumination and reflection separation block
- [[.wiki/domain/model-blocks-attention]] — Multi-head attention blocks

## 연결
- [[.wiki/index]] — 전체 카탈로그

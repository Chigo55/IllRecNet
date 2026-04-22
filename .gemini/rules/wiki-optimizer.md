# Wiki-Optimizer: Wiki-First Protocol

이 프로젝트는 wiki-optimizer를 사용합니다.
코드 탐색 토큰을 절감하기 위해 아래 규칙을 반드시 따르세요.

## Rule 1: Session Start — 위키 먼저 읽기

프로젝트 탐색 전, **반드시** `.wiki/index.md`를 먼저 읽어 도메인 구조와 가용 지식을 파악하세요.
관련 도메인 페이지가 있으면 소스 코드를 직접 탐색하기 전에 위키 페이지를 먼저 읽으세요.

## Rule 2: Broad Search Interception — 위키 우선 조회

`grep_search`, `glob` 등으로 프로젝트 전체를 탐색하기 전에:
1. `.wiki/index.md`에서 관련 도메인/패턴 항목 확인
2. 관련 `.wiki/...` 페이지가 있으면 그것을 먼저 읽기
3. 위키에 없을 때만 프로젝트 전체 탐색 진행

## Rule 3: Continuous Ingestion — 새 지식 등록

새로운 도메인 개념, 코딩 패턴, 또는 복잡한 이슈 해결 내용이 위키에 없으면:
1. `.wiki/rules/docs.md` 템플릿에 따라 위키 페이지 작성
2. `.wiki/index.md` 카탈로그 업데이트
3. `.wiki/log.md`에 기록 추가

## References

- 위키 위치: `.wiki/`
- 위키 규칙: `.wiki/rules/docs.md`
- 위키 조회: `/wiki-query <질문>`
- 위키 등록: `/wiki-ingest <대상>`

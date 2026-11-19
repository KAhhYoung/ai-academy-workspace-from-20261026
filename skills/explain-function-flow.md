---
name: explain-function-flow
description: Explain what a code entry point or function is trying to accomplish in a Markdown table organized by function and execution order. Use for requests such as “무슨 작업을 하려고 하는지 함수단위로 설명해줘”.
---

# 함수별 실행 흐름 설명

- 답변 첫 문단에서 전체 목적을 한두 문장으로 설명한다. 진입점이 다른 파일의 함수를 가져와 실행한다면 실제 구현을 찾아 그 함수의 흐름을 설명한다.
- 관련 함수의 구현을 읽고, 실제 호출 순서대로 Markdown 표를 작성한다. 기본 열은 아래와 같다.

| 호출 순서 | 함수 | 하는 일 |
|---|---|---|

- 중첩 호출은 필요에 따라 `audit() → check_meta()`처럼 표시한다. 각 행에는 함수가 처리하는 입력, 조건, 반환값 또는 저장 결과 중 흐름을 이해하는 데 필요한 내용을 짧게 설명한다. 조건부 호출은 실행 조건을 함께 적는다.
- 함수가 실제로 생성하는 것을 구체적으로 표현한다. 예를 들어 본문 없이 ID·버전·위치만 담는 `make()`는 샘플 청크 메타데이터를 생성한다고 설명한다.
- 예제의 최종 결과를 코드에서 확인할 수 있다면 표 아래에 짧게 적고, 주요 구현 위치에 파일 링크를 제공한다.

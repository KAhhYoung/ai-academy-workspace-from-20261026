---
name: write-markdown-exercise-reports
description: Write Korean Markdown reports for coding exercises, tests, and experiment results using the user's report format. Apply when the user asks for a report about a coding exercise or test results, such as “보고서 작성해줘”, or explicitly requests this format.
---

# Markdown 실습 보고서 작성

작성 전에 [참고 보고서](references/rev_test_plan.md)를 읽고 제목, 환경 구성, 테스트 결과 표, 최종 요약의 형식을 참고한다. 현재 실습의 코드와 결과 파일을 확인해 다음 순서로 작성한다.

1. 보고서 제목에 실습명과 검증 대상을 넣는다.
2. **1. 환경 구성**에는 확인된 실습일, 실행 대상, 환경, 주요 도구나 데이터 조건을 기록한다. 실습자 이름은 현재 요청에서 확인된 경우에만 기재한다.
3. 테스트 결과를 이해하는 데 필요한 입력 데이터나 기준 설정이 있으면 표 또는 목록으로 먼저 보여 준다.
4. **2. 테스트 조건 및 결과**에는 실행 방식, 결과 파일, 주요 테스트 조건을 정리한다. 사례별로 번호, 상황, 입력 또는 조건, 실제 결과, 결과 해석을 표에 담는다. 기대값과 실제 결과를 함께 다루는 경우 각각의 열을 둔다. 검색 옵션, 거리, 출처 버전 같은 열은 실습 주제에 맞게 사용한다.
5. **최종 요약**에는 확인한 결과와 조건이 결과에 미친 영향을 짧게 정리한다. 실행 기록이 없는 사례는 미실행 또는 미확인으로 표시한다.

참고 파일의 날짜, 사람 이름, 경로, 데이터, 측정값은 예시다. 현재 실습에서 확보한 정보로 내용을 작성한다. 저장 경로를 지정받았다면 그 위치에 Markdown 파일을 만든다.

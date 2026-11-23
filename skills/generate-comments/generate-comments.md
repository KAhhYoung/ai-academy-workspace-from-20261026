---
name: generate-comments
description: references/tasks.py 양식에 맞춰 Python 소스에 한국어 주석을 추가하거나 다듬는다. 코드에 주석을 달아 달라는 요청에 사용한다.
---

# Python 주석 생성

[references/tasks.py](references/tasks.py)를 주석 양식의 기준으로 읽는다. 대상 코드와 필요한 호출부를 확인한 뒤, 요청된 범위에 실제 동작을 설명하는 한국어 주석을 작성한다.

- import 바로 위에 가져오는 기능의 목적을 적는다. 심볼별 역할이 필요하면 `# - 이름 : 역할` 형식으로 덧붙인다.
- 함수 정의 바로 위에 `# ... 함수`로 역할을 요약한다. 다음 줄부터 `# - ...` 형식으로 핵심 처리, 검증 조건, 반환 결과를 필요한 만큼 적는다.
- 함수 내부에서는 관련 문장을 한 처리 단계로 묶고, 해당 블록 바로 위에 같은 들여쓰기로 짧은 명사형 주석을 단다. 분기 조건, 검증 실패 시 동작, 반환값처럼 흐름 이해에 필요한 정보를 포함한다.
- 기존 `BEGIN`, `END`, `TODO`, 요구, 힌트 등의 실습 주석은 보존한다. 수정 범위는 요청된 설명 주석으로 한정하고, diff에서 실행 코드와 기존 docstring이 보존됐는지 확인한다.

```python
# JSON 입출력 기능 사용
# - load_json : JSON 파일 읽기
from coursekit.io import load_json

# 매니페스트 로드 및 검증 함수
# - 빈 목록 검사 후 문서 목록 반환
def read_manifest(path):
    # 매니페스트 JSON 데이터 로드
    rows = load_json(path)
    # 매니페스트 빈 목록 여부 검사
    if not rows:
        raise ValueError("empty manifest")
    return rows
```

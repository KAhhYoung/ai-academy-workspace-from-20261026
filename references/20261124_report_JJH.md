# Day27 — RAG 검색 개선
# by.정준하님

---

## 1. Rewrite / Reranking

### [파이프라인]

사용자 → FastAPI → [Rewrite + 검색 조건 설정] → Retrieval → [Reranking] → Context → Prompt → LLM → Validation → 응답

### [목적]

문서 검색의 정확도를 높이기 위함

- **Rewrite**: 모호한 질문을 명확하게 재구성하는 방식(Retrieval 단계에서 정확한 근거 자료를 검색할 수 있도록)
  - 질문이 명확한 경우: History(사용자의 과거 대화이력) 없이도 검색 가능.
  - 질문이 모호하지만 History에서 대상을 확인할 수 있는 경우: Rewrite를 통해 질문을 재구성.
  - 질문이 모호하고 History에서도 대상을 확인할 수 없는 경우: 임의로 재구성하지 않고 사용자에게 추가 확인 요청.
  - Rewrite 단계에서도 LLM과 Validation은 사용될 수 있다.

- **Reranking**: 사용자의 검색 질문을 기준으로 수집한 후보 문서의 관련성을 다시 평가하고 재정렬한다.

---

## 2. Query Expansion / Multi Query

### [파이프라인]

... → Rewrite + 검색 조건 설정 → Query Expansion / Multi Query → Retrieval → ...

- 실습 함수 반영: Multi Query → `unique_queries()` → Retrieval → `merge_found()` → Reranking → Context

### [목적]

Retrieval 단계에서 필요한 근거 자료를 놓치지 않고 검색하기 위한 방법

- **Query Expansion**: 기존 질문 + 관련 검색어 및 보충 단어 추가 = 질문 1개 유지(이번 수업에서의 구분)
- **Multi Query**: 기존 질문 + 여러 개의 검색을 위한 추가 질문 = 질문 여러 개 생성

### [코드 분석(coursekit)]

- **`unique_queries()` — 검색용 질문 정리**
  - 실행 시점: Retrieval 이전
  - 목적: 불필요한 검색 방지
  - 처리: 빈 질문 제거, 중복 질문 제거, 검색 질문 개수 제한
  - 반환: 정리된 검색용 질문 목록

- **`merge_found()` — 검색된 근거 정리**
  - 실행 시점: Retrieval 이후, Reranking 이전
  - 목적: 중복된 근거 제거 및 발견 경로 관리
  - 처리: 검색 결과 통합, chunk_id 기준 중복 제거, 어떤 질문으로 발견했는지 기록
  - 반환: 통합된 근거 자료 목록

---

## 3. Candidate Retrieval

### [단계]

이름에서 알 수 있듯이 Retrieval 단계.

### [목적]

후보 문서를 넉넉하게 확보하여 이후 Reranking에서 더 적합한 문서를 선정할 수 있도록 함.

- `candidate_k`: 1차 검색에서 검색 질문당 확보할 최대 문서 수(5개)
- `context_k`: Context에 포함할 최대 문서 수(2개)

---

## 4. Cross Encoder

### [단계]

Reranking 단계에서 활용

### [목적]

답변에 더 적합한 문서를 선택하여 Context의 품질을 높이는 것

- 질문과 각 후보 문서의 본문을 하나의 쌍으로 입력받아 관련성 점수를 계산하는 모델이다.
- 해당 점수를 기준으로 후보 문서를 재정렬한다.

---

## 5. LangChain RAG 파이프라인 구성

### 5-1. 파이프라인 준비 및 실행 흐름

**입력 준비 → 기능 준비 → 기능 연결 → 실제 실행 → 결과 저장**

#### ① 입력 준비 (`prepare_pipeline_request`)

원본 질문, History, 검색 조건, 각 기능의 사용 여부 설정

- `candidate_k`: 검색 질문당 최대 후보 문서 수
- `context_k`: Context에 포함할 최대 문서 수
- Rewrite, Multi Query, Reranking 사용 여부 설정

#### ② 기능 준비 (`create_dependencies`)

각 단계에서 사용할 모델과 함수 준비

#### ③ 기능 연결 (`create_rag_pipeline`)

LangChain의 `|` 문법을 이용하여 각 기능을 실행 순서대로 연결

#### ④ 실제 실행 (`chain.invoke`)

준비한 입력값을 전달하여 파이프라인 실행

#### ⑤ 결과 저장 (`save_report`)

답변, 출처, 처리 이력을 JSON 파일로 저장


### 5-2. 파이프라인 내부의 데이터 처리 흐름

**입력 → Rewrite → Multi Query → Retrieval → Reranking → Context → Prompt → Generation → Validation → 응답**

1. **입력**: 원본 질문, History, 검색 조건 전달 및 검증
2. **Rewrite**: 검색에 적합한 독립적인 질문 생성
3. **Multi Query**: 추가 검색 질문 생성 및 중복 질문 제거
4. **Retrieval**: 관련 후보 문서 검색 및 중복 문서 제거
5. **Reranking**: 후보 문서의 관련성을 정밀하게 평가하여 순위 재조정
6. **Context**: 최종 문서 선택 및 근거 자료 구성
7. **Prompt**: 답변 규칙 + 질문 + Context 결합
8. **Generation**: LLM을 이용하여 답변 생성
9. **Validation**: 답변 형식 및 출처 검증
10. **응답**: 답변, 출처, 처리 이력 반환

- Rewrite, Multi Query, Reranking은 선택적으로 사용 가능
- 각 단계의 처리 결과는 하나의 딕셔너리에 누적되어 다음 단계로 전달

### [두 흐름의 차이]

- **5-1**: 파이프라인을 어떻게 준비하고 작동시킬 것인가?
- **5-2**: 작동된 파이프라인이 질문을 어떻게 처리할 것인가?

> 5-1의 ④ 실제 실행(`chain.invoke`)에서 5-2의 흐름이 시작된다.

---
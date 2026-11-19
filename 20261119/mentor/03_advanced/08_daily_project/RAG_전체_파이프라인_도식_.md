# 실무 RAG 파이프라인 전체 지도


## 1. 전체 도식

```mermaid
flowchart TB
    subgraph PREP["문서 수집 · 전처리 · 인덱싱 파이프라인 (RAG 데이터 준비)"]
        direction LR
        C1["1. Collect<br/>문서 수집 · 원본 확보<br/>PDF / HTML / DB 등"]
        C2["2. Parse<br/>텍스트 추출 · 구조화<br/>문서 → 텍스트"]
        C3["3. Clean<br/>텍스트 정제<br/>중복 · 불필요한 내용 제거"]
        C4["4. Chunk<br/>청크 분할<br/>검색 단위 생성"]
        C5["5. Validate<br/>데이터 검증<br/>형식 · 길이 · 메타데이터"]
        C6["6. Embed<br/>임베딩 생성<br/>문장/청크 → 벡터"]
        C7["7. Publish<br/>인덱싱 · 저장<br/>pgvector에 저장<br/>(원문 ID · 메타데이터 포함)"]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    end

    DB[("PostgreSQL<br/>원문 · Metadata")]
    VDB[("pgvector<br/>청크 · 벡터 · 출처")]
    C7 --> VDB
    C7 -. "원문·메타데이터 관리" .-> DB

    subgraph SERVE["사용자 요청 · 응답 생성 (애플리케이션 · Backend)"]
        direction LR
        U["사용자<br/>질문 · 요청"]
        API["FastAPI<br/>요청 · 응답"]
        CTX["Context 구성<br/>검색 자료 선택"]
        PR["Prompt 조립<br/>지시 · 근거 · 계약"]
        LLM["LLM Client<br/>응답 생성 호출"]
        OUT["출력 검사<br/>Schema · ID · 근거"]
        EVI["근거 검토<br/>원문 사실 확인"]
        U -->|질문| API
        API -->|질문| CTX
        CTX -->|문맥| PR
        PR -->|입력| LLM
        LLM -->|응답| OUT
        OUT -->|자동 통과| EVI
        EVI -->|검토 결과 반환| API
        API -->|응답| U
    end

    R["8. Retriever<br/>Top-k · 원문 후보"]
    VDB -->|벡터 유사도 검색| R
    DB -->|메타데이터| CTX

    STOP["실패 · 중단<br/>오류 구분 · 기록"]
    EVAL["검색 결과 평가<br/>원문 관련성 · 이유"]
    LLM -. "통신 오류" .-> STOP
    OUT -. "형식 오류" .-> STOP
    PR -. "요청 조건" .-> EVAL
    R -->|후보 평가| EVAL
    EVAL -->|검색한 근거| CTX

    classDef prep fill:#f7f0ff,stroke:#8c48c8,color:#22132f,stroke-width:1px
    classDef retrieve fill:#214f80,stroke:#a84b1f,color:#ffffff,stroke-width:2px
    classDef central fill:#214f80,stroke:#a84b1f,color:#ffffff,stroke-width:2px
    classDef storage fill:#eef4ff,stroke:#5781b3,color:#132b48
    classDef error fill:#fff5ec,stroke:#bc6c34,color:#552600
    class C1,C2,C3,C4,C5,C6,C7 prep
    class R retrieve
    class CTX central
    class DB,VDB storage
    class STOP error
```

## 2. 문서 준비와 질문 처리의 연결

```mermaid
flowchart LR
    A["문서 수집"] --> B["Parse"] --> C["Clean"] --> D["Chunk"] --> E["Validate"] --> F["Embed"] --> G["Publish"]
    G --> H[("PostgreSQL · pgvector")]
    Q["사용자 질문"] --> J["8. Retriever"]
    H --> J
    J --> K["Context 구성"] --> L["Prompt 조립"] --> M["LLM Client"] --> N["출력 검사"] --> O["근거 검토"] --> P["사용자 응답"]
```

## 3. 각 단계의 위치와 역할

> **1~7단계**는 문서 수집·전처리·인덱싱(데이터 준비)이고, **8단계**는 질문 시점의 검색(Retrieval)입니다.  


| No. | 단계 | 주요 역할 |
|:---:|---|---|
| 1 | Collect | PDF·HTML·DB 등의 원본 수집 |
| 2 | Parse | 파일에서 텍스트를 추출하고 구조화 |
| 3 | Clean | 중복·불필요한 내용 정제 |
| 4 | Chunk | 검색 가능한 크기로 텍스트 분할 |
| 5 | Validate | 청크의 형식·길이·메타데이터 검증 |
| 6 | Embed | 청크를 임베딩 벡터로 변환 |
| 7 | Publish | 청크·벡터·원문 ID·메타데이터 저장 및 검색 가능 상태로 등록 |
| 8 | Retriever | 저장된 데이터에서 관련 원문 후보를 Top-k로 검색 |
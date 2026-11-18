# Day23 / 실습 06 / 심화 index_sql / mentor
from day23kit.db import HNSW_SQL


# BEGIN D23-06-A01
def create_index_sql():
    """
    [강사용 정답] D23-06-A01 · create_index_sql
        day23_search_docs.embedding의 Cosine HNSW 인덱스를 정의하세요.
        USING hnsw (embedding vector_cosine_ops)를 사용합니다.
        인덱스는 선택 항목이며 9건에서 빠른 성능을 보장하지 않습니다.
    """
    # 해설 D23-06-A01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return HNSW_SQL
# END D23-06-A01

# BEGIN D23-06-A02
def explain_sql():
    """
    [강사용 정답] D23-06-A02 · explain_sql
        SELECT id FROM day23_search_docs를 거리순으로 검색하는
        EXPLAIN (ANALYZE, BUFFERS) SQL을 작성하세요. 바인딩은 %s 두 개.
        인덱스 사용 여부를 임의로 주장하면 안 됩니다.
    """
    # 해설 D23-06-A02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # --------------------------------------------------------
    # 이 SQL은 질문 벡터와 코사인 거리가 가까운 문서를 최대 3개 찾는 쿼리를 실제로 실행하고, PostgreSQL이 그 검색을 어떻게 수행했는지 보여줍니다. 출력에는 문서 ID 목록 대신 실행 계획이 표시됩니다.
    # | 구문 | 의미 |
    # |-----|------|
    # | SELECT id FROM day23_search_docs | 검색 결과에서 문서 ID를 선택합니다. |
    # | WHERE embedding_model = %s | 첫 번째 %s에 전달한 모델로 만든 문서 임베딩만 검색합니다. |
    # | ORDER BY embedding <=> %s ASC | 두 번째 %s에 전달한 질문 벡터와 코사인 거리를 계산하고, 가까운 순서로 정렬합니다. |
    # | LIMIT 3 | 최대 3개만 선택합니다. |

    # ANALYZE는 쿼리를 실제로 실행해 실제 처리 시간과 행 수를 보여줍니다. BUFFERS는 PostgreSQL 버퍼에서 찾은 블록(shared hit)과 추가로 읽어 온 블록(shared read)을 보여줍니다. PostgreSQL EXPLAIN 옵션
    # 앞에서 만든 HNSW 인덱스를 사용했는지 보려면 실행 계획에서 **Index Scan using day23_search_docs_hnsw**를 찾으세요. **Seq Scan과 Sort**가 보이면 테이블을 순차적으로 읽고 거리를 기준으로 정렬한 것입니다.
    # --------------------------------------------------------    
    return """EXPLAIN (ANALYZE, BUFFERS)
    SELECT id FROM day23_search_docs
    WHERE embedding_model=%s
    ORDER BY embedding <=> %s ASC LIMIT 3"""
# END D23-06-A02

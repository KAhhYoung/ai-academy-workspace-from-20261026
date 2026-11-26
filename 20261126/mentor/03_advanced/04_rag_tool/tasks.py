"""04 · 기존 RAG를 rag_search Tool로 · 심화 · 강사용 정답.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.
"""

'''
# wrapping 작업: 들어오게 되는 파라미터가 주민번호 등으로 늘어날 수 있고
# prepare_search 안의 validator 호출해서(pydantic 으로도 처리 가능) wrapper 마다 다르게 로직을 처리할 수 있음
# serach_request 는 단지 기능함수이고, 받는 파라미터 또한 기본값을 None 으로 처리하여 여러 개로 함수의 호출을 받을 수 있음
# @tool 붙게 되는 함수는 search_request 가 아니라 prepare_search / @tool 별로 권한 따로 부여
# 여기서 tool 에 대한 정보도 로그로 남기고
# 여기서 tool 에 대한 status 도 반환함
'''
def prepare_search(query: str, category: str = 'ALL') -> dict:
    """실제 6개 카테고리만 허용하고 query/category를 반환."""
    # 해설 D29-04-A-01: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    from coursekit.contracts import search_request
    # search_request 에서 하는 작업: query, category 검증
    query,category=search_request(query,category)
    return {'query':query,'category':category}

def invoke_rag(request: dict) -> dict:
    """실제 기존 DB의 pgvector/CUDA RAG를 호출합니다. 가짜 검색 결과 금지."""
    # 해설 D29-04-A-02: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    from coursekit.rag import rag_answer
    return rag_answer(request['query'],request['category'])


# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_04 import format_evidence, validate_evidence

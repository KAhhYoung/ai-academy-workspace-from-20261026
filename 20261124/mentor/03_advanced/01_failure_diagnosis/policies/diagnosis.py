"""단계별 역할 분리: policies/diagnosis.py / 독립 실행은 main.py. """

# 사례의 첫 실패 단계 판정 함수
# - 사례 딕셔너리의 조건을 순서대로 검사하고 처음 해당하는 실패 단계 반환
def diagnose_stage(case):
    """어떤 문제가 먼저 발생했는지 위에서부터 조건을 검사합니다."""
    # BEGIN D27-01-A01
    # ANSWER D27-01-A01: 어떤 문제가 먼저 발생했는지 위에서부터 조건을 검사합니다.
    # 1. 입력: case (필요한 값만 받습니다.)
    # 2. 절차: 어떤 문제가 먼저 발생했는지 위에서부터 조건을 검사합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 접근 권한, 현재 유효 버전, 인덱스 준비 상태 순으로 검사
    if not case['authorized']:return 'authorization_filter'
    if not case['current']:return 'version_filter'
    if not case['ready']:return 'index_not_ready'
    # 질문이 모호하면 재작성 단계로 판정
    if case['ambiguous']:return 'rewrite'
    # 정답 근거의 후보 누락은 검색 단계, 최종 문맥 누락은 재순위화 단계로 판정
    if not case['gold_in_candidates']:return 'retrieval'
    if not case['gold_in_context']:return 'reranking'
    # 출처 검증 값이 False이면 검증 실패, 앞 조건을 모두 통과하면 정상으로 판정
    if case['valid_source'] is False:return 'validation'
    return 'healthy' 
    # END D27-01-A01


# 진단 단계별 개선 방법 조회 함수
# - 단계 이름에 대응하는 안내 문자열 반환
def recommend_action(stage):
    """문제 단계에 맞는 개선 방법을 반환합니다. Reranker는 누락된 후보를 새로 찾지 않습니다."""
    # BEGIN D27-01-A02
    # ANSWER D27-01-A02: 문제 단계에 맞는 개선 방법을 반환합니다. Reranker는 누락된 후보를 새로 찾지 않습니다.
    # 1. 입력: stage (필요한 값만 받습니다.)
    # 2. 절차: 문제 단계에 맞는 개선 방법을 반환합니다. Reranker는 누락된 후보를 새로 찾지 않습니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 진단 단계 이름을 키로 개선 방법 조회
    return {'authorization_filter':'서버에서 접근 권한을 검사합니다.',
            'version_filter':'현재 유효한 문서만 검색합니다.',
            'index_not_ready':'Indexing 공개 상태를 확인합니다.',
            'rewrite':'History에서 명확한 대상만 복원합니다.',
            'retrieval':'검색 대상·필터·Top-k를 점검합니다.',
            'reranking':'후보 문서의 관련성 순위를 다시 평가합니다.',
            'validation':'답변과 실제 근거를 대조합니다.',
            'healthy':'정상 흐름을 유지합니다.'}[stage]
    # END D27-01-A02

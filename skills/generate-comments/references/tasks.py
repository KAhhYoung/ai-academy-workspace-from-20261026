"""실습 08 Advanced · 오프라인 준비 + --live-db 실제 통합 실행."""
# 파일 경로 처리 기능 사용
# - Path : 파일 및 디렉터리 경로 관리
from pathlib import Path
# 파이썬 실행 환경 제어 기능 사용
# - sys : 모듈 검색 경로 및 명령행 인자 접근
import sys
# 현재 실습 파일 경로 조회
# - 결과 파일 저장 위치 계산
# - 상대 경로 계산 기준
HERE=Path(__file__).resolve().parent
# 상위 디렉터리 모듈 경로 등록
# - coursekit 패키지 import 가능
sys.path.insert(0,str(HERE.parents[1]))
# 입출력 및 완료 출력 기능 사용
# - DATA : 실습 데이터 디렉터리
# - load_json : JSON 파일 읽기
# - save_json : JSON 파일 저장
# - show_success : 실행 완료 메시지 출력
from coursekit.io import DATA,load_json,save_json,show_success
# 문서 파싱 및 마크다운 단위 분할 기능 사용
# - parse_one : 문서 형식별 파싱 수행
# - parse_markdown_units : 마크다운 본문 단위 분할 파싱
from coursekit.documents import parse_one,parse_markdown_units
# 텍스트 정제, 청킹 및 스팬 검증 기능 사용
# - make_chunks : 단위 본문 기반 청크 목록 생성
# - clean_prose : 일반 텍스트 정제
# - validate_span : 청크 문자 오프셋 및 본문 정합성 검증
from coursekit.processing import make_chunks,clean_prose,validate_span

"""매니페스트 로드 및 오프라인 전처리(파싱·정제·청킹·검증)부터 E5 임베딩, PostgreSQL 스테이징/공개, v2 버전 회귀 및 최종 리포트 생성까지 통합 수행하는 모듈"""

# 매니페스트 로드 및 검증 함수
# - manifest.json 파일 로드
# - 빈 목록 및 중복 doc_id 검사 후 doc_id 기준 정렬 반환
def read_manifest()->list[dict]:
    # BEGIN D24-08-A01
    # TODO D24-08-A01 : Manifest의 빈 목록·중복 doc_id를 검사하고, unsupported 문서도 포함해 ID순 목록을 반환합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 매니페스트 JSON 데이터 로드
    rows = load_json(DATA/'manifest.json')
    # 문서 ID 중복 여부 검사
    ids = [r['doc_id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('duplicate doc_id')
    # 매니페스트 빈 목록 여부 검사
    if len(rows) == 0: raise ValueError('empty manifest')
    # doc_id 기준 오름차순 정렬 반환
    return sorted(rows, key = lambda r : r['doc_id'])
    # raise NotImplementedError("D24-08-A01 TODO 미완성")
    # END D24-08-A01

# 전체 문서 오프라인 준비 함수
# - 문서별 파싱 및 상태 수집
# - 파싱 성공 문서 대상 정제·청킹·스팬 검증 수행
# - 정상 준비된 문서·청크 목록과 상태 목록 반환
def prepare_all(rows:list[dict])->tuple[list[dict],list[dict]]:
    # BEGIN D24-08-A02
    # TODO D24-08-A02 : Parsing·Cleaning·Chunking·Span 검사 후 정상 청크만 준비합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 준비 완료 문서 목록 및 파싱 상태 목록 초기화
    prepared = []
    statuses = []
    # 매니페스트 문서 순회 처리
    for doc in rows:
        # 문서 파싱 실행 및 상태 기록
        result = parse_one(doc)
        state = result['status']
        statuses.append({'doc_id' : doc['doc_id'], 'status' : state, 'reason' : result.get('reason')})
        # 파싱 완료('parsed') 상태가 아닌 문서는 건너뛰기
        if state != 'parsed': continue
        # 유닛별 정제, 청킹 및 문자 스팬 검증 수행
        chunks = []
        for unit in result['units']:
            clean = clean_prose(unit['raw_text'])
            for c in make_chunks(doc, unit, 110, 15):
                validate_span({'char_start' : c['char_start'], 'char_end' : c['char_end'], 'text' : c['body']}, clean)
                chunks.append(c)
        # 생성된 청크 부재 시 예외 발생
        if not chunks:raise ValueError(f'empty chunks: {doc["doc_id"]}')
        # 정상 준비 항목 추가
        prepared.append({'doc' : doc, 'chunks' : chunks})
    # 준비된 데이터 목록과 상태 목록 반환
    return prepared, statuses
    # raise NotImplementedError("D24-08-A02 TODO 미완성")
    # END D24-08-A02

# 문서 배치 임베딩 생성 함수
# - 사전 로드된 E5 모델을 재사용하여 문서별 청크 벡터 계산
# - 각 항목에 벡터 목록을 병합하여 반환
def embed_batches(prepared:list[dict],model)->list[dict]:
    # BEGIN D24-08-A03
    # TODO D24-08-A03 : main()이 한 번 로드한 E5 모델을 재사용해 문서별 passage 입력 길이·벡터 정규화를 검증합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 패시지 인코딩 함수 import 및 결과 리스트 초기화
    from coursekit.live import encode_passages
    result = []
    # 준비된 문서별 청크 임베딩 수행
    for item in prepared:
        vecs = encode_passages(model, item['chunks'])
        result.append({**item, 'vectors' : vecs})
    # 벡터가 결합된 결과 목록 반환
    return result
    # raise NotImplementedError("D24-08-A03 TODO 미완성")
    # END D24-08-A03

# 스테이징 및 공개 트랜잭션 실행 함수
# - 문서별로 분리된 트랜잭션에서 stage_version 및 publish_version 수행
# - 커밋 완료된 문서와 공개 버전 목록 반환
def save_and_publish(conn,embedded:list[dict])->list[dict]:
    # BEGIN D24-08-A04
    # TODO D24-08-A04 : 각 문서를 분리된 Transaction에서 Stage하고 완성본만 Publish합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 스테이징/공개 DB 함수 import 및 커밋 이력 리스트 초기화
    from coursekit.db import stage_version, publish_version
    committed = []
    # 임베딩 완료 문서별 개별 트랜잭션 처리
    for item in embedded:
        doc, chunks = item['doc'], item['chunks']
        with conn.transaction():
            stage_version(conn, doc, doc['file'], chunks, item['vectors'])
            publish_version(conn, doc['doc_id'], doc['source_version'], chunks[0]['pipeline_version'])
        # 공개 완료 문서 기록 추가
        committed.append({'doc_id' : doc['doc_id'], 'published_version' : doc['source_version']})
    # 커밋 완료 목록 반환
    return committed
    # raise NotImplementedError("D24-08-A04 TODO 미완성")
    # END D24-08-A04

# v2 버전 회귀 및 롤백 검증 함수
# - attendance_v2.md 파싱, 청킹 및 임베딩 생성
# - 강제 예외 발생 시 롤백으로 공개 포인터 유지 확인 후 재시도 공개 수행
def regression_v2(conn,model)->dict:
    # BEGIN D24-08-A05
    # TODO D24-08-A05 : v2에도 동일한 Markdown Parser를 적용하고, 실패·Rollback·정상 Publish 전후의 공개 포인터를 비교합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 관련 DB, 문서, 청킹 및 임베딩 모듈 import
    from coursekit.db import stage_version, publish_version
    from coursekit.documents import by_doc
    from coursekit.processing import make_chunks
    from coursekit.live import encode_passages
    # v2 마크다운 원문 로드 및 문서 메타데이터 구성
    src = (DATA/'versions'/'attendance_v2.md').read_text(encoding = 'utf-8')
    doc = {**by_doc('DOC-ATTEND'), 'source_version' : 'v2'}
    # v2 마크다운 본문 단위 파싱 및 청크 생성
    # - 제목/절 단위 분할 후 청크 크기 110, 중첩 15 기준으로 분할
    units = parse_markdown_units(src, 'attendance_v2.md')
    chunks = [chunk for unit in units for chunk in make_chunks(doc, unit, 110, 15)]
    # v2 청크 존재 여부 검사 및 패시지 벡터 계산
    if not chunks: raise ValueError('empty v2 chunks')
    vectors = encode_passages(model, chunks)
    # 현재 공개 버전 조회 헬퍼 정의 및 실패 전 포인터 기록
    pointer = lambda : conn.execute('SELECT current_version FROM ingest_documents WHERE doc_id=%s', (doc['doc_id'], )).fetchone()[0]
    before = pointer()
    # 실패 주입 트랜잭션 실행 및 롤백 검증
    try:
        with conn.transaction():
            stage_version(conn, doc, 'versions/attendance_v2.md', chunks, vectors)
            publish_version(conn, doc['doc_id'], 'v2', chunks[0]['pipeline_version'])
            raise RuntimeError('injected_failure_before_commit')
    # 의도된 실패 예외만 허용하고 그 외 예외는 재전파
    except RuntimeError as exc:
        if str(exc) != 'injected_failure_before_commit': raise
    # 롤백 후 공개 포인터 미변경 확인
    after_failure = pointer()
    if before != after_failure: raise ValueError('failed transaction changed public pointer')
    # 정상 트랜잭션으로 v2 스테이징 및 재공개 수행
    with conn.transaction():
        stage_version(conn, doc, 'versions/attendance_v2.md', chunks, vectors)
        publish_version(conn, doc['doc_id'], 'v2', chunks[0]['pipeline_version'])
    # 실패 전, 실패 후, 재시도 후 포인터 상태 반환
    return {'before' : before, 'after_failure' : after_failure, 'after_retry' : pointer()}
    # raise NotImplementedError("D24-08-A05 TODO 미완성")
    # END D24-08-A05

# 최종 통합 리포트 생성 함수
# - 상태별 건수 집계 및 입력 총량 일관성 검증
# - 오프라인 준비 요약과 선택적 Live 실행(검색·인용·로케이터) 결과 결합
def make_final_report(rows:list[dict],statuses:list[dict],prepared:list[dict],live:dict|None=None)->dict:
    # BEGIN D24-08-A06
    # TODO D24-08-A06 : 문서별 상태·청크 수를 기록하고, Live 실행 시에만 검색 결과·Citation ID·원문 Locator를 포함합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 힌트: 이 함수의 docstring과 상위 실행 함수에 전달되는 값을 확인합니다.
    # TODO를 완성한 뒤 아래 NotImplementedError 줄을 삭제합니다.

    # 문서 상태별 건수 집계 및 전체 문서 수 일치 확인
    counts = {s : sum(r['status'] == s for r in statuses) for s in sorted({r['status'] for r in statuses})}
    if sum(counts.values()) != len(rows): raise ValueError('input count inconsistent')
    # 전체 준비 청크 수 집계
    chunks = sum(len(r['chunks']) for r in prepared)
    # 최종 리포트 딕셔너리 구성 및 반환
    return {'source_mode' : 'e5_pgvector_live' if live is not None else 'prepared_only_not_embedded',
            'document_count' : len(rows), 'by_status' : counts, 'prepared_chunk_count' : chunks,
            'prepared_doc_ids' : [p['doc']['doc_id'] for p in prepared], 'live' : live if live is not None else 'not_run'}
    # raise NotImplementedError("D24-08-A06 TODO 미완성")
    # END D24-08-A06

# 프로그램 진입 함수
# - 매니페스트 읽기 및 오프라인 전처리 실행
# - --live-db 옵션 시 DB 스키마 초기화, 적재·공개, v2 회귀 및 벡터 검색 수행
# - ingestion_report.json 저장 및 완료 알림 출력
def main():
    # 매니페스트 로드 및 오프라인 문서 준비
    rows=read_manifest();prepared,statuses=prepare_all(rows)
    live=None
    # --live-db 플래그 전달 시 실제 DB 연동 및 검증 수행
    if '--live-db' in sys.argv:
        from coursekit.live import connect_db,load_encoder,encode_query
        from coursekit.db import initialize,search_current,use_schema
        # 모델 로드 및 배치 임베딩 생성
        model=load_encoder();embedded=embed_batches(prepared,model)
        # DB 연결, 스키마 전환 및 테이블 초기화
        with connect_db() as conn:
            use_schema(conn,'d24_adv08')
            initialize(conn)
            # 문서 스테이징/공개, v2 버전 회귀 및 질의 검색 실행
            committed=save_and_publish(conn,embedded)
            revision=regression_v2(conn,model)
            q=encode_query(model,'출석 기준을 알려 주세요')
            hits=search_current(conn,q,'학사',3)
            # 검색 결과 및 인용 메타데이터 구성
            live={'published':committed,'revision':revision,
                  'hits':[{'citation_id':h[0],'chunk_id':h[1],
                           'body':h[2],'source_locator':h[3],'doc_id':h[4],
                           'source_version':h[5],'cosine_distance':h[6]} for h in hits]}
    # 최종 리포트 저장 및 완료 메시지 출력
    out=save_json(HERE,'ingestion_report.json',make_final_report(rows,statuses,prepared,live))
    show_success('08 Advanced',out)

"""
매니페스트 검증과 오프라인 전처리(파싱·정제·청킹·스팬 검증)를 표준화하고, 
Live 모드에서는 임베딩 배치 생성, 문서별 분리 트랜잭션 공개, v2 롤백/회귀 검증 및 인용 포함 검색까지 일괄 연결하였다.
이를 통해 인제스트 전 과정의 데이터 무결성과 버전 전환 안정성을 최종 보고서 형태로 추적할 수 있게 구현하였다.
"""
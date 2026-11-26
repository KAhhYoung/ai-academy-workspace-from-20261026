"""02 · AIMessage.tool_calls / ToolMessage · 심화 · 실제 11/20 report_agent_lab 기반 실습."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import tasks


def main():
    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()
    if args.check:
        # 실습 진입점에서의 확인은 코드 구조만 진단합니다. 정답 코드는 학생이 작성합니다.
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    serve=args.serve
    from coursekit.db import fetch_report_status

    # 아래줄: 63번 보고서의 상태를 조회한 결과를 담는 변수
    # 보고서를 찾으면 raw에는 category_code, index_status, revision, vector_fresh, indexed_revision, embedding_model도 들어감
    '''
        2번 실습목적:
        fetch_report_status: DB에서 직접 조회한 결과
        make_request: tool calling 으로 가져온 결과
        match_result: 위 2가지의 결과를 비교: 결과적으로 가져온 내용은 같지만 구성 필드는 다름. make_request 에는 권한 정보 등도 포함. 원래는 도구상태도 포함되어야 함(ex. successful)
    '''
    raw=fetch_report_status(63)
    call=tasks.make_request('lookup_status',{'report_id':63},'call_001')
    message=tasks.match_result(call,raw)
    print('[DB]',raw)
    print('[Message]',message)
    print('[Audit]',tasks.summarize_pairs([call],[message]))

if __name__=='__main__':main()

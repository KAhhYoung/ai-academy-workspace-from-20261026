"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""
# 파일 및 디렉터리 경로 처리 기능 사용
from pathlib import Path
# 모듈 검색 경로 등록 기능 사용
import sys
# 실습 폴더와 coursekit을 포함한 상위 폴더 확인 및 검색 경로 등록
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


# 명령줄 옵션 파싱 기능 사용
import argparse

# 실습 번호별 실행 함수 조회
from coursekit.workshops import STAGE_RUNNERS

# 보고서 JSON 저장 기능 사용
from coursekit.io import save_report



# 사례별 실패 단계 판정 기능 사용
from policies.diagnosis import diagnose_stage

# 판정 단계별 개선 방법 조회 기능 사용
from policies.diagnosis import recommend_action

# 진단 근거 기록 기능 사용
from services.inspector import reason_for_stage

# 사례별 정책 상태 기록 기능 사용
from services.inspector import audit_case

# 실습 01 실패 진단 실행 함수
# - 명령줄 옵션 처리 후 고정 사례 진단 및 보고서 저장
'''
    이번 실습의 목적은,
    어느 단계에서 문제가 발생했는지 판정 (질문, 검색된 후보, 최종 답변 등)
    & 개선안 연결
'''
def main():
    # 명령줄 파서 구성 및 --offline 플래그 처리
    # - 옵션 지정 시 True, 생략 시 False
    parser=argparse.ArgumentParser(description='Day27 실습 01 · 01_failure_diagnosis')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # 실습 번호 1로 run_01 선택 및 진단·권고·근거·감사 함수 목록 전달
    report=STAGE_RUNNERS[1](args.offline,[diagnose_stage,recommend_action,reason_for_stage,audit_case])
    # 동일한 고정 진단 결과를 옵션에 따라 fixture 또는 live 접미사가 붙은 JSON 파일에 저장
    path=save_report(HERE,'day27_01_'+('fixture' if args.offline else 'live')+'.json',report)
    # 저장 경로 및 보고서 검토 안내 출력
    print('실습 01 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 파일을 직접 실행할 때 main 호출
if __name__=='__main__':main()

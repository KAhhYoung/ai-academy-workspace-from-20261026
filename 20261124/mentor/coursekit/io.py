"""실습 실행 위치와 관계없이 공통 데이터와 outputs 경로를 사용합니다."""
# 데이터 및 보고서 파일 경로 구성 기능 사용
from pathlib import Path
# JSON 읽기 및 보고서 직렬화 기능 사용
import json
ROOT = Path(__file__).resolve().parents[1]

def load_json(name):
    return json.loads((ROOT/'data'/name).read_text(encoding='utf-8'))

# 실습 보고서 저장 함수
# - outputs 폴더에 UTF-8 JSON 파일을 저장하고 파일 경로 반환
def save_report(folder, name, obj):
    # 보고서 경로 계산 및 상위 폴더 생성
    path=Path(folder)/'outputs'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    # 한글을 유지하는 들여쓰기 JSON으로 직렬화 후 UTF-8 저장
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
    # 저장 위치 출력 및 경로 반환
    print('결과 기록:',path)
    return path

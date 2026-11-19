"""
디렉터리 내 파이썬 파일 탐색 및 AST(추상 구문 트리) 기반 특정 함수 검색 유틸리티 스크립트

현재 디렉터리 및 모든 하위 폴더를 재귀적으로 순회하며, 
지정된 키워드를 포함하는 함수 정의(일반 및 비동기 함수)를 찾아 파일 경로와 라인 번호를 출력
"""
# 디렉터리 및 파일 경로 탐색 기능 사용
# - os : 운영체제 경로 조작 및 파일 시스템 순회 기능 제공
import os
# 파이썬 소스 코드 구문 분석 기능 사용
# - ast : 소스코드를 Abstract Syntax Tree(AST)로 파싱하여 노드 분석 수행
import ast

"""지정된 디렉터리 내의 모든 .py 파일을 대상으로 AST 파싱을 수행하여 특정 키워드가 포함된 함수를 검색하는 유틸리티 함수"""

# 디렉터리 내 함수 검색 메인 함수
# - current_dir : 검색을 시작할 기준 디렉터리 설정
# - exclude_dirs : 탐색에서 제외할 가상환경 및 시스템 폴더 정의
# - 카운터 변수 초기화 : 전체 파일 수 및 발견된 함수 수 누적 관리
def find_function_from_current_dir(keyword):
    # 검색 기준 폴더 설정 및 절대 경로 출력
    # - 현재 디렉터리('.')를 기준으로 하위 폴더 전체 탐색 선언
    current_dir = "."
    print(f"검색 기준 폴더: {os.path.abspath(current_dir)} (모든 하위 폴더 포함)")
    
    # 탐색 제외 대상 디렉터리 집합 정의
    # - 가상환경, 버전 관리, 캐시 등 불필요한 시스템 폴더 필터링
    exclude_dirs = {'venv', '.venv', 'env', '.git', '__pycache__', 'node_modules'}
    
    # 파일 탐색 및 함수 발견 카운터 변수 초기화
    file_count = 0
    found_count = 0

    # os.walk를 통한 디렉터리 재귀 순회
    # - 현재 디렉터리 및 모든 하위 폴더와 파일 목록 탐색
    for root, dirs, files in os.walk(current_dir):
        # 불필요한 디렉터리 원천 차단 (in-place 수정)
        # - exclude_dirs에 포함되거나 숨김 폴더('.')로 시작하는 디렉터리 탐색 제외
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        # 현재 디렉터리 내 파일 목록 순회
        for file in files:
            # 파이썬 소스 파일(.py) 필터링
            if file.endswith(".py"):
                # 탐색된 파이썬 파일 수 누적 증가
                file_count += 1
                # 파일의 절대/상대 경로 조합 생성
                filepath = os.path.join(root, file)
                
                try:
                    # 파일 소스 코드 읽기 시도 (UTF-8 인코딩 우선 적용)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            source = f.read()
                    # UnicodeDecodeError 발생 시 CP949 인코딩으로 재시도
                    except UnicodeDecodeError:
                        with open(filepath, "r", encoding="cp949") as f:
                            source = f.read()

                    # 소스 코드를 AST(추상 구문 트리)로 파싱
                    # - 파일 경로 정보를 포함하여 파싱 오류 추적 지원
                    tree = ast.parse(source, filename=filepath)
                    
                    # AST 전체 노드 순회
                    # - 트리 구조의 모든 하위 노드를 탐색하여 함수 정의 노드 색인
                    for node in ast.walk(tree):
                        # 함수 정의 노드(일반 함수 및 비동기 함수) 타입 확인
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # 함수 이름에 검색 키워드 포함 여부 검사 (대소문자 무시)
                            if keyword.lower() in node.name.lower():
                                # 키워드 매칭 성공 시 함수명, 파일 경로, 라인 번호 출력
                                print(f"[찾음] {node.name}() -> {filepath} (줄: {node.lineno})")
                                # 발견된 함수 개수 누적 증가
                                found_count += 1
                except Exception:
                    # 파일 읽기 또는 AST 파싱 중 발생하는 예외 무시 (손상된 파일 등 방어)
                    pass

    # 전체 검색 완료 결과 출력
    # - 탐색한 총 파일 수 및 매칭된 함수 총 개수 요약 표기
    print(f"\n검색 완료! 총 {file_count}개의 파일에서 {found_count}개의 함수를 찾았습니다.")

# 검색 대상 키워드 설정
# - 찾고자 하는 함수 이름 또는 키워드 정의
target_keyword = "함수_이름"

# 함수 검색 실행 호출
# - 설정된 키워드를 인자로 전달하여 검색 프로세스 구동
find_function_from_current_dir(target_keyword)

"""
파이썬의 os.walk와 AST(Abstract Syntax Tree) 분석 기능을 결합하여, 
지정된 디렉터리 내부의 모든 파이썬 소스 파일을 정밀하게 탐색하고 특정 함수 정의를 찾아내는 유틸리티 스크립트이다.
가상환경이나 숨김 폴더 등 불필요한 시스템 경로를 지능적으로 배제하고, 
UTF-8 및 CP949 다중 인코딩 방어 처리를 적용하여 파일 읽기 안정성을 높였으며, 
함수 이름 기반의 대소문자 무시 검색을 통해 개발 과정에서 필요한 함수 위치를 신속하게 역추적할 수 있도록 구현하였다.
"""
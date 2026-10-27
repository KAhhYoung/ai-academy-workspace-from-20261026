"""실습06의 K 실험을 종합 프로젝트로 옮깁니다. test는 함수 인자로도 받지 않습니다."""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from preprocessing import build_pipeline
from evaluate import regression_metrics,classification_metrics
from course_io import validate_k


""" load_data():
d.X_train   # 학습 입력: 특성 4개
d.X_valid   # 검증 입력: 특성 4개

d.yc_train  # 분류 학습 정답: failed
d.yc_valid  # 분류 검증 정답: failed

d.yr_train  # 회귀 학습 정답: elapsed_ms
d.yr_valid  # 회귀 검증 정답: elapsed_ms

d.test_ids       # 테스트 데이터의 ID만 보관
d.manifest       # 분할표
d.source_hashes  # 원본 파일의 변경 여부 확인용 해시
"""

"""
[course_io.py] validate_k(k, n_train)
[preprocessing.py] build_pipeline(estimator)
[evaluate.py] regression_metrics(y, pred)
[evaluate.py] classification_metrics(y, pred)
"""
def search_k(d,candidates=(1,3,5,9,15)):
    """task,K,scale,weights,p,metric,train_score,valid_score의 10행 표를 만든다.
    회귀/분류는 서로 다른 y와 지표를 사용한다. 조건은 standard,uniform,p=2로 고정한다.
    """
    # TODO D6-07-H4: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    

# choose_k: 분류와 회귀 각각에서 가장 좋은 K 값을 하나씩 고르는 함수
def choose_k(table):
    result={}
    for task in ['classification','regression']:
        p=table.loc[table.task==task].sort_values(['valid_score','K'],
                    ascending=[task=='regression',True],kind='stable')
        if p.empty: raise ValueError('K 탐색표에 두 문제의 결과가 모두 필요합니다.')
        result[task]=int(p.iloc[0]['K'])
    return result
